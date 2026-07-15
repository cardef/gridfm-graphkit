# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Strict topology, gate, configuration, and campaign manifest validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from gridfm_graphkit.fm_scaling.contracts import ContractError, GeometryBudget
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.registry import load_geometry_bundle
from gridfm_graphkit.fm_scaling.splits import validate_materialized_splits
from gridfm_graphkit.fm_scaling.topology import load_grid_topology, raw_data_sha256


CAMPAIGN_SCHEMA = "fm-scaling-campaign-v1"
GATE_SCHEMA = "fm-scaling-gate-v1"
REQUIRED_GATES = tuple(
    [f"I{index:03d}" for index in range(1, 11)]
    + [f"R{index:03d}" for index in range(1, 15)]
    + [f"C{index:03d}" for index in range(1, 4)]
    + [f"P{index:03d}" for index in range(1, 5)]
    + ["S001"],
)
GATE_EVIDENCE_KIND = {
    **{
        f"I{index:03d}": kind
        for index, kind in enumerate(
            (
                "provenance",
                "contract-tests",
                "partition-tests",
                "kron-tests",
                "quotient-tests",
                "registry-tests",
                "seam-tests",
                "data-tests",
                "training-tests",
                "compatibility-report",
            ),
            1,
        )
    },
    **{
        f"R{index:03d}": kind
        for index, kind in enumerate(
            (
                "inventory",
                "source-split",
                "geometry-selection",
                "capacity-freeze",
                "budget-calibration",
                "loss-selection",
                "dispersion-estimate",
                "power-design",
                "target-freeze",
                "runtime-bounds",
                "evaluator-smoke",
                "campaign-budget",
                "freeze-manifest",
                "authorization-review",
            ),
            1,
        )
    },
    **{f"C{index:03d}": "calibration-run" for index in range(1, 4)},
    **{f"P{index:03d}": "profile-run" for index in range(1, 5)},
    "S001": "smoke-run",
}
GATE_REQUIRED_CHECKS = {
    gate_id: (f"{gate_id.lower()}_criteria", "immutable_inputs")
    for gate_id in REQUIRED_GATES
}
GATE_REQUIRED_CHECKS["I001"] = (
    "worktree_is_clean",
    "fork_commit_is_reachable_from_origin_ref",
    "datakit_git_root_is_exact_checkout",
    "datakit_commit_matches_pin",
    "datakit_worktree_is_clean",
    "datakit_commit_is_reachable_from_origin_ref",
)
FORBIDDEN_CONFIG_TOKENS = {
    "v_aff",
    "cbus_x",
    "helm",
    "helm2",
    "GNS_hetero_hier",
    "CoarseVoltageMSE",
    "HeteroDataMVANormalizer",
    "HeteroDataPerSampleMVANormalizer",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _assert_file(root: Path, record: dict, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ContractError(f"{label} requires exactly path and sha256")
    path = _resolve(root, str(record["path"]))
    if not path.is_file():
        raise ContractError(f"missing {label} file {path}")
    observed = file_sha256(path)
    if observed != record["sha256"]:
        raise ContractError(f"{label} hash mismatch: {observed} != {record['sha256']}")
    return path


def validate_gate_evidence(path: Path, gate_id: str) -> dict:
    payload = json.loads(path.read_text())
    if gate_id == "I001":
        if payload.get("schema_version") != 1:
            raise ContractError("I001 requires the exact preflight schema")
    elif (
        payload.get("schema_version") != "fm-scaling-evidence-v1"
        or payload.get("gate_id") != gate_id
    ):
        raise ContractError(f"{gate_id} evidence has the wrong typed schema")
    if payload.get("status") != "PASS":
        raise ContractError(f"{gate_id} evidence is not PASS")
    if gate_id != "I001":
        inputs = payload.get("inputs")
        results = payload.get("results")
        if not isinstance(inputs, list) or not inputs or not isinstance(results, dict):
            raise ContractError(f"{gate_id} evidence lacks hashed inputs/results")
        resolved_inputs = []
        for record in inputs:
            if set(record) != {"path", "sha256"}:
                raise ContractError(f"{gate_id} evidence has an invalid input record")
            input_path = Path(str(record["path"]))
            if not input_path.is_absolute():
                input_path = (path.parent / input_path).resolve()
            if not input_path.is_file() or file_sha256(input_path) != record["sha256"]:
                raise ContractError(f"{gate_id} evidence input changed: {input_path}")
            resolved_inputs.append(input_path)
        if gate_id == "R003":
            selected_policy = payload.get("selected_policy")
            candidates = payload.get("candidates")
            if (
                not isinstance(selected_policy, dict)
                or not isinstance(candidates, list)
                or len(resolved_inputs) != 2
            ):
                raise ContractError("R003 evidence lacks policy candidates")
            selected_hash = GeometryBudget(**selected_policy).policy_hash
            if (
                payload.get("selected_policy_hash") != selected_hash
                or results.get("selected_policy_hash") != selected_hash
            ):
                raise ContractError("R003 selected policy hash is not derived")
            matching = [
                candidate
                for candidate in candidates
                if candidate.get("policy") == selected_policy
                and candidate.get("policy_hash") == selected_hash
                and candidate.get("status") == "PASS"
                and not candidate.get("failures")
                and candidate.get("measurements")
            ]
            if len(matching) != 1:
                raise ContractError("R003 selected policy is not a feasible candidate")
            feasible = [
                candidate
                for candidate in candidates
                if candidate.get("status") == "PASS"
            ]
            try:
                deterministic = min(
                    feasible,
                    key=lambda candidate: (
                        max(item["residual"] for item in candidate["measurements"]),
                        max(
                            item["condition_number"]
                            for item in candidate["measurements"]
                        ),
                        sum(
                            item["cross_nnz"] + item["coarse_nnz"]
                            for item in candidate["measurements"]
                        ),
                        candidate["policy_hash"],
                    ),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ContractError("R003 candidate table is incomplete") from error
            if deterministic.get("policy_hash") != selected_hash:
                raise ContractError("R003 selected policy violates the frozen rule")
            candidate_input = yaml.safe_load(resolved_inputs[1].read_text())
            if selected_policy not in candidate_input.get("candidates", []):
                raise ContractError(
                    "R003 selected policy is absent from candidate input",
                )
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ContractError(f"{gate_id} evidence has no derived checks")
    by_name = {
        str(check.get("name")): check.get("passed")
        for check in checks
        if isinstance(check, dict)
    }
    missing = [
        name for name in GATE_REQUIRED_CHECKS[gate_id] if by_name.get(name) is not True
    ]
    if missing or any(value is not True for value in by_name.values()):
        raise ContractError(f"{gate_id} evidence failed checks {missing}")
    return payload


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    core: str
    g_level: str
    seed: int
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class CampaignManifest:
    path: Path
    fork_commit: str
    upstream_commit: str
    merge_base: str
    topology_manifest: Path
    geometry_bundle: Path
    geometry_report: Path
    geometry_failures: dict[str, dict]
    split_manifest: Path
    split_root: Path
    data_root: Path
    mlflow_store: Path
    result_root: Path
    topology_payload: dict
    gates: dict[str, Path]
    runs: tuple[RunSpec, ...]


def expected_run_matrix() -> dict[str, tuple[str, str, int]]:
    matrix = {}
    run = 1
    for g_level in ("G8", "G16", "G28"):
        for core in ("flat", "global", "kron"):
            for seed in (0, 1):
                matrix[f"E{run:03d}"] = (core, g_level, seed)
                run += 1
    for seed in (0, 1):
        matrix[f"E{run:03d}"] = ("quotient", "G28", seed)
        run += 1
    return matrix


def _walk_tokens(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_tokens(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_tokens(item)
    elif isinstance(value, str):
        yield value


def validate_run_config(
    config_path: Path,
    spec: RunSpec,
    topology_payload: dict,
    failed_target_networks: set[str],
) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ContractError(f"run {spec.run_id} config is not a map")
    tokens = set(_walk_tokens(config))
    contamination = sorted(tokens & FORBIDDEN_CONFIG_TOKENS)
    if contamination:
        raise ContractError(f"run {spec.run_id} contains forbidden {contamination}")
    if config.get("seed") != spec.seed:
        raise ContractError(f"run {spec.run_id} seed mismatch")
    if config.get("task", {}).get("task_name") != "FMScalingPowerFlow":
        raise ContractError(f"run {spec.run_id} uses the wrong task")
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    if data.get("normalization") != "CaseDeclaredMVANormalizer":
        raise ContractError(f"run {spec.run_id} uses a forbidden normalizer")
    if data.get("confirmatory") is not True or data.get("hierarchy", {}).get(
        "enable",
        False,
    ):
        raise ContractError(f"run {spec.run_id} does not disable legacy hierarchy")
    if (
        model.get("type") != "FMScalingPF"
        or model.get("communication_core") != spec.core
    ):
        raise ContractError(f"run {spec.run_id} model/core mismatch")
    if training.get("losses") != ["GraphBalancedMaskedVMVA", "GraphBalancedPBE"]:
        raise ContractError(f"run {spec.run_id} objective is not masked VM/VA plus PBE")
    if len(training.get("loss_weights", [])) != 2:
        raise ContractError(f"run {spec.run_id} loss weights are incomplete")
    networks = data.get("networks")
    train_networks = data.get("train_networks")
    if (
        not isinstance(networks, list)
        or not isinstance(train_networks, list)
        or len(train_networks) != int(spec.g_level[1:])
        or not set(train_networks).issubset(networks)
    ):
        raise ContractError(f"run {spec.run_id} has the wrong source-set size")
    topologies = topology_payload["topologies"]
    for network in train_networks:
        if network not in topologies or topologies[network]["split"] != "source":
            raise ContractError(f"run {spec.run_id} reads non-source network {network}")
    expected_targets = {
        network for network, record in topologies.items() if record["split"] == "target"
    } - failed_target_networks
    if set(networks) != set(train_networks):
        raise ContractError(f"run {spec.run_id} training config can read targets")
    target_networks = data.get("target_networks")
    if set(target_networks or []) != expected_targets:
        raise ContractError(
            f"run {spec.run_id} target set differs from frozen topology manifest",
        )
    if set(data.get("failed_target_networks", [])) != failed_target_networks:
        raise ContractError(f"run {spec.run_id} omits frozen target failures")
    if len(data.get("target_scenarios", [])) != len(target_networks):
        raise ContractError(f"run {spec.run_id} target scenario vector is incomplete")
    groups = data.get("provenance_groups")
    if groups != [
        topologies[network]["provenance_group"] for network in train_networks
    ]:
        raise ContractError(
            f"run {spec.run_id} provenance groups disagree with manifest",
        )
    if not data.get("same_grid_batches") or int(data.get("samples_total", 0)) <= 0:
        raise ContractError(f"run {spec.run_id} lacks balanced exact sampling")
    group_count = len(set(groups))
    batch_size = int(training.get("batch_size", 0))
    samples_total = int(data.get("samples_total", 0))
    if group_count < 1 or batch_size < 1 or samples_total % (group_count * batch_size):
        raise ContractError(
            f"run {spec.run_id} samples_total is not exactly group/batch balanced",
        )
    flop_checkpoints = training.get("flop_checkpoints")
    if (
        not isinstance(flop_checkpoints, list)
        or len(flop_checkpoints) != 3
        or [int(value) for value in flop_checkpoints]
        != sorted({int(value) for value in flop_checkpoints})
    ):
        raise ContractError(
            f"run {spec.run_id} lacks three cumulative-FLOP checkpoints",
        )
    evaluation = config.get("evaluation", {})
    if (
        evaluation.get("run_id") != spec.run_id
        or evaluation.get("g_level") != spec.g_level
    ):
        raise ContractError(f"run {spec.run_id} evaluation identity mismatch")
    runtime_output = training.get("runtime_output_path")
    if not isinstance(runtime_output, str) or not runtime_output.endswith(
        f"{spec.run_id}-runtime.json",
    ):
        raise ContractError(f"run {spec.run_id} lacks an immutable runtime output")
    if (
        min(float(evaluation.get("vm_scale", 0)), float(evaluation.get("va_scale", 0)))
        <= 0
    ):
        raise ContractError(f"run {spec.run_id} lacks frozen metric scales")
    return config


def load_campaign_manifest(path: Path, repo_root: Path) -> CampaignManifest:
    payload = yaml.safe_load(path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CAMPAIGN_SCHEMA
    ):
        raise ContractError("campaign manifest has the wrong schema")
    topology_path = _assert_file(
        repo_root,
        payload["topology_manifest"],
        "topology manifest",
    )
    geometry_path = _assert_file(
        repo_root,
        payload["geometry_bundle"],
        "geometry bundle",
    )
    geometry_report_path = _assert_file(
        repo_root,
        payload["geometry_report"],
        "geometry report",
    )
    geometry_report = json.loads(geometry_report_path.read_text())
    if geometry_report.get("schema_version") != "fm-scaling-geometry-build-v1":
        raise ContractError("geometry report has the wrong schema")
    if geometry_report.get("geometry_bundle_sha256") != file_sha256(geometry_path):
        raise ContractError("geometry report does not bind the geometry bundle")
    geometry_failures = {
        network: item
        for network, item in geometry_report.get("topologies", {}).items()
        if item.get("status") == "FAIL"
    }
    topology_payload = load_topology_manifest(topology_path)
    if set(geometry_report.get("topologies", {})) != set(
        topology_payload["topologies"],
    ):
        raise ContractError("geometry report does not cover the topology manifest")
    non_target_failures = [
        network
        for network in geometry_failures
        if topology_payload["topologies"][network]["split"] != "target"
    ]
    if non_target_failures:
        raise ContractError(
            f"source geometry failures block training: {non_target_failures}",
        )
    data_root = _resolve(repo_root, str(payload["data_root"]))
    generation_policy_hashes = set()
    for network, record in topology_payload["topologies"].items():
        if (
            record.get("integrity_status") != "PASS"
            or not record.get("data_hash")
            or not record.get("config_path")
            or not record.get("config_sha256")
            or not record.get("raw_sha256")
            or not record.get("generation_provenance_sha256")
        ):
            raise ContractError(
                f"topology {network} lacks finalized PASS data integrity evidence",
            )
        config_path = _resolve(repo_root, str(record["config_path"]))
        if (
            not config_path.is_file()
            or file_sha256(config_path) != record["config_sha256"]
        ):
            raise ContractError(f"topology {network} datakit config hash mismatch")
        generation_config = yaml.safe_load(config_path.read_text())
        policy = {
            "load": {
                key: value
                for key, value in generation_config["load"].items()
                if key != "scenarios"
            },
            "topology_perturbation": generation_config["topology_perturbation"],
            "generation_perturbation": generation_config["generation_perturbation"],
            "admittance_perturbation": generation_config["admittance_perturbation"],
            "settings": {
                key: value
                for key, value in generation_config["settings"].items()
                if key not in {"num_processes", "data_dir", "seed"}
            },
        }
        generation_policy_hashes.add(
            hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        )
        observed_raw = raw_data_sha256(data_root / network / "raw")
        if observed_raw != record["raw_sha256"]:
            raise ContractError(f"topology {network} raw data hash mismatch")
        provenance_path = _resolve(repo_root, str(record["generation_provenance_path"]))
        if file_sha256(provenance_path) != record["generation_provenance_sha256"]:
            raise ContractError(f"topology {network} provenance hash mismatch")
        observed_data = hashlib.sha256(
            (
                f"{record['config_sha256']}:{observed_raw}:"
                f"{record['generation_provenance_sha256']}"
            ).encode(),
        ).hexdigest()
        if observed_data != record["data_hash"]:
            raise ContractError(f"topology {network} combined data hash mismatch")
    target_records = [
        record
        for record in topology_payload["topologies"].values()
        if record["split"] == "target"
    ]
    if len({record["provenance_group"] for record in target_records}) < 6:
        raise ContractError("target freeze has fewer than six provenance groups")
    tercile_counts = {
        tercile: sum(record.get("size_tercile") == tercile for record in target_records)
        for tercile in ("smallest", "middle", "largest")
    }
    if min(tercile_counts.values(), default=0) < 4:
        raise ContractError(
            f"target size terciles are not frozen with >=4 cases: {tercile_counts}",
        )
    ordered_targets = sorted(
        (
            (network, record)
            for network, record in topology_payload["topologies"].items()
            if record["split"] == "target"
        ),
        key=lambda item: (int(item[1]["bus_count"]), item[0]),
    )
    quotient, remainder = divmod(len(ordered_targets), 3)
    cursor = 0
    for index, label in enumerate(("smallest", "middle", "largest")):
        size = quotient + (index < remainder)
        for network, record in ordered_targets[cursor : cursor + size]:
            if record.get("size_tercile") != label:
                raise ContractError(f"target {network} has a non-derived size tercile")
        cursor += size
    if any("extrapolation" not in record for record in target_records):
        raise ContractError("every target requires a frozen extrapolation flag")
    source_max = max(
        int(record["bus_count"])
        for record in topology_payload["topologies"].values()
        if record["split"] == "source"
    )
    for record in target_records:
        if record["extrapolation"] != (int(record["bus_count"]) > source_max):
            raise ContractError("target extrapolation labels are not derived from size")
    extrapolative = [record for record in target_records if record["extrapolation"]]
    if (
        len(extrapolative) < 4
        or len({record["provenance_group"] for record in extrapolative}) < 2
    ):
        raise ContractError("target extrapolation subset is too small")
    frozen = topology_payload.get("target_freeze", {})
    if frozen.get("source_max_bus_count") != source_max:
        raise ContractError("target-freeze derivation metadata is missing or stale")
    datakit_commits = {
        str(record.get("datakit_commit"))
        for record in topology_payload["topologies"].values()
    }
    if len(datakit_commits) != 1 or len(next(iter(datakit_commits))) != 40:
        raise ContractError("topology manifest mixes or omits datakit commits")
    if len(generation_policy_hashes) != 1:
        raise ContractError("topology manifest mixes data-generation policies")
    geometries, _ = load_geometry_bundle(geometry_path)
    geometry_by_key = {
        (geometry.kind, geometry.topology_key): geometry for geometry in geometries
    }
    expected_geometry_keys = {
        (kind, record["topology_key"])
        for network, record in topology_payload["topologies"].items()
        if network not in geometry_failures
        for kind in ("kron", "quotient")
    }
    if set(geometry_by_key) != expected_geometry_keys:
        raise ContractError("geometry bundle does not cover every topology and arm")
    policy_hashes = {geometry.provenance.policy_hash for geometry in geometries}
    if len(policy_hashes) != 1:
        raise ContractError("geometry bundle mixes multiple frozen policies")
    for network, record in topology_payload["topologies"].items():
        if network in geometry_failures:
            continue
        topology = load_grid_topology(data_root, network, record)
        kron = geometry_by_key[("kron", record["topology_key"])]
        quotient = geometry_by_key[("quotient", record["topology_key"])]
        if (
            kron.provenance.topology_hash != topology.topology_hash
            or quotient.provenance.topology_hash != topology.topology_hash
        ):
            raise ContractError(f"geometry topology hash mismatch for {network}")
        if kron.partition != quotient.partition:
            raise ContractError(f"Kron/Quotient partition mismatch for {network}")
    split_manifest_path = _assert_file(
        repo_root,
        payload["split_manifest"],
        "split manifest",
    )
    split_root = _resolve(repo_root, str(payload["split_root"]))
    validate_materialized_splits(
        split_manifest_path,
        split_root,
        topology_payload,
    )
    analysis_files = payload.get("analysis_files")
    if not isinstance(analysis_files, list) or len(analysis_files) != 2:
        raise ContractError("campaign requires exactly two locked analysis files")
    for index, record in enumerate(analysis_files):
        _assert_file(repo_root, record, f"analysis file {index}")

    gate_payload = payload.get("gates", {})
    if set(gate_payload) != set(REQUIRED_GATES):
        missing = sorted(set(REQUIRED_GATES) - set(gate_payload))
        extra = sorted(set(gate_payload) - set(REQUIRED_GATES))
        raise ContractError(f"gate matrix mismatch; missing={missing}; extra={extra}")
    gates = {}
    r003_policy_hash = None
    for gate_id in REQUIRED_GATES:
        gate_path = _assert_file(repo_root, gate_payload[gate_id], f"gate {gate_id}")
        gate_record = json.loads(gate_path.read_text())
        if gate_record.get("gate_id", gate_record.get("run_id")) != gate_id:
            raise ContractError(f"gate {gate_id} identity mismatch")
        if gate_record.get("status") != "PASS":
            raise ContractError(f"gate {gate_id} is not PASS")
        if gate_record.get("schema_version") != GATE_SCHEMA:
            raise ContractError(f"gate {gate_id} has the wrong schema")
        if gate_record.get("fork_commit") != payload["fork_commit"]:
            raise ContractError(f"gate {gate_id} belongs to another fork commit")
        evidence = gate_record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ContractError(f"gate {gate_id} lacks hashed evidence")
        for index, evidence_record in enumerate(evidence):
            if evidence_record.get("kind") != GATE_EVIDENCE_KIND[gate_id]:
                raise ContractError(
                    f"gate {gate_id} requires evidence kind "
                    f"{GATE_EVIDENCE_KIND[gate_id]}",
                )
            _assert_file(
                repo_root,
                {key: evidence_record[key] for key in ("path", "sha256")},
                f"gate {gate_id} evidence {index}",
            )
            evidence_path = _resolve(repo_root, str(evidence_record["path"]))
            typed_evidence = validate_gate_evidence(evidence_path, gate_id)
            if gate_id == "I001" and typed_evidence["environment"].get(
                "datakit_commit",
            ) != next(iter(datakit_commits)):
                raise ContractError("I001 datakit commit differs from generated data")
            if gate_id == "R003":
                r003_policy_hash = typed_evidence["selected_policy_hash"]
        gates[gate_id] = gate_path
    if r003_policy_hash != next(iter(policy_hashes)):
        raise ContractError("R003 selected policy differs from geometry bundle")

    expected = expected_run_matrix()
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != len(expected):
        raise ContractError("campaign must contain exactly 20 explicit runs")
    runs = []
    seen = set()
    for record in raw_runs:
        required = {"run_id", "core", "g_level", "seed", "config"}
        if set(record) != required:
            raise ContractError("run records require run_id/core/g_level/seed/config")
        run_id = str(record["run_id"])
        if run_id in seen or run_id not in expected:
            raise ContractError(f"duplicate or unexpected run {run_id}")
        seen.add(run_id)
        actual = (str(record["core"]), str(record["g_level"]), int(record["seed"]))
        if actual != expected[run_id]:
            raise ContractError(f"run {run_id} differs from frozen matrix")
        config_path = _assert_file(repo_root, record["config"], f"config {run_id}")
        spec = RunSpec(
            run_id=run_id,
            core=actual[0],
            g_level=actual[1],
            seed=actual[2],
            config_path=config_path,
            config_sha256=str(record["config"]["sha256"]),
        )
        config = validate_run_config(
            config_path,
            spec,
            topology_payload,
            set(geometry_failures),
        )
        configured_split_root = _resolve(
            repo_root,
            str(config["data"]["split_from_existing_files"]),
        )
        if configured_split_root != split_root:
            raise ContractError(f"run {run_id} uses a non-frozen split root")
        runs.append(spec)
    if seen != set(expected):
        raise ContractError("campaign run IDs are incomplete")

    samples_totals = {
        yaml.safe_load(spec.config_path.read_text())["data"]["samples_total"]
        for spec in runs
    }
    if len(samples_totals) != 1:
        raise ContractError("all diversity levels must share one samples_total")
    return CampaignManifest(
        path=path.resolve(),
        fork_commit=str(payload["fork_commit"]),
        upstream_commit=str(payload["upstream_commit"]),
        merge_base=str(payload["merge_base"]),
        topology_manifest=topology_path,
        geometry_bundle=geometry_path,
        geometry_report=geometry_report_path,
        geometry_failures=geometry_failures,
        split_manifest=split_manifest_path,
        split_root=split_root,
        data_root=data_root,
        mlflow_store=_resolve(repo_root, str(payload["mlflow_store"])),
        result_root=_resolve(repo_root, str(payload["result_root"])),
        topology_payload=topology_payload,
        gates=gates,
        runs=tuple(runs),
    )
