# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Audit generated parquet files and finalize their topology manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from experiments.fm_scaling.datakit_retry import (
    RETRY_POLICY,
    RETRY_STATE_FILE,
    build_retry_config,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import load_topology_manifest
from gridfm_graphkit.fm_scaling.manifest import file_sha256
from gridfm_graphkit.fm_scaling.topology import load_grid_topology, raw_data_sha256


RAW_FILES = (
    "bus_data.parquet",
    "branch_data.parquet",
    "gen_data.parquet",
    "y_bus_data.parquet",
)


def _logged_configs(path: Path) -> list[dict]:
    lines = path.read_text().splitlines(keepends=True)
    markers = [
        index
        for index, line in enumerate(lines)
        if line.startswith("New generation started at ")
    ]
    if not markers:
        raise ContractError(f"{path} has no generation records")
    payloads = []
    for marker_index, start in enumerate(markers):
        stop = markers[marker_index + 1] if marker_index + 1 < len(markers) else len(lines)
        payload = yaml.safe_load("".join(lines[start + 1 : stop]))
        if not isinstance(payload, dict):
            raise ContractError(f"{path} contains an invalid YAML configuration")
        payloads.append(payload)
    return payloads


def _assert_generation_config(config: dict, network: str, scenario_count: int) -> None:
    if config.get("network", {}).get("name") != network:
        raise ContractError(f"{network} args.log names a different network")
    if config.get("settings", {}).get("mode") != "pf":
        raise ContractError(f"{network} was not generated in PF mode")
    if config.get("settings", {}).get("include_dc_res") is not False:
        raise ContractError(f"{network} unexpectedly includes DC results")
    for field in (
        "topology_perturbation",
        "generation_perturbation",
        "admittance_perturbation",
    ):
        if config.get(field, {}).get("type") != "none":
            raise ContractError(f"{network} enables forbidden {field}")
    if int(config.get("load", {}).get("scenarios", -1)) != int(scenario_count):
        raise ContractError(f"{network} scenario count differs from manifest")


def _assert_generation_log(
    configs: list[dict],
    prepared: dict,
    network: str,
    record: dict,
    provenance: dict | None,
    raw: Path,
) -> None:
    target = int(record["scenario_count"])
    if configs[0] != prepared:
        raise ContractError(f"{network} initial args.log config differs from freeze")
    _assert_generation_config(configs[0], network, target)
    attempts = provenance.get("generation_attempts") if provenance else None
    if attempts is None:
        if len(configs) != 1:
            raise ContractError(f"{network} has unproven retry generation records")
        return
    if (
        provenance.get("retry_policy") != RETRY_POLICY
        or int(provenance.get("requested_scenario_count", -1)) != target
        or int(provenance.get("successful_scenario_count", -1)) != target
        or not isinstance(attempts, list)
        or len(attempts) != len(configs)
    ):
        raise ContractError(f"{network} retry provenance is inconsistent")

    successful = 0
    for index, (config, attempt) in enumerate(zip(configs, attempts)):
        requested = target - successful
        expected_config = (
            prepared
            if index == 0
            else build_retry_config(prepared, index, requested)
        )
        if config != expected_config:
            raise ContractError(f"{network} retry config {index} differs from policy")
        _assert_generation_config(config, network, requested)
        expected_seed = int(expected_config["settings"]["seed"])
        if (
            int(attempt.get("retry_round", -1)) != index
            or int(attempt.get("seed", -1)) != expected_seed
            or int(attempt.get("requested", -1)) != requested
            or int(attempt.get("successful_before", -1)) != successful
        ):
            raise ContractError(f"{network} retry attempt {index} is inconsistent")
        after = attempt.get("successful_after")
        if after is None or not successful <= int(after) <= successful + requested:
            raise ContractError(f"{network} retry attempt {index} has invalid progress")
        successful = int(after)
    if successful != target:
        raise ContractError(f"{network} retry policy did not complete the frozen pool")

    retry_state_path = raw / RETRY_STATE_FILE
    if (
        not retry_state_path.is_file()
        or file_sha256(retry_state_path) != provenance.get("retry_state_sha256")
        or json.loads(retry_state_path.read_text()).get("attempts") != attempts
    ):
        raise ContractError(f"{network} retry state hash or attempts mismatch")


def _scenario_ids(path: Path) -> set[int]:
    frame = pd.read_parquet(path, columns=["scenario"])
    if frame.empty or frame["scenario"].isna().any():
        raise ContractError(f"{path} has empty or null scenario IDs")
    return {int(value) for value in frame["scenario"].unique()}


def _assert_no_degenerate_outcomes(path: Path) -> None:
    frame = pd.read_parquet(path)
    required = {"scenario", "Pd", "Qd", "Vm"}
    if not required.issubset(frame.columns):
        raise ContractError(f"{path} lacks fixed outcome-integrity columns")
    bad = frame.loc[
        (frame["Pd"] == 0) & (frame["Qd"] == 0) & (frame["Vm"].abs() < 0.1),
        "scenario",
    ]
    if not bad.empty:
        raise ContractError(
            "fixed outcome-integrity rule rejects the whole topology; "
            f"degenerate scenarios={sorted(int(value) for value in bad.unique())[:10]}",
        )


def _finalize_one(
    network: str,
    record: dict,
    config_dir: Path,
    data_root: Path,
    inventory_sha256: str | None,
) -> None:
    if "scenario_count" not in record or "config_sha256" not in record:
        raise ContractError(f"{network} draft lacks preparation evidence")
    config_path = config_dir / f"{network}.yaml"
    if file_sha256(config_path) != record["config_sha256"]:
        raise ContractError(f"{network} prepared config hash mismatch")
    prepared = yaml.safe_load(config_path.read_text())
    _assert_generation_config(prepared, network, int(record["scenario_count"]))
    raw = data_root / network / "raw"
    provenance = None
    provenance_value = record.get("generation_provenance_path")
    if provenance_value:
        provenance_path = Path(str(provenance_value)).resolve()
        provenance = json.loads(provenance_path.read_text())
        if (
            provenance.get("status") != "GENERATED"
            or provenance.get("config_sha256") != record["config_sha256"]
            or provenance.get("datakit_commit") != record["datakit_commit"]
            or provenance.get("inventory_sha256") != inventory_sha256
        ):
            raise ContractError(f"{network} generation provenance mismatch")
        record["generation_provenance_sha256"] = file_sha256(provenance_path)
    missing = [name for name in RAW_FILES if not (raw / name).exists()]
    if not (raw / "args.log").is_file():
        missing.append("args.log")
    if missing:
        raise ContractError(f"{network} misses raw files {missing}")
    logged = _logged_configs(raw / "args.log")
    _assert_generation_log(logged, prepared, network, record, provenance, raw)
    expected_ids = set(range(int(record["scenario_count"])))
    for name in RAW_FILES:
        observed = _scenario_ids(raw / name)
        if observed != expected_ids:
            missing_ids = sorted(expected_ids - observed)
            extra_ids = sorted(observed - expected_ids)
            raise ContractError(
                f"{network}/{name} scenario mismatch; "
                f"missing={missing_ids[:10]}; extra={extra_ids[:10]}",
            )
    _assert_no_degenerate_outcomes(raw / "bus_data.parquet")
    topology = load_grid_topology(data_root, network, record)
    if topology.base_mva != float(record["baseMVA"]):
        raise ContractError(f"{network} baseMVA changed during topology load")
    record["raw_sha256"] = raw_data_sha256(raw)
    provenance_hash = record.get("generation_provenance_sha256", "legacy-test")
    record["data_hash"] = hashlib.sha256(
        f"{record['config_sha256']}:{record['raw_sha256']}:{provenance_hash}".encode(),
    ).hexdigest()
    record["integrity_status"] = "PASS"


def finalize(
    draft_path: Path,
    config_dir: Path,
    data_root: Path,
    output_path: Path,
    splits: set[str] | None = None,
) -> dict:
    manifest = load_topology_manifest(draft_path)
    failures = []
    inventory_sha256 = manifest.get("selection_freeze", {}).get("inventory_sha256")
    for network, record in manifest["topologies"].items():
        if splits is not None and record["split"] not in splits:
            continue
        try:
            _finalize_one(
                network,
                record,
                config_dir,
                data_root,
                inventory_sha256,
            )
        except Exception as error:
            record["integrity_status"] = "FAIL"
            record["integrity_failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            failures.append(f"{network}: {error}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    if failures:
        raise ContractError("; ".join(failures))
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", dest="splits")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    finalize(
        args.draft.resolve(),
        args.config_dir.resolve(),
        args.data_root.resolve(),
        args.output.resolve(),
        splits=set(args.splits) if args.splits else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
