# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""Case-declared per-unit normalization and topology-key attachment."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from torch_geometric.transforms import BaseTransform

from gridfm_graphkit.datasets.globals import (
    ANG_MAX,
    ANG_MIN,
    BS,
    C0_H,
    C1_H,
    C2_H,
    GS,
    MAX_PG,
    MAX_QG_H,
    MIN_PG,
    MIN_QG_H,
    PD_H,
    PG_H,
    PG_OUT,
    PG_OUT_GEN,
    P_E,
    QD_H,
    QG_H,
    QG_OUT,
    Q_E,
    RATE_A,
    VA_H,
    YFF_TT_R,
    YFT_TF_I,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError, SCHEMA_VERSION
from gridfm_graphkit.io.registries import NORMALIZERS_REGISTRY, TRANSFORM_REGISTRY


BUS_EDGE = ("bus", "connects", "bus")
_ALLOWED_TOPOLOGY_FIELDS = {
    "topology_key",
    "baseMVA",
    "provenance_group",
    "split",
    "bus_count",
    "data_hash",
    "config_sha256",
    "config_path",
    "raw_sha256",
    "integrity_status",
    "scenario_count",
    "datakit_commit",
    "size_tercile",
    "extrapolation",
    "generation_provenance_path",
    "generation_provenance_sha256",
    "integrity_failure",
}


def load_topology_manifest(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("topology manifest has the wrong schema version")
    topologies = payload.get("topologies")
    if not isinstance(topologies, dict) or not topologies:
        raise ContractError("topology manifest requires a nonempty topologies map")
    selection_freeze = payload.get("selection_freeze")
    if selection_freeze is not None:
        required = {"inventory_sha256"}
        if (
            not isinstance(selection_freeze, dict)
            or not required.issubset(
                selection_freeze,
            )
            or len(str(selection_freeze["inventory_sha256"])) != 64
        ):
            raise ContractError("invalid pre-generation selection freeze")
    keys = set()
    for network, record in topologies.items():
        if not isinstance(network, str) or not isinstance(record, dict):
            raise ContractError("invalid topology manifest entry")
        unexpected = set(record) - _ALLOWED_TOPOLOGY_FIELDS
        if unexpected:
            raise ContractError(
                f"topology {network} contains forbidden fields {sorted(unexpected)}",
            )
        required = {"topology_key", "baseMVA", "provenance_group", "split", "bus_count"}
        missing = required - set(record)
        if missing:
            raise ContractError(f"topology {network} misses {sorted(missing)}")
        if record["topology_key"] in keys:
            raise ContractError("topology_key values must be unique")
        keys.add(record["topology_key"])
        if not float(record["baseMVA"]) > 0 or int(record["bus_count"]) < 2:
            raise ContractError(f"topology {network} has invalid metadata")
        if record["split"] not in {"source", "source_dev", "target"}:
            raise ContractError(f"topology {network} has invalid split")
        if "size_tercile" in record and record["size_tercile"] not in {
            "smallest",
            "middle",
            "largest",
        }:
            raise ContractError(f"topology {network} has invalid size tercile")
        if "extrapolation" in record and not isinstance(record["extrapolation"], bool):
            raise ContractError(f"topology {network} has invalid extrapolation flag")
        if "integrity_status" in record and record["integrity_status"] not in {
            "PENDING",
            "PASS",
            "FAIL",
        }:
            raise ContractError(f"topology {network} has invalid integrity status")
    return payload


@NORMALIZERS_REGISTRY.register("CaseDeclaredMVANormalizer")
class CaseDeclaredMVANormalizer:
    """Per-unit conversion using immutable case metadata and no fitted outputs."""

    fit_strategy = "fit_on_dataset"

    def __init__(self, args):
        self.manifest_path = Path(str(args.data.topology_manifest)).resolve()
        self.manifest = load_topology_manifest(self.manifest_path)
        self.network: str | None = None
        self.topology_key: str | None = None
        self.baseMVA: float | None = None

    def set_network(self, network: str) -> None:
        try:
            record = self.manifest["topologies"][network]
        except KeyError as error:
            raise ContractError(
                f"network {network} is absent from topology manifest",
            ) from error
        self.network = network
        self.topology_key = str(record["topology_key"])
        self.baseMVA = float(record["baseMVA"])

    def to(self, device):
        return self

    def fit(self, data_path: str, scenario_ids: list[int]) -> dict:
        del data_path, scenario_ids
        if self.network is None or self.baseMVA is None or self.topology_key is None:
            raise ContractError("set_network must be called before normalizer fit")
        return {
            "network": self.network,
            "topology_key": self.topology_key,
            "baseMVA": torch.tensor(self.baseMVA, dtype=torch.float64),
            "source": "case-declared-metadata",
        }

    def fit_from_dict(self, params: dict) -> None:
        if self.network is None or self.baseMVA is None or self.topology_key is None:
            raise ContractError("set_network must be called before restoring stats")
        restored_base = float(params["baseMVA"])
        if not math.isclose(restored_base, self.baseMVA, rel_tol=0, abs_tol=1e-12):
            raise ContractError("saved baseMVA disagrees with topology manifest")
        if (
            params["network"] != self.network
            or params["topology_key"] != self.topology_key
        ):
            raise ContractError("saved normalizer identity disagrees with manifest")
        if params.get("source") != "case-declared-metadata":
            raise ContractError("normalizer provenance is not case-declared metadata")

    def _require_base(self) -> float:
        if self.baseMVA is None:
            raise ContractError("normalizer has not been bound to a network")
        return self.baseMVA

    def transform(self, data):
        base = self._require_base()
        bus = data.x_dict["bus"]
        gen = data.x_dict["gen"]
        edge = data.edge_attr_dict[BUS_EDGE]
        for column in (PD_H, QD_H, QG_H, MIN_QG_H, MAX_QG_H):
            bus[:, column] /= base
        bus[:, VA_H] *= torch.pi / 180.0
        for column in (PG_H, MIN_PG, MAX_PG):
            gen[:, column] /= base
        for column in (C0_H, C1_H, C2_H):
            gen[:, column] = torch.sign(gen[:, column]) * torch.log1p(
                torch.abs(gen[:, column]),
            )
        for column in (P_E, Q_E, RATE_A):
            edge[:, column] /= base
        edge[:, ANG_MIN] *= torch.pi / 180.0
        edge[:, ANG_MAX] *= torch.pi / 180.0

        data.y_dict["bus"][:, PD_H] /= base
        data.y_dict["bus"][:, QD_H] /= base
        data.y_dict["bus"][:, QG_H] /= base
        data.y_dict["bus"][:, VA_H] *= torch.pi / 180.0
        data.y_dict["gen"][:, PG_H] /= base
        data.baseMVA = torch.tensor(base, dtype=bus.dtype)
        data.is_normalized = torch.tensor(True, dtype=torch.bool)
        return data

    def inverse_transform(self, data):
        base = self._require_base()
        if not bool(data.is_normalized.all()):
            raise ContractError("attempting to invert unnormalized data")
        if not bool(torch.allclose(data.baseMVA, data.baseMVA.new_tensor(base))):
            raise ContractError("sample baseMVA disagrees with normalizer")
        bus = data.x_dict["bus"]
        gen = data.x_dict["gen"]
        edge = data.edge_attr_dict[BUS_EDGE]
        for column in (PD_H, QD_H, QG_H, MIN_QG_H, MAX_QG_H, GS, BS):
            bus[:, column] *= base
        for column in (PG_H, MIN_PG, MAX_PG):
            gen[:, column] *= base
        for column in (C0_H, C1_H, C2_H):
            gen[:, column] = torch.sign(gen[:, column]) * torch.expm1(
                torch.abs(gen[:, column]),
            )
        for column in (P_E, Q_E, RATE_A):
            edge[:, column] *= base
        edge[:, YFF_TT_R : YFT_TF_I + 1] *= base
        edge[:, ANG_MIN] *= 180.0 / torch.pi
        edge[:, ANG_MAX] *= 180.0 / torch.pi
        data.y_dict["bus"][:, PD_H] *= base
        data.y_dict["bus"][:, QD_H] *= base
        data.y_dict["bus"][:, QG_H] *= base
        data.y_dict["gen"][:, PG_H] *= base
        data.is_normalized = torch.tensor(False, dtype=torch.bool)
        return data

    def inverse_output(self, output, batch):
        del batch
        base = self._require_base()
        output["bus"][:, PG_OUT] *= base
        output["bus"][:, QG_OUT] *= base
        output["gen"][:, PG_OUT_GEN] *= base
        return output

    def get_stats(self) -> dict:
        if self.network is None or self.topology_key is None:
            raise ContractError("normalizer has not been bound to a network")
        return {
            "network": self.network,
            "topology_key": self.topology_key,
            "baseMVA": self._require_base(),
            "source": "case-declared-metadata",
        }


class AttachTopologyKey(BaseTransform):
    """Attach only a manifest key; geometry remains registry-owned."""

    def __init__(self, args):
        super().__init__()
        self.manifest = load_topology_manifest(
            Path(str(args.data.topology_manifest)).resolve(),
        )
        self.network: str | None = None
        self.record: dict | None = None

    def set_root(self, root: str) -> None:
        network = Path(root).name
        try:
            record = self.manifest["topologies"][network]
        except KeyError as error:
            raise ContractError(
                f"network {network} is absent from topology manifest",
            ) from error
        self.network = network
        self.record = record

    def forward(self, data):
        if self.record is None:
            raise ContractError("AttachTopologyKey.set_root was not called")
        if data.x_dict["bus"].size(0) != int(self.record["bus_count"]):
            raise ContractError("sample bus count disagrees with topology manifest")
        data.topology_key = str(self.record["topology_key"])
        return data


@TRANSFORM_REGISTRY.register("FMScalingPowerFlow")
class FMScalingPowerFlowTransforms:
    """Legacy-free PF transform composition for confirmatory data."""

    def __new__(cls, args):
        from torch_geometric.transforms import Compose

        from gridfm_graphkit.datasets.masking import AddPFHeteroMask
        from gridfm_graphkit.datasets.transforms import (
            ApplyMasking,
            RemoveInactiveBranches,
            RemoveInactiveGenerators,
        )

        if not getattr(args.data, "confirmatory", False):
            raise ContractError("FMScalingPowerFlow requires data.confirmatory=true")
        return Compose(
            [
                RemoveInactiveBranches(),
                RemoveInactiveGenerators(),
                AddPFHeteroMask(),
                ApplyMasking(args=args),
                AttachTopologyKey(args=args),
            ],
        )
