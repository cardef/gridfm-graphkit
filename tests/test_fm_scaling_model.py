# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.data import Batch, HeteroData

from experiments.fm_scaling.check_gpu_compatibility import _merge_generator_q_limits
from gridfm_graphkit.datasets.globals import (
    PG_H,
    PQ_H,
    PV_H,
    REF_H,
    VA_H,
    VM_H,
)
from gridfm_graphkit.fm_scaling.contracts import ContractError
from gridfm_graphkit.fm_scaling.data import (
    CaseDeclaredMVANormalizer,
    load_topology_manifest,
)
from gridfm_graphkit.fm_scaling.model import FMScalingPF
from gridfm_graphkit.fm_scaling.registry import save_geometry_bundle
from gridfm_graphkit.fm_scaling.task import (
    FMScalingPowerFlowTask,
    graph_family_balanced_error,
)
from tests.test_fm_scaling_geometry import budget, builders, synthetic_topology


def test_gpu_probe_reproduces_generator_q_limit_merge():
    bus = pd.DataFrame(
        {
            "scenario": [0, 0, 0],
            "bus": [0, 1, 2],
            "Pd": [1.0, 2.0, 3.0],
        },
    )
    gen = pd.DataFrame(
        {
            "scenario": [0, 0, 0],
            "bus": [0, 0, 2],
            "min_q_mvar": [-2.0, -3.0, -4.0],
            "max_q_mvar": [5.0, 7.0, 11.0],
        },
    )
    merged = _merge_generator_q_limits(bus, gen)

    assert merged["min_q_mvar"].tolist() == [-5.0, 0.0, -4.0]
    assert merged["max_q_mvar"].tolist() == [12.0, 0.0, 11.0]


def _manifest(tmp_path):
    path = tmp_path / "topologies.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "fm-scaling-v1",
                "topologies": {
                    "synthetic": {
                        "topology_key": "synthetic-6",
                        "baseMVA": 100.0,
                        "provenance_group": "synthetic",
                        "split": "source",
                        "bus_count": 6,
                        "data_hash": "test",
                    },
                },
            },
            sort_keys=True,
        ),
    )
    return path


def _sample(topology_key="synthetic-6"):
    data = HeteroData()
    bus = torch.zeros((6, 15), dtype=torch.float32)
    bus[:, VM_H] = 1.0
    bus[:, VA_H] = torch.linspace(0, 0.05, 6)
    bus[:, PQ_H] = 1
    bus[0, PQ_H] = 0
    bus[0, REF_H] = 1
    bus[2, PQ_H] = 0
    bus[2, PV_H] = 1
    gen = torch.zeros((2, 6), dtype=torch.float32)
    gen[:, PG_H] = torch.tensor([0.8, 0.2])
    data["bus"].x = bus
    data["bus"].y = bus[:, :5].clone()
    data["gen"].x = gen
    data["gen"].y = gen[:, :1].clone()

    undirected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    directed = undirected + [(target, source) for source, target in undirected]
    edge_index = torch.tensor(directed, dtype=torch.long).T
    edge_attr = torch.zeros((len(directed), 10), dtype=torch.float32)
    edge_attr[:, 2] = 1.0
    edge_attr[:, 3] = -2.0
    edge_attr[:, 4] = -1.0
    edge_attr[:, 5] = 2.0
    data["bus", "connects", "bus"].edge_index = edge_index
    data["bus", "connects", "bus"].edge_attr = edge_attr
    data["gen", "connected_to", "bus"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    data["bus", "connected_to", "gen"].edge_index = torch.tensor(
        [[0, 2], [0, 1]],
        dtype=torch.long,
    )

    bus_mask = torch.zeros_like(bus, dtype=torch.bool)
    bus_mask[:, VM_H] = True
    bus_mask[:, VA_H] = True
    gen_mask = torch.zeros_like(gen, dtype=torch.bool)
    gen_mask[0, PG_H] = True
    data.mask_dict = {
        "bus": bus_mask,
        "gen": gen_mask,
        "branch": torch.zeros_like(edge_attr, dtype=torch.bool),
        "PQ": bus[:, PQ_H].bool(),
        "PV": bus[:, PV_H].bool(),
        "REF": bus[:, REF_H].bool(),
    }
    data.topology_key = topology_key
    return data


def _args(variant, manifest, geometry_bundle=None):
    return SimpleNamespace(
        task=SimpleNamespace(task_name="PowerFlow"),
        data=SimpleNamespace(
            normalization="CaseDeclaredMVANormalizer",
            confirmatory=True,
            hierarchy=SimpleNamespace(enable=False),
            topology_manifest=str(manifest),
        ),
        model=SimpleNamespace(
            communication_core=variant,
            hidden_size=16,
            edge_dim=10,
            input_bus_dim=15,
            input_gen_dim=6,
            l_pre=1,
            l_post=1,
            flat_blocks=1,
            geometry_bundle=str(geometry_bundle) if geometry_bundle else "unused",
        ),
    )


@pytest.mark.parametrize("variant", ["flat", "global", "kron", "quotient"])
def test_all_cores_share_output_schema_and_ignore_targets(tmp_path, variant):
    manifest = _manifest(tmp_path)
    kron_builder, quotient_builder = builders()
    topology = synthetic_topology()
    bundle = tmp_path / "geometry.pt"
    save_geometry_bundle(
        bundle,
        [
            kron_builder.build(topology, budget()),
            quotient_builder.build(topology, budget()),
        ],
    )
    torch.manual_seed(4)
    model = FMScalingPF(_args(variant, manifest, bundle)).eval()
    first_sample = _sample()
    second_sample = deepcopy(first_sample)
    second_sample["bus"].y = torch.randn_like(second_sample["bus"].y) * 1000
    second_sample["gen"].y = torch.randn_like(second_sample["gen"].y) * 1000

    with torch.no_grad():
        first = model(Batch.from_data_list([first_sample]))
        second = model(Batch.from_data_list([second_sample]))

    assert set(first) == {"bus", "gen"}
    assert first["bus"].shape == (6, 4)
    assert first["gen"].shape == (2, 1)
    assert torch.equal(first["bus"], second["bus"])
    assert torch.equal(first["gen"], second["gen"])


def test_case_declared_normalizer_fit_is_target_independent(tmp_path):
    manifest = _manifest(tmp_path)
    args = _args("flat", manifest)
    normalizer = CaseDeclaredMVANormalizer(args)
    normalizer.set_network("synthetic")
    first_stats = normalizer.fit("unused", [0, 1])
    first = _sample()
    second = deepcopy(first)
    second["bus"].y = torch.randn_like(second["bus"].y) * 1e6
    second["gen"].y = torch.randn_like(second["gen"].y) * 1e6

    normalizer.transform(first)
    second_normalizer = CaseDeclaredMVANormalizer(args)
    second_normalizer.set_network("synthetic")
    second_stats = second_normalizer.fit("unused", [999])
    second_normalizer.transform(second)

    assert first_stats["baseMVA"].item() == second_stats["baseMVA"].item() == 100
    assert torch.equal(first.x_dict["bus"], second.x_dict["bus"])
    assert torch.equal(first.x_dict["gen"], second.x_dict["gen"])


def test_family_balanced_metric_is_masked_wrapped_and_euclidean():
    prediction = torch.zeros((3, 4))
    target = torch.zeros((3, 5))
    mask = torch.zeros((3, 15), dtype=torch.bool)
    batch = torch.zeros(3, dtype=torch.long)
    mask[0, VM_H] = True
    mask[1, VA_H] = True
    prediction[0, 0] = 0.02
    prediction[1, 1] = 2 * torch.pi - torch.pi / 180
    # This huge unmasked error must not enter the metric.
    prediction[2, 0] = 1000.0
    vm, va, error = graph_family_balanced_error(
        prediction,
        target,
        mask,
        batch,
        1,
        vm_scale=0.01,
        va_scale=torch.pi / 180,
    )
    assert vm.item() == pytest.approx(0.02)
    assert va.item() == pytest.approx(float(torch.pi) / 180, rel=1e-5)
    assert error.item() == pytest.approx((5 / 2) ** 0.5, rel=1e-5)


def test_hierarchy_core_batches_multiple_topology_keys(tmp_path):
    manifest = _manifest(tmp_path)
    kron_builder, _ = builders()
    first_topology = synthetic_topology()
    second_topology = replace(first_topology, topology_key="synthetic-6b")
    bundle = tmp_path / "geometry.pt"
    save_geometry_bundle(
        bundle,
        [
            kron_builder.build(first_topology, budget()),
            kron_builder.build(second_topology, budget()),
        ],
    )
    model = FMScalingPF(_args("kron", manifest, bundle)).eval()
    batch = Batch.from_data_list([_sample(), _sample("synthetic-6b")])
    with torch.no_grad():
        output = model(batch)
    assert output["bus"].shape == (12, 4)
    assert output["gen"].shape == (4, 1)


def test_confirmatory_task_uses_finite_graph_balanced_objective(tmp_path):
    manifest = _manifest(tmp_path)
    args = _args("flat", manifest)
    args.task.task_name = "FMScalingPowerFlow"
    args.model.type = "FMScalingPF"
    args.data.networks = ["synthetic"]
    args.training = SimpleNamespace(
        batch_size=2,
        losses=["GraphBalancedMaskedVMVA", "GraphBalancedPBE"],
        loss_args=[SimpleNamespace(), SimpleNamespace()],
        loss_weights=[0.5, 0.5],
    )
    args.evaluation = SimpleNamespace(
        vm_scale=1.0,
        va_scale=1.0,
        run_id="S001",
        g_level="G8",
        checkpoint="smoke",
        output_path=str(tmp_path / "metrics.json"),
    )
    normalizer = CaseDeclaredMVANormalizer(args)
    normalizer.set_network("synthetic")
    task = FMScalingPowerFlowTask(args, [normalizer])
    _, losses = task.shared_step(Batch.from_data_list([_sample(), _sample()]))
    assert torch.isfinite(losses["loss"])
    assert "Graph-balanced VM MSE" in losses
    assert "Graph-balanced PBE" in losses


def test_topology_manifest_rejects_outcome_fields(tmp_path):
    manifest = _manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text())
    payload["topologies"]["synthetic"]["target_metric"] = 0.1
    manifest.write_text(yaml.safe_dump(payload))
    with pytest.raises(ContractError, match="forbidden fields"):
        load_topology_manifest(manifest)


def test_confirmatory_import_subprocess_denies_legacy_modules(tmp_path):
    manifest = _manifest(tmp_path)
    code = f"""
import importlib.abc
import sys
from types import SimpleNamespace

banned = {{
    'gridfm_graphkit.models.gnn_hetero_hier',
    'gridfm_graphkit.datasets.hierarchy',
    'gridfm_graphkit.datasets.normalizers',
}}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in banned:
            raise ImportError('forbidden confirmatory import: ' + fullname)
        return None
sys.meta_path.insert(0, Deny())
from gridfm_graphkit.fm_scaling.model import FMScalingPF
args = SimpleNamespace(
    task=SimpleNamespace(task_name='PowerFlow'),
    data=SimpleNamespace(
        normalization='CaseDeclaredMVANormalizer', confirmatory=True,
        hierarchy=SimpleNamespace(enable=False),
        topology_manifest={str(manifest)!r}),
    model=SimpleNamespace(
        communication_core='flat', hidden_size=8, edge_dim=10,
        input_bus_dim=15, input_gen_dim=6, l_pre=1, l_post=1,
        flat_blocks=1, geometry_bundle='unused'))
FMScalingPF(args)
assert not (banned & set(sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_full_task_entrypoint_denies_legacy_modules(tmp_path):
    manifest = _manifest(tmp_path)
    code = f"""
import importlib.abc
import sys
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task, get_task_transforms, load_normalizer

banned = {{
    'gridfm_graphkit.models.gnn_hetero_hier',
    'gridfm_graphkit.datasets.hierarchy',
    'gridfm_graphkit.datasets.normalizers',
}}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in banned:
            raise ImportError('forbidden confirmatory import: ' + fullname)
        return None
sys.meta_path.insert(0, Deny())
args = NestedNamespace(**{{
    'task': {{'task_name': 'FMScalingPowerFlow'}},
    'data': {{
        'normalization': 'CaseDeclaredMVANormalizer', 'confirmatory': True,
        'hierarchy': {{'enable': False}}, 'topology_manifest': {str(manifest)!r},
        'networks': ['synthetic'], 'mask_value': 0.0}},
    'model': {{
        'type': 'FMScalingPF', 'communication_core': 'flat',
        'hidden_size': 8, 'edge_dim': 10, 'input_bus_dim': 15,
        'input_gen_dim': 6, 'l_pre': 1, 'l_post': 1, 'flat_blocks': 1,
        'geometry_bundle': 'unused'}},
    'training': {{
        'batch_size': 1, 'losses': ['GraphBalancedMaskedVMVA', 'GraphBalancedPBE'],
        'loss_args': [{{}}, {{}}], 'loss_weights': [0.5, 0.5]}},
    'evaluation': {{
        'vm_scale': 1.0, 'va_scale': 1.0, 'run_id': 'S001',
        'g_level': 'G8', 'checkpoint': 'smoke', 'output_path': 'unused.json'}},
    'optimizer': {{
        'learning_rate': 1e-3, 'beta1': 0.9, 'beta2': 0.999,
        'lr_decay': 0.7, 'lr_patience': 5}},
    'seed': 0, 'verbose': False}})
normalizer = load_normalizer(args)
normalizer.set_network('synthetic')
get_task_transforms(args)
get_task(args, [normalizer])
assert not (banned & set(sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
