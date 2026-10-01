"""Checks that fail if the symmetry actions or the canonical wrapper are wrong.

Uses the repository test fixture (tests/data/case14_ieee, 72 scenarios).
"""

import math
import os

import pytest
import torch
import yaml
from torch_geometric.loader import DataLoader

from gridfm_graphkit.datasets.globals import (
    PG_H,
    QG_H,
    TAP,
    VA_H,
    VM_H,
    YFF_TT_I,
    YFF_TT_R,
    YFT_TF_I,
    YFT_TF_R,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task_transforms,
    load_model,
    load_normalizer,
)
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)
from gridfm_graphkit.tasks.pf_task import (
    _build_bus_target,
    _clamp_known_to_ground_truth,
)

from experiments.ac_pf_symmetries.symmetries import (
    Canonicalize,
    CovariantAugment,
    act,
    act_output,
    act_phase,
    equivariance_error,
    line_pairs,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIG = os.path.join(ROOT, "tests/config/datamodule_test_base_config.yaml")
DATA = os.path.join(ROOT, "tests/data/case14_ieee")


@pytest.fixture(scope="module")
def args():
    return NestedNamespace(**yaml.safe_load(open(CONFIG)))


@pytest.fixture(scope="module")
def loader(args):
    normalizer = load_normalizer(args)
    ds = HeteroGridDatasetDisk(DATA, normalizer, transform=get_task_transforms(args))
    normalizer.fit(os.path.join(DATA, "raw"), list(range(len(ds))))
    return DataLoader(ds, batch_size=8, shuffle=False)


def ground_truth_residual(data):
    """Power-balance residual of the labelled state, in the framework's own physics."""
    target, _, _ = _build_bus_target(data, data["bus"].x.size(0))
    ei, attr = data[E_KEY].edge_index, data[E_KEY].edge_attr
    pft, qft = ComputeBranchFlow()(target, ei, attr)
    p_in, q_in = ComputeNodeInjection()(pft, qft, ei, target.size(0))
    res_p, res_q = ComputeNodeResiduals()(p_in, q_in, target, data["bus"].x)
    return torch.stack([res_p, res_q], 1)


E_KEY = ("bus", "connects", "bus")


def state_residual(out, data):
    """Mean |(rP, rQ)| over buses of a predicted state, in the framework's own physics (run.py's PBE)."""
    n = data["bus"].x.size(0)
    target, gen_to_bus, _ = _build_bus_target(data, n)
    ev = _clamp_known_to_ground_truth(out["bus"], target, data, gen_to_bus, n)
    ei, attr = data[E_KEY].edge_index, data[E_KEY].edge_attr
    pft, qft = ComputeBranchFlow()(ev, ei, attr)
    p_in, q_in = ComputeNodeInjection()(pft, qft, ei, n)
    res_p, res_q = ComputeNodeResiduals()(p_in, q_in, ev, data["bus"].x)
    return torch.sqrt(res_p**2 + res_q**2).mean()


def perturbed(data):
    """A wrong state (residuals O(0.1)), so invariance is tested on non-trivial numbers."""
    d = data.clone()
    n = d["bus"].y.size(0)
    d["bus"].y[:, VM_H] += 0.05 * torch.sin(torch.arange(n, dtype=torch.float))
    d["bus"].y[:, VA_H] += 0.10 * torch.cos(torch.arange(n, dtype=torch.float))
    return d


def test_labelled_state_is_a_pf_solution(loader):
    data = next(iter(loader))
    assert ground_truth_residual(data).abs().max() < 1e-3
    assert ground_truth_residual(perturbed(data)).abs().max() > 1e-2


@pytest.mark.parametrize(
    "sym,param,factor",
    [
        ("phase", 0.7, 1.0),
        ("scale", 4.0, 1 / 4.0),
        ("flip", 0.5, 1.0),
        ("fliptrafo", 1.0, 1.0),
    ],
)
def test_physics_is_invariant_under_action(loader, sym, param, factor):
    data = perturbed(next(iter(loader)))
    base = ground_truth_residual(data)
    transformed = ground_truth_residual(act(data, sym, param))
    assert torch.allclose(transformed, base * factor, atol=1e-6, rtol=1e-4)


def test_scale_action_rescales_labels_only_where_expected(loader):
    data = next(iter(loader))
    d = act(data, "scale", 2.0)
    assert torch.allclose(d["bus"].y[:, [VM_H, VA_H]], data["bus"].y[:, [VM_H, VA_H]])
    assert torch.allclose(d["bus"].y[:, QG_H], data["bus"].y[:, QG_H] / 2)
    assert torch.allclose(d["gen"].y[:, PG_H], data["gen"].y[:, PG_H] / 2)


def test_transformer_flip_is_the_reversed_device(loader):
    """Row swap + tap -> 1/tap is the MATPOWER reversal (1/tau, -phi, tau^2 z, b/tau^2), not (1/tau, -phi)."""
    data = next(iter(loader))
    _, fwd, line = line_pairs(data)
    t = fwd & ~line
    assert t.any(), "fixture has no transformer"

    def y(attr, re, im):
        return torch.complex(attr[t, re].double(), attr[t, im].double())

    a = data[E_KEY].edge_attr
    tau = a[t, TAP].double()
    ys = -tau * y(a, YFT_TF_R, YFT_TF_I)  # Yft = -ys / tau (no phase shift in the data)
    shunt = tau**2 * y(a, YFF_TT_R, YFF_TT_I) - ys  # Yff = (ys + j b/2) / tau^2
    tau_r, ys_r, shunt_r = 1 / tau, ys / tau**2, shunt / tau**2  # the reversed device
    b = act(data, "fliptrafo", 1.0)[E_KEY].edge_attr
    assert torch.allclose(b[t, TAP].double(), tau_r, rtol=1e-6)
    assert torch.allclose(
        y(b, YFF_TT_R, YFF_TT_I),
        (ys_r + shunt_r) / tau_r**2,
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(y(b, YFT_TF_R, YFT_TF_I), -ys_r / tau_r, rtol=1e-5, atol=1e-5)
    naive = (
        ys + shunt
    ) / tau_r**2  # (1/tau, -phi) without referring z and b through the tap
    assert not torch.allclose(y(b, YFF_TT_R, YFF_TT_I), naive, rtol=1e-3)


def test_phase_group_law(loader):
    data = next(iter(loader))
    back = act_phase(act_phase(data, 1.3), -1.3)
    assert torch.allclose(back["bus"].x, data["bus"].x, atol=1e-6)
    assert torch.allclose(back["bus"].y, data["bus"].y, atol=1e-6)


def test_output_action_matches_label_action(loader):
    data = next(iter(loader))
    target, _, _ = _build_bus_target(data, data["bus"].x.size(0))
    for sym, param in (("phase", 0.4), ("scale", 3.0)):
        d = act(data, sym, param)
        target_g, _, _ = _build_bus_target(d, d["bus"].x.size(0))
        rho_target = act_output({"bus": target}, sym, param, data)["bus"]
        assert torch.allclose(target_g, rho_target, atol=1e-6, rtol=1e-5)


def test_bare_model_breaks_phase_scale_and_trafo_flip_but_not_line_flip(loader, args):
    torch.manual_seed(0)
    model = load_model(args).eval()
    assert equivariance_error(model, loader, "phase", math.pi / 2)["VA"]["rel"] > 0.1
    assert equivariance_error(model, loader, "scale", 10.0)["VM"]["rel"] > 1e-3
    assert equivariance_error(model, loader, "flip", 0.5)["VA"]["rel"] < 1e-4
    # the tap column is the only orientation-dependent input (measured: VM rel 1.4e-2)
    assert equivariance_error(model, loader, "fliptrafo", 1.0)["VM"]["rel"] > 1e-3


def test_canonical_model_is_exactly_equivariant(loader, args):
    torch.manual_seed(0)
    model = Canonicalize(load_model(args)).fit_scale_ref(loader).eval()
    probes = (
        ("phase", math.pi / 2),
        ("scale", 10.0),
        ("scale", 0.01),
        ("fliptrafo", 1.0),
    )
    for sym, param in probes:
        ee = equivariance_error(model, loader, sym, param)
        assert all(v["rel"] < 1e-4 for v in ee.values()), (sym, ee)


@torch.no_grad()
def test_covariant_augment_evaluates_the_loss_in_the_sample_frame(loader, args):
    """Training mode: the rescaled physics residual equals the physics of the de-augmented prediction on the
    original sample (fails by a factor k^2 if the residual were divided by k, or if rho(g) were not undone)."""
    torch.manual_seed(0)
    model = load_model(args)
    wrapped = CovariantAugment(model).train()
    model.eval()  # keep the augmentation, drop any train-only behaviour of the inner model
    data = next(iter(loader))
    wrapped.draw = lambda n: (torch.linspace(-2.0, 2.0, n), 7.0)
    out = wrapped(data)
    last = max(wrapped.layer_residuals)
    assert math.isclose(
        float(wrapped.layer_residuals[last]),
        float(state_residual(out, data)),
        rel_tol=1e-4,
    )
    wrapped.eval()
    assert torch.allclose(wrapped(data)["bus"], model(data)["bus"])
