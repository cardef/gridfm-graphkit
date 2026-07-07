# Copyright contributors to the gridfm-graphkit project
#
# SPDX-License-Identifier: Apache-2.0

"""HELM2 unpool tail (R021) vs the reference recurrence (scipy, pilot
conventions: idea-stage/helm_unpool_pilot.py)."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from gridfm_graphkit.models.gnn_hetero_hier import HELM_C0_MIN, helm2_tail


def _random_system(n_i=24, n_b=6, seed=0):
    """Grid-like complex system: admittance rows sum to ~0 (shunt-sized
    remainder), so the harmonic extension of V_b ~ 1 p.u. gives |c0| ~ 1."""
    rng = np.random.default_rng(seed)
    off = -(0.5 + rng.random((n_i, n_i))) * np.exp(1j * rng.normal(0, 0.2, (n_i, n_i)))
    off *= rng.random((n_i, n_i)) < 0.2  # sparse-ish coupling
    np.fill_diagonal(off, 0)
    Yib = -(0.5 + rng.random((n_i, n_b))) * np.exp(1j * rng.normal(0, 0.2, (n_i, n_b)))
    Yib *= rng.random((n_i, n_b)) < 0.3
    Yib[Yib.sum(axis=1) == 0, 0] = -1.0  # every interior row sees a boundary
    Yii = off.copy()
    diag = -(off.sum(axis=1) + Yib.sum(axis=1)) + 0.05j
    np.fill_diagonal(Yii, diag)
    Vb = (1.0 + 0.05 * rng.normal(size=n_b)) * np.exp(
        1j * 0.1 * rng.normal(size=n_b),
    )
    S_I = 0.05 * (rng.normal(size=n_i) + 1j * rng.normal(size=n_i))
    return Yii, Yib, Vb, S_I


def _reference_tail(Yii, c0, cS):
    """Pilot recurrence (helm_unpool_pilot.py lines c1/c2), full precision."""
    lu = spla.splu(sp.csc_matrix(Yii))
    w0 = 1.0 / np.conj(c0)
    c1 = lu.solve(cS * w0)
    w1 = -w0 * np.conj(c1) * w0
    c2 = lu.solve(cS * w1)
    return c1 + c2


def test_helm2_tail_matches_reference():
    Yii, Yib, Vb, S_I = _random_system()
    lu_np = spla.splu(sp.csc_matrix(Yii))
    c0 = lu_np.solve(-(Yib @ Vb))
    assert np.abs(c0).min() > HELM_C0_MIN  # floor inactive on sane voltages
    cS = np.conj(S_I)
    v_aff = lu_np.solve(cS)
    ref = _reference_tail(Yii, c0, cS)

    yii_t = torch.tensor(Yii, dtype=torch.complex64)
    lu, piv = torch.linalg.lu_factor(yii_t)
    tail, canary = helm2_tail(
        lu,
        piv,
        yii_t,
        torch.tensor(c0, dtype=torch.complex64).unsqueeze(0),
        torch.tensor(v_aff, dtype=torch.complex64).unsqueeze(0),
    )
    assert tail.shape == (1, len(c0))
    err = np.abs(tail[0].numpy() - ref)
    assert err.max() < 1e-5  # complex64 vs float64 reference
    assert torch.isfinite(canary).all()
    assert float(canary.median()) < 1.0  # convergent regime on this system


def test_helm2_tail_batched_equals_per_sample():
    Yii, Yib, _, S_I = _random_system(seed=1)
    lu_np = spla.splu(sp.csc_matrix(Yii))
    cS = np.conj(S_I)
    v_aff = lu_np.solve(cS)
    rng = np.random.default_rng(2)
    yii_t = torch.tensor(Yii, dtype=torch.complex64)
    lu, piv = torch.linalg.lu_factor(yii_t)

    c0s, singles = [], []
    for _ in range(3):
        Vb = 1.0 + 0.05 * (
            rng.normal(size=Yib.shape[1]) + 1j * rng.normal(size=Yib.shape[1])
        )
        c0 = torch.tensor(lu_np.solve(-(Yib @ Vb)), dtype=torch.complex64)
        c0s.append(c0)
        va = torch.tensor(v_aff, dtype=torch.complex64)
        t, _ = helm2_tail(lu, piv, yii_t, c0.unsqueeze(0), va.unsqueeze(0))
        singles.append(t[0])
    va_b = torch.tensor(v_aff, dtype=torch.complex64).unsqueeze(0).expand(3, -1)
    batched, _ = helm2_tail(lu, piv, yii_t, torch.stack(c0s), va_b)
    assert torch.allclose(batched, torch.stack(singles), atol=1e-6)


def test_helm2_tail_backward_and_floor():
    """Gradients flow through the solves; the |c0| floor keeps them finite
    even when the coarse decoder emits near-zero voltages."""
    Yii, Yib, Vb, S_I = _random_system(seed=3)
    lu_np = spla.splu(sp.csc_matrix(Yii))
    v_aff = torch.tensor(lu_np.solve(np.conj(S_I)), dtype=torch.complex64)
    yii_t = torch.tensor(Yii, dtype=torch.complex64)
    lu, piv = torch.linalg.lu_factor(yii_t)

    for scale in (1.0, 1e-3):  # sane voltages / degenerate near-zero c0
        c0 = torch.tensor(
            scale * lu_np.solve(-(Yib @ Vb)),
            dtype=torch.complex64,
        ).unsqueeze(0)
        c0.requires_grad_(True)
        tail, _ = helm2_tail(lu, piv, yii_t, c0, v_aff.unsqueeze(0))
        tail.abs().sum().backward()
        assert torch.isfinite(c0.grad).all()
        assert torch.isfinite(tail).all()
