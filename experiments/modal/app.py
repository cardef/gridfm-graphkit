"""Modal smoke test for the M1 pipeline (see experiments/m1/README.md).

Scope: validate the toolchain (Julia/PowerModels via gridfm-datakit, then
gridfm_graphkit train) on Modal cheaply before committing to the full
7-grid datagen + ~40-config R010 matrix. Two jobs, gated in sequence:

  A. CPU: `gridfm_datakit generate` for case14_ieee (small, ~10k scenarios).
  B. GPU (T4): one `gridfm_graphkit train` run on the existing, already
     locally-validated experiments/m0/FLAT_PF_case14_overfit.yaml config.

Run:  modal run experiments/modal/app.py
"""

import os.path as osp

import modal

REPO_ROOT = "/root/gridfm-graphkit"
DATAKIT_ROOT = "/root/gridfm-datakit"
LOCAL_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
LOCAL_DATAKIT_ROOT = osp.abspath(osp.join(LOCAL_REPO_ROOT, "..", "gridfm-datakit"))

app = modal.App("gridfm-m1")
data_volume = modal.Volume.from_name("gridfm-m1-data", create_if_missing=True)
mlruns_volume = modal.Volume.from_name("gridfm-m1-mlruns", create_if_missing=True)

# Large/irrelevant/output dirs -- NOT derived from either repo's .gitignore:
# gridfm-datakit gitignores gridfm_datakit/grids/*.m (the pglib case files),
# which datagen needs at runtime and which exist locally despite being
# untracked -- .gitignore excludes what git shouldn't track, not what the
# image build needs, so those two lists diverge here.
#
# "experiments/modal" is excluded deliberately: none of the @app.function
# bodies read from it (they only touch experiments/m0, experiments/m1), and
# baking this launcher script itself into the image meant every edit to it
# changed the copied tree's hash, invalidating the image cache and forcing a
# full Julia/pip rebuild on the next run -- caught this after a real ~20min
# rebuild got killed by a client-side timeout mid-build.
GRAPHKIT_IGNORE = [
    ".git",
    "data",
    "mlruns",
    "__pycache__",
    ".venv",
    "*.egg-info",
    "experiments/modal",
]
DATAKIT_IGNORE = [
    ".git",
    "data",
    "data_out",
    "__pycache__",
    ".venv",
    "venv",
    "venv_pp",
    "*.egg-info",
    "build",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "build-essential")
    # Julia 1.12, matching gridfm-datakit's CI (julia-actions/setup-julia@v1
    # pins 1.12); juliaup places binaries under ~/.juliaup/bin.
    .run_commands(
        "curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12",
    )
    .env({"PATH": "/root/.juliaup/bin:$PATH"})
    .add_local_dir(LOCAL_DATAKIT_ROOT, remote_path=DATAKIT_ROOT, copy=True, ignore=DATAKIT_IGNORE)
    .add_local_dir(LOCAL_REPO_ROOT, remote_path=REPO_ROOT, copy=True, ignore=GRAPHKIT_IGNORE)
    .run_commands(
        "python -m pip install --upgrade pip",
        # Install graphkit first: it pins an exact gridfm-datakit==<version>,
        # so installing datakit editable before it gets silently clobbered
        # back to the PyPI pin (see this repo's CLAUDE.md gotcha; hit this
        # for real on the first build -- graphkit pins 1.0.5rc1, local
        # datakit checkout is 1.0.5rc2, pip uninstalled the editable one).
        # Re-installing datakit editable last restores the local checkout.
        f"pip install -e {REPO_ROOT}",
        f"pip install -e {DATAKIT_ROOT}",
        # Resolve/install/precompile the pinned Julia packages into the image
        # layer -- same line gridfm-datakit's CI uses, so the depot is baked
        # into the image (no cold-start precompile, no persistent volume
        # needed for it).
        "python -c \"from juliacall import Main as jl; "
        "jl.seval('using PowerModels, Ipopt, Memento')\"",
    )
    .env({"MLFLOW_ALLOW_FILE_STORE": "true"})
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    volumes={f"{REPO_ROOT}/data": data_volume},
    timeout=1800,
)
def datagen_case14():
    import subprocess

    subprocess.run(
        ["gridfm_datakit", "generate", "experiments/m1/datakit_configs/case14_ieee.yaml"],
        cwd=REPO_ROOT,
        check=True,
    )
    data_volume.commit()
    return "datagen case14 done"


@app.function(
    image=image,
    gpu="T4",
    volumes={f"{REPO_ROOT}/data": data_volume, f"{REPO_ROOT}/mlruns": mlruns_volume},
    timeout=1800,
)
def train_case14_smoke():
    import subprocess

    import yaml

    # FLAT_PF_case14_overfit.yaml hardcodes accelerator: cpu (it's a local
    # M0 CPU-only overfit check) -- override to actually exercise the T4.
    src = osp.join(REPO_ROOT, "experiments/m0/FLAT_PF_case14_overfit.yaml")
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["accelerator"] = "gpu"
    patched = "/tmp/flat_case14_gpu_smoke.yaml"
    with open(patched, "w") as f:
        yaml.safe_dump(cfg, f)

    subprocess.run(
        [
            "gridfm_graphkit",
            "train",
            "--config",
            patched,
            "--data_path",
            "data",
            "--exp_name",
            "modal_smoke_test",
            "--run_name",
            "flat_case14_smoke",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    mlruns_volume.commit()
    return "train smoke done"


# Memory probe: is a T4 (16GB) enough for the heaviest R010 case2000 arm
# (flat_case2000_d48, natural width h48, ~85M params, batch 4)? Cheap way to
# answer without provisioning a bigger GPU on a guess. Uses a small reduced
# scenario count -- memory depends on grid topology (case2000's real 2000
# buses) and batch/model shape, not on how many scenarios exist, so a full
# 40960-scenario datagen isn't needed just to probe memory.
#
# First attempt at 500 scenarios timed out at 1800s (~300/500 done, ~5s/
# scenario after a 101s Julia warm-up on the first one) -- case2000 per-
# scenario cost is far higher than case14's. Also large_chunk_size: 512 >
# any request here means nothing is written to disk until ALL requested
# scenarios finish, so a mid-run timeout loses everything; lowered it so
# progress survives a retry.
PROBE_SCENARIOS = 80
PROBE_CHUNK_SIZE = 16


@app.function(
    image=image,
    cpu=16,
    memory=16384,
    volumes={f"{REPO_ROOT}/data": data_volume},
    timeout=1800,
)
def datagen_case2000_probe():
    import subprocess

    import yaml

    cfg_path = osp.join(REPO_ROOT, "experiments/m1/datakit_configs/case2000_goc.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["load"]["scenarios"] = PROBE_SCENARIOS
    cfg["settings"]["large_chunk_size"] = PROBE_CHUNK_SIZE
    probe_cfg_path = "/tmp/case2000_probe_datakit.yaml"
    with open(probe_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    subprocess.run(
        ["gridfm_datakit", "generate", probe_cfg_path],
        cwd=REPO_ROOT,
        check=True,
    )
    data_volume.commit()
    return "datagen case2000 probe done"


def _run_mem_probe(bf16: bool, compile_mode: str = None):
    import math
    import time

    import yaml
    import torch

    from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (
        LitGridHeteroDataModule,
    )
    from gridfm_graphkit.io.param_handler import (
        NestedNamespace,
        get_loss_function,
        load_model,
    )

    cfg_path = osp.join(REPO_ROOT, "experiments/m1/configs/flat_case2000_d48.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    args = NestedNamespace(**cfg)

    class _T:
        is_global_zero = True
        logger = None

    dm = LitGridHeteroDataModule(args, data_dir=osp.join(REPO_ROOT, "data"))
    dm.trainer = _T()
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader())).to("cuda")
    model = load_model(args).to("cuda")
    loss_fn = get_loss_function(args)

    if bf16:
        model = model.to(torch.bfloat16)
        batch.apply(lambda t: t.to(torch.bfloat16) if t.is_floating_point() else t)

    # Mirror cli.py::main_cli's torch.compile setup exactly (same inductor
    # flags, same call site) so the probe reflects what a real R010 run
    # would actually do, not a naive torch.compile(model) call.
    if compile_mode is not None:
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        if compile_mode in ("max-autotune", "max-autotune-no-cudagraphs"):
            import torch._inductor.config as inductor_cfg

            inductor_cfg.max_autotune_gemm_backends = "ATEN,TRITON"
        model = torch.compile(model, mode=compile_mode)

    opt = torch.optim.AdamW(model.parameters(), lr=args.optimizer.learning_rate)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        opt.zero_grad()
        out = model(batch)
        ld = loss_fn(
            out,
            batch.y_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            batch.mask_dict,
            model=model,
            x_dict=batch.x_dict,
        )
        loss_val = float(ld["loss"].item())
        ld["loss"].backward()
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        result = {
            "status": "NAN" if not math.isfinite(loss_val) else "OK",
            "bf16": bf16,
            "compile_mode": compile_mode,
            "n_params": n_params,
            "batch_size": args.training.batch_size,
            "loss": loss_val,
            "peak_gb": peak / 1e9,
            "first_step_s": time.perf_counter() - t0,
        }
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        result = {"status": "OOM", "bf16": bf16, "compile_mode": compile_mode, "error": str(e)}
    print("MEM_PROBE_RESULT:", result)
    return result


@app.function(
    image=image,
    gpu="T4",
    volumes={f"{REPO_ROOT}/data": data_volume},
    timeout=900,
)
def mem_probe_flat_case2000_d48():
    return _run_mem_probe(bf16=False)


@app.function(
    image=image,
    gpu="T4",
    volumes={f"{REPO_ROOT}/data": data_volume},
    timeout=900,
)
def mem_probe_flat_case2000_d48_bf16():
    return _run_mem_probe(bf16=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={f"{REPO_ROOT}/data": data_volume},
    # max-autotune's kernel search is slow on a first call for a 48-layer
    # model -- generous timeout, not a steady-state estimate.
    timeout=2400,
)
def mem_probe_flat_case2000_d48_bf16_compile():
    return _run_mem_probe(bf16=True, compile_mode="max-autotune")


@app.local_entrypoint()
def main():
    print(datagen_case14.remote())
    print(train_case14_smoke.remote())
