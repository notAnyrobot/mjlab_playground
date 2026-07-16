# Docker CUDA 12.8 Workflow

This guide is the end-to-end workflow for using the
`mjlab-playground:cuda128-dev` development and training image on a workstation
and on the rootless HPC host.

Validated HPC scope as of June 11, 2026:

- `mjlab-playground:cuda128-dev` exists and loads on the HPC host.
- The HPC launcher works in rootless Docker manual GPU mode.
- `nvidia-smi` works inside the container and sees the RTX 4090 GPUs.
- `list-envs` works and includes `Mjlab-Getup-Flat-Unitree-Go1`.
- A short 2-GPU Go1 getup training smoke finished successfully.

This is enough to start HPC experiments through the documented path. It is not
yet evidence for long training runs, rendering/video workflows, checkpoint
resume, or every task.

## Runtime Model

The image provides CUDA 12.8, Python, `uv`, PyTorch, MuJoCo, MuJoCo-Warp, and
the locked runtime dependency environment. The build uses the sibling local
`mjlab` checkout as a `uv` workspace member, so MuJoCo and MuJoCo-Warp policy
stays owned by `mjlab`.

There are three important paths inside a running container:

```text
/app
/workspace/mujocolab/mjlab
/workspace/mujocolab/mjlab_playground
```

`/app/.venv` is the baked virtual environment. The launchers mount the live
`mjlab` and `mjlab_playground` checkouts under `/workspace/mujocolab` and set:

```bash
PATH=/app/.venv/bin:$PATH
PYTHONPATH=/workspace/mujocolab/mjlab/src:/workspace/mujocolab/mjlab_playground/src
UV_PROJECT_ENVIRONMENT=/app/.venv
VIRTUAL_ENV=/app/.venv
UV_NO_SYNC=1
MUJOCO_GL=egl
```

Python imports source from the mounted checkouts while runtime dependencies come
from `/app/.venv`.

## What The Image Does Not Include

The image is not a test image. It is built with a locked non-dev sync:

```bash
uv sync --locked --no-editable --no-dev
```

That means pytest and other dev dependencies are not installed in the container
by default. Run repository tests on the workstation with the local development
environment:

```bash
uv run pytest tests/ -v
```

## 1. Build On The Workstation

Start from the `mjlab_playground` repository root:

```bash
cd /home/android/Code/mujocolab/mjlab_playground
DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-context mjlab=/home/android/Code/mujocolab/mjlab \
  -f Dockerfile \
  -t mjlab-playground:cuda128-dev .
```

The named `mjlab` build context must point at the local checkout whose
`pyproject.toml` owns the MuJoCo and MuJoCo-Warp dependency policy.

On a slow or unreliable route to Ubuntu, GitHub, or PyPI, pass alternate sources
for apt packages, uv-managed Python, and Python package downloads. Host-level
mirror settings on the workstation are not inherited into this BuildKit build;
pass the mirror URLs as Docker build arguments:

```bash
DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-context mjlab=/home/android/Code/mujocolab/mjlab \
  --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ \
  --build-arg UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone/ \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg UV_HTTP_TIMEOUT=3600 \
  --build-arg UV_CONCURRENT_DOWNLOADS=1 \
  -f Dockerfile \
  -t mjlab-playground:cuda128-dev .
```

`APT_MIRROR` rewrites the Ubuntu package sources used by `apt-get`, but it does
not affect `uv sync`. `UV_PYTHON_INSTALL_MIRROR` is the source for Astral Python
standalone release artifacts. When `PIP_INDEX_URL` is set, the sync helper first checks that `uv.lock` is current, then uses that URL as the default `uv` package index for dependency sync steps. `UV_HTTP_TIMEOUT` gives large CUDA wheel downloads more time on slow links. `UV_CONCURRENT_DOWNLOADS` limits simultaneous package downloads so large CUDA wheels do not compete as aggressively on weak routes.

The image uses:

```text
nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04
```

## 2. Validate On The Workstation

Print the resolved workstation launcher configuration:

```bash
./scripts/launch_docker_ws.sh print-config
```

Check GPU visibility and core CLI access:

```bash
./scripts/launch_docker_ws.sh nvidia-smi
./scripts/launch_docker_ws.sh run list-envs --keyword Getup
./scripts/launch_docker_ws.sh run python -c "import mjlab; import mjlab_playground; import mujoco_warp"
```

Open an interactive shell when you want to inspect the image:

```bash
./scripts/launch_docker_ws.sh shell
```

Run a short task-specific training command before transfer:

```bash
./scripts/launch_docker_ws.sh run train Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 128 \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.logger tensorboard \
  --agent.run-name ws_smoke
```

The launcher intentionally stays generic. It provides `shell`, `bash`,
`nvidia-smi`, `python`, `run`, `print-config`, and `help`; it does not hardcode
task-specific smoke commands.

## 3. Save And Transfer The Image

Save the validated workstation image:

```bash
mkdir -p /media/android/data/docker_images
docker save mjlab-playground:cuda128-dev \
  -o /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar
```

The current HPC paths are:

```text
workstation image tag: mjlab-playground:cuda128-dev
workstation tar:      /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar
HPC tar:              /data/atom7/Data/docker_images/mjlab_playground_cuda128_dev.tar
HPC mjlab:            /data/atom7/Code/mujocolab/mjlab
HPC playground:       /data/atom7/Code/mujocolab/mjlab_playground
```

Transfer the image archive:

```bash
ssh atom7@172.16.9.7 'mkdir -p /data/atom7/Data/docker_images /data/atom7/Code/mujocolab'
rsync -avh \
  /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar \
  atom7@172.16.9.7:/data/atom7/Data/docker_images/
```

## 4. Sync Source To HPC

Use Git for both checkouts when possible so experiments can be tied to commits.
For ad hoc iteration, use the repo helper for `mjlab_playground`:

```bash
./scripts/sync_codebase.sh push
```

The helper syncs this checkout to:

```text
atom7@172.16.9.7:/data/atom7/Code/mujocolab/mjlab_playground
```

It excludes local runtime outputs such as `logs/`, `wandb/`, `artifacts/`, and
`data/`. It uses `rsync --delete` for source files, but it does not use
`--delete-excluded`, so remote runtime outputs are protected.

Manage the sibling `mjlab` checkout separately:

```bash
rsync -avh \
  --exclude .git \
  --exclude .venv \
  --exclude .ruff_cache \
  --exclude .pytest_cache \
  /home/android/Code/mujocolab/mjlab/ \
  atom7@172.16.9.7:/data/atom7/Code/mujocolab/mjlab/
```

Datasets and local symlink layers are not part of this code sync unless you
explicitly choose to provision them.

## 5. Load And Validate On HPC

On the HPC host:

```bash
cd /data/atom7/Code/mujocolab/mjlab_playground
docker load -i /data/atom7/Data/docker_images/mjlab_playground_cuda128_dev.tar
```

Check the launcher configuration and basic runtime:

```bash
./scripts/launch_docker_hpc.sh print-config
./scripts/launch_docker_hpc.sh nvidia-smi
./scripts/launch_docker_hpc.sh run list-envs --keyword Getup
```

The HPC launcher defaults to manual GPU passthrough:

```bash
MJLAB_PLAYGROUND_GPU_MODE=manual
MJLAB_PLAYGROUND_GPUS=all
```

This rootless Docker path avoids Docker's `--gpus` NVIDIA prestart hook. The
normal runtime path previously failed on this host with:

```text
bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not permitted
```

`--runtime=nvidia` also failed there with:

```text
unknown or invalid runtime name: nvidia
```

Manual mode passes `/dev/nvidia*` devices directly. It uses host `ldconfig` to
resolve these exact NVIDIA driver libraries:

```text
libcuda.so.1
libnvidia-ml.so.1
libnvidia-ptxjitcompiler.so.1
```

It bind-mounts only those files into:

```text
/usr/local/cuda/compat
```

The launcher does not mount the whole host library directory. This matters on
hosts where NVIDIA libraries live next to host `libc.so.6`; exposing that
directory through `LD_LIBRARY_PATH` can make the container load the wrong glibc
before `/bin/bash` starts.

If `ldconfig` cannot find the driver files, point the launcher at the directory
that contains them. `MJLAB_PLAYGROUND_HPC_NVIDIA_LIBS=/lib/x86_64-linux-gnu` is
safe with the current launcher because only exact files are mounted:

```bash
MJLAB_PLAYGROUND_HPC_NVIDIA_LIBS=/lib/x86_64-linux-gnu \
  ./scripts/launch_docker_hpc.sh nvidia-smi
```

If another host supports normal NVIDIA Docker, opt back into the `--gpus` path
explicitly:

```bash
MJLAB_PLAYGROUND_GPU_MODE=gpus ./scripts/launch_docker_hpc.sh nvidia-smi
```

## 6. Run The Validated HPC Smoke

Use this as the minimal 2-GPU training check before a real experiment:

```bash
./scripts/launch_docker_hpc.sh run train Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 128 \
  --gpu-ids '[0,1]' \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.logger tensorboard \
  --agent.run-name hpc_smoke_2gpu
```

When you open `./scripts/launch_docker_hpc.sh shell`, inside the container, use `train` directly instead of `uv run train`:

```bash
train Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 128 \
  --gpu-ids '[0,1]' \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.logger tensorboard \
  --agent.run-name hpc_smoke_2gpu
```

Important CLI detail: `--gpu-ids 0,1` fails. Use a tyro list literal such as
`--gpu-ids '[0,1]'`, or use `--gpu-ids all`.

The validated run showed:

- `Warp 1.12.0 initialized`.
- Devices included `cuda:0` and `cuda:1`, both RTX 4090 47 GiB.
- Environment built with `Number of environments | 128`.
- PPO ran iterations `0/2` and `1/2`.
- `Total steps: 12288`.
- `Workers exited without errors.`
- `All workers completed successfully.`
- Run directory:
  `logs/rsl_rl/go1_getup/2026-06-11_10-25-22_hpc_smoke_2gpu`.

## 7. Run A Real HPC Experiment

Scale only after the smoke run passes. For all visible GPUs:

```bash
MJLAB_PLAYGROUND_GPUS=all ./scripts/launch_docker_hpc.sh run train \
  Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 4096 \
  --gpu-ids all \
  --agent.logger tensorboard \
  --agent.run-name hpc_go1_getup_all_gpu
```

For a fixed pair of GPUs, expose and select them explicitly:

```bash
MJLAB_PLAYGROUND_GPUS=0,1 ./scripts/launch_docker_hpc.sh run train \
  Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 4096 \
  --gpu-ids '[0,1]' \
  --agent.logger tensorboard \
  --agent.run-name hpc_go1_getup_gpu01
```

Docker GPU visibility and `mjlab` training GPU selection are separate layers:

- `MJLAB_PLAYGROUND_GPUS=all` or `0,1` controls which devices the container can
  see.
- `--gpu-ids all` or `--gpu-ids '[0,1]'` controls which devices `train` uses.

Use `--agent.logger tensorboard` for unattended HPC runs unless the container
home is already logged in to W&B.

## 8. Pull Results Back

Runtime outputs are written into the mounted HPC checkout:

```text
/data/atom7/Code/mujocolab/mjlab_playground/logs
/data/atom7/Code/mujocolab/mjlab_playground/wandb
/data/atom7/Code/mujocolab/mjlab_playground/artifacts
```

Pull them to the workstation:

```bash
./scripts/sync_codebase.sh pull-artifacts
```

`pull-logs` is an alias:

```bash
./scripts/sync_codebase.sh pull-logs
```

## Rebuild Rules

Rebuild the image after changing dependency or image inputs:

- `pyproject.toml`
- `uv.lock`
- `Dockerfile`
- Python version
- CUDA base image
- system package list
- local `mjlab` checkout dependency policy
- MuJoCo, MuJoCo-Warp, PyTorch, or other runtime dependency pins

Do not rebuild for ordinary mounted source edits:

- `src/mjlab_playground/**`
- task configs
- robot assets
- docs
- training commands

If `mjlab` dependency policy changes in the sibling checkout, run:

```bash
uv lock
DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-context mjlab=/home/android/Code/mujocolab/mjlab \
  -f Dockerfile \
  -t mjlab-playground:cuda128-dev .
```

Then repeat workstation runtime checks, save the image, transfer it, load it on
HPC, and rerun the HPC smoke.

## Useful Environment Variables

Override image:

```bash
MJLAB_PLAYGROUND_IMAGE=mjlab-playground:cuda128-dev
```

Override visible GPUs:

```bash
MJLAB_PLAYGROUND_GPUS=0
MJLAB_PLAYGROUND_GPUS=all
```

Disable Docker GPU arguments:

```bash
MJLAB_PLAYGROUND_GPUS=none
```

Override the GPU runtime mode:

```bash
MJLAB_PLAYGROUND_GPU_MODE=manual
MJLAB_PLAYGROUND_GPU_MODE=gpus
MJLAB_PLAYGROUND_GPU_MODE=none
```

Override rootless-HPC NVIDIA paths when auto-discovery is not enough:

```bash
MJLAB_PLAYGROUND_HPC_DEV_DIR=/dev
MJLAB_PLAYGROUND_HPC_NVIDIA_LIBS=/lib/x86_64-linux-gnu
MJLAB_PLAYGROUND_HPC_NVIDIA_SMI=/usr/bin/nvidia-smi
```

Override published port:

```bash
MJLAB_PLAYGROUND_PORT=18080
```

Override container workdir:

```bash
MJLAB_PLAYGROUND_CONTAINER_WORKDIR=/workspace/mujocolab/mjlab_playground
```

Override the host `mjlab` checkout:

```bash
MJLAB_REPO_ROOT=/data/atom7/Code/mujocolab/mjlab
```

## Troubleshooting

If Docker startup fails with the cgroup device-filter error, stay on
`MJLAB_PLAYGROUND_GPU_MODE=manual`. That is the validated rootless Docker path
for this host.

If the CUDA base image prints `NVIDIA Driver was not detected` under manual
passthrough, check `./scripts/launch_docker_hpc.sh nvidia-smi`. The warning is
not blocking when `nvidia-smi` and training work.

If this warning appears:

```text
OpenGL.platform.ctypesloader: Failed to load libOpenGL.so.0
```

No-video training can continue. Rendering, video recording, or viewer workflows
need separate validation.

If this warning appears:

```text
CUDA peer access: Not supported
```

The validated smoke can still be correct. Treat it as a potential performance
limitation for multi-GPU scaling, not as an automatic correctness failure.

If W&B fails because the HPC container is not logged in, use:

```bash
--agent.logger tensorboard
```

If `/bin/bash` fails with GLIBC version errors at container startup, check that
you are not mounting a whole host library directory into `LD_LIBRARY_PATH`. The
launcher should mount only exact NVIDIA driver files into
`/usr/local/cuda/compat`.
