# Container Script Guide

These scripts run `mjlab_playground` in the `mjlab-playground:cuda128-dev`
container on either a workstation or the rootless HPC host. They are launcher
and sync helpers only; task-specific experiment commands stay in the shell or in
the docs.

## Script Inventory

- `launch_docker_ws.sh`: workstation launcher. It defaults to
  `MJLAB_PLAYGROUND_GPU_MODE=gpus` and uses Docker's native `--gpus` support.
- `launch_docker_hpc.sh`: HPC launcher. It defaults to
  `MJPG_GPU_MODE=manual` for rootless Docker and passes NVIDIA
  devices and exact driver files through by bind mount.
- `_docker_launcher_common.sh`: shared launcher implementation. Do not run this
  file directly.
- `sync_codebase.sh`: rsync helper for the `mjlab_playground` checkout on
  HPC-1. It does not sync the sibling `mjlab` checkout.
- `batch_convert_pyroki_to_gmr.sh`: converts all direct Astro PyRoki-retargeted
  AMASS splits under the mounted dataset root into sibling `gmr-astro`
  directories.
- `docker_uv_sync_locked.sh`: image-build helper used by `Dockerfile` to keep
  locked `uv sync` behavior consistent when a package mirror is configured.

## Workstation Quick Start

Build or load `mjlab-playground:cuda128-dev`, then validate the local launcher:

```bash
./scripts/launch_docker_ws.sh print-config
./scripts/launch_docker_ws.sh nvidia-smi
./scripts/launch_docker_ws.sh run list-envs --keyword Getup
```

Open an interactive shell:

```bash
./scripts/launch_docker_ws.sh shell
```

Run a command directly:

```bash
./scripts/launch_docker_ws.sh run train Mjlab-Getup-Flat-Unitree-Go1 --env.scene.num-envs 128
```

## HPC Quick Start

On HPC-1, start from the remote checkout:

```bash
cd /data/atom7/Code/mujocolab/mjlab_playground
./scripts/launch_docker_hpc.sh print-config
./scripts/launch_docker_hpc.sh nvidia-smi
./scripts/launch_docker_hpc.sh run list-envs --keyword Getup
```

The HPC launcher defaults to:

```bash
MJPG_GPU_MODE=manual
MJPG_GPUS=all
```

Manual mode avoids Docker's NVIDIA prestart hook, which fails on the validated
rootless host with a cgroup device-filter error. If another host supports normal
NVIDIA Docker, opt in explicitly:

```bash
MJPG_GPU_MODE=gpus ./scripts/launch_docker_hpc.sh nvidia-smi
```

## Image And Source Sync

Save the workstation image after local validation:

```bash
docker save mjlab-playground:cuda128-dev \
  -o /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar
```

Transfer it to HPC and load it there:

```bash
rsync -avh \
  /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar \
  atom7@192.168.24.9:/data/atom7/Data/docker_images/

ssh atom7@192.168.24.9 \
  'docker load -i /data/atom7/Data/docker_images/mjlab_playground_cuda128_dev.tar'
```

Push only `mjlab_playground` source changes:

```bash
./scripts/sync_codebase.sh push
```

Pull runtime outputs back from HPC:

```bash
./scripts/sync_codebase.sh pull-artifacts
```

Manage `/data/atom7/Code/mujocolab/mjlab` separately with Git or a separate
rsync. The launcher mounts both checkouts under `/workspace/mujocolab`.

## GPU Selection

Docker visibility and `mjlab` training device selection are separate.

Expose all GPUs to the container:

```bash
MJPG_GPUS=all ./scripts/launch_docker_hpc.sh shell
```

Expose GPU 0 only:

```bash
MJPG_GPUS=0 ./scripts/launch_docker_hpc.sh shell
```

In manual mode, comma-separated physical IDs are remapped to container-local
CUDA IDs. For example, `MJPG_GPUS=2,3` exposes those physical
devices and sets `CUDA_VISIBLE_DEVICES=0,1` inside the container.

For multi-GPU training, pass the training flag too:

```bash
./scripts/launch_docker_hpc.sh run train Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 128 \
  --gpu-ids '[0,1]' \
  --agent.logger tensorboard
```

To use every visible training GPU:

```bash
./scripts/launch_docker_hpc.sh run train Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 4096 \
  --gpu-ids all \
  --agent.logger tensorboard
```

`--gpu-ids 0,1` is not valid tyro syntax for this CLI. Use a quoted list such
as `--gpu-ids '[0,1]'`, or use `--gpu-ids all`.

## Inside The Container

The launchers prepend `/app/.venv/bin` to `PATH`, so inside the container, use `train` directly instead of `uv run train`.

```bash
train Mjlab-Getup-Flat-Unitree-Go1 \
  --env.scene.num-envs 128 \
  --gpu-ids '[0,1]' \
  --agent.logger tensorboard
```

The baked environment uses `UV_NO_SYNC=1` and
`UV_PROJECT_ENVIRONMENT=/app/.venv`; it is not intended to resync dependencies
inside a running container.

## Dataset Mount

The launchers mount the same `motion_datasets` roots used by the ProtoMotions
Newton/PyRoki container, at the same absolute path inside the container:

```bash
# Workstation default
/media/android/data/motion_datasets

# HPC default
/data/share/motion_datasets
```

The dataset mount is read-only by default, which is the expected mode for policy
learning from reference motion data. Use a writable mount only for explicit data
generation or conversion jobs:

```bash
MJPG_DATASET_READONLY=0 ./scripts/launch_docker_ws.sh shell
MJPG_DATASET_READONLY=0 ./scripts/launch_docker_hpc.sh shell
```

Override the dataset root only when the host uses a different shared path:

```bash
MJPG_DATASET_ROOT=/path/to/motion_datasets ./scripts/launch_docker_ws.sh shell
```

The longer `MJLAB_PLAYGROUND_*` environment names remain supported for existing
scripts. Prefer the shorter `MJPG_*` aliases for manual commands; if both forms
are set, `MJPG_*` wins.

## Motion Conversion

Launch the container with a writable dataset mount before generating GMR data:

```bash
MJPG_DATASET_READONLY=0 MJPG_GPUS=0 MJPG_PORT=18080 ./scripts/launch_docker_hpc.sh shell
```

Inside the container, convert every direct Astro split that contains
`pyroki-retargeted-astro`:

```bash
scripts/batch_convert_pyroki_to_gmr.sh --fps 30
```

The script writes beside the PyRoki source-of-truth directory for each split:

```text
/data/share/motion_datasets/protomotions/astro/<split>/pyroki-retargeted-astro
/data/share/motion_datasets/protomotions/astro/<split>/gmr-astro
```

It refuses to overwrite existing `.pkl` files unless requested:

```bash
scripts/batch_convert_pyroki_to_gmr.sh --fps 30 --force-remake
```

For a subset validation run, pass one or more direct child split names:

```bash
scripts/batch_convert_pyroki_to_gmr.sh --fps 30 --split dancedb --split accad
```

## Logs And Artifacts

The launchers mount the repo root, so these directories persist on the host:

- `logs/`
- `wandb/`
- `artifacts/`
- `.container_home/`
- `.container_cache/`

Use TensorBoard for unattended HPC runs unless W&B is already logged in:

```bash
--agent.logger tensorboard
```

## Troubleshooting

- If rootless Docker fails with `bpf_prog_query(BPF_CGROUP_DEVICE)`, keep
  `MJPG_GPU_MODE=manual`.
- If `/bin/bash` fails with GLIBC version errors, do not mount a full host
  library directory. The launcher should mount only exact NVIDIA files into
  `/usr/local/cuda/compat`.
- If the CUDA image prints `NVIDIA Driver was not detected` but `nvidia-smi`
  works, the warning is expected in manual passthrough mode.
- If `OpenGL.platform.ctypesloader: Failed to load libOpenGL.so.0` appears,
  no-video training can continue. Rendering or video workflows need separate
  validation.
- If W&B errors on HPC, switch to `--agent.logger tensorboard` or log in to W&B
  inside the container home.
