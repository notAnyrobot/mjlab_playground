# Docker CUDA 12.8 Workflow

This guide describes the `mjlab-playground:cuda128-dev` development/training
container for local workstation experiments and remote HPC experiments.

The image provides the CUDA, Python, `uv`, PyTorch, MuJoCo, MuJoCo-Warp, and
`mjlab` dependency environment. The launchers mount the live
`mjlab_playground` checkout so task and module source edits are visible without
rebuilding the image.

## Mental Model

There are two important paths inside a running container:

```text
/app
/workspace/mjlab_playground
```

`/app` is the source snapshot copied into the image at build time. Its virtual
environment lives at:

```text
/app/.venv
```

`/workspace/mjlab_playground` is the live host checkout mounted by the launcher.
The launchers set:

```bash
PATH=/app/.venv/bin:$PATH
PYTHONPATH=/workspace/mjlab_playground/src
UV_PROJECT_ENVIRONMENT=/app/.venv
VIRTUAL_ENV=/app/.venv
UV_NO_SYNC=1
MUJOCO_GL=egl
```

Python imports `mjlab_playground` source from the mounted checkout while runtime
dependencies come from `/app/.venv`.

## What The Image Does Not Include

The image is not a test image. It is built with:

```bash
uv sync --locked --no-editable --no-dev
```

That means pytest and the rest of the dev dependency group are not installed in
the container by default. Run repository tests on the workstation with the local
development environment:

```bash
uv run pytest tests/ -v
```

## Build On The Workstation

Start from the repository root:

```bash
cd /home/android/Code/mujocolab/mjlab_playground
docker build --pull=false -t mjlab-playground:cuda128-dev .
```

The image uses:

```text
nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04
```

## Local Workstation Launcher

Print the resolved launcher configuration:

```bash
./scripts/launch_docker_ws.sh print-config
```

Open an interactive shell:

```bash
./scripts/launch_docker_ws.sh shell
```

Run a one-off command:

```bash
./scripts/launch_docker_ws.sh run list-envs --keyword Getup
```

Check GPU visibility:

```bash
./scripts/launch_docker_ws.sh nvidia-smi
```

Run training manually through the generic launcher:

```bash
./scripts/launch_docker_ws.sh run train Mjlab-Getup-Flat-Unitree-Go1 --env.scene.num-envs 4096
```

The launcher intentionally does not hardcode task-specific smoke commands.

## Local Runtime Checks Before Transfer

Run these checks before saving the image for HPC:

```bash
./scripts/launch_docker_ws.sh nvidia-smi
./scripts/launch_docker_ws.sh run list-envs --keyword Getup
./scripts/launch_docker_ws.sh run python -c "import mjlab; import mjlab_playground; import mjlab.tasks"
```

For training validation, enter the container or run a short command chosen for
the current task:

```bash
./scripts/launch_docker_ws.sh shell
train Mjlab-Getup-Flat-Unitree-Go1 --env.scene.num-envs 128
```

## Save And Transfer To HPC

Save the validated workstation image:

```bash
mkdir -p /media/android/data/docker_images
docker save mjlab-playground:cuda128-dev \
  -o /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar
```

The example HPC paths in this guide mirror the existing `mjlab` workflow:

```text
/data/atom7/Data/docker_images/mjlab_playground_cuda128_dev.tar
/data/atom7/Code/mujocolab/mjlab_playground
```

Create destination directories and transfer the image archive:

```bash
ssh hpc-2 'mkdir -p /data/atom7/Data/docker_images /data/atom7/Code/mujocolab'
rsync -avh \
  /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar \
  hpc-2:/data/atom7/Data/docker_images/
```

Use Git for the checkout when possible so the experiment can be tied to a
commit. For an ad hoc source sync, use:

```bash
rsync -avh \
  --exclude .git \
  --exclude .venv \
  --exclude .container_cache \
  --exclude .container_home \
  --exclude logs \
  --exclude wandb \
  --exclude artifacts \
  /home/android/Code/mujocolab/mjlab_playground/ \
  hpc-2:/data/atom7/Code/mujocolab/mjlab_playground/
```

## Load And Run On HPC

On the HPC host, load the image:

```bash
docker load -i /data/atom7/Data/docker_images/mjlab_playground_cuda128_dev.tar
```

From the HPC `mjlab_playground` checkout:

```bash
./scripts/launch_docker_hpc.sh print-config
./scripts/launch_docker_hpc.sh nvidia-smi
./scripts/launch_docker_hpc.sh run list-envs --keyword Getup
./scripts/launch_docker_hpc.sh shell
```

Run training manually:

```bash
./scripts/launch_docker_hpc.sh run train Mjlab-Getup-Flat-Unitree-Go1 --env.scene.num-envs 4096
```

## Rebuild Rules

Rebuild the image after changing dependency or image inputs:

- `pyproject.toml`
- `uv.lock`
- `Dockerfile`
- Python version
- CUDA base image
- system package list
- pinned `mjlab` revision
- MuJoCo, MuJoCo-Warp, PyTorch, or other runtime dependency pins

Do not rebuild for ordinary mounted source edits:

- `src/mjlab_playground/**`
- task configs
- robot assets
- docs
- training commands

If the pinned `mjlab` revision changes in `pyproject.toml`, run:

```bash
uv lock
docker build --pull=false -t mjlab-playground:cuda128-dev .
```

Then repeat workstation runtime checks, save the image, transfer it, and load it
on HPC.

## GPU Selection

Docker GPU visibility and `mjlab` training GPU selection are separate layers.

Expose all GPUs to Docker:

```bash
MJLAB_PLAYGROUND_GPUS=all ./scripts/launch_docker_hpc.sh shell
```

Expose one GPU to Docker:

```bash
MJLAB_PLAYGROUND_GPUS=device=0 ./scripts/launch_docker_hpc.sh shell
```

Inside the container, use `mjlab` training flags to choose training devices. For
multi-GPU training, pass the appropriate `--gpu-ids` value to `train`, for
example:

```bash
./scripts/launch_docker_hpc.sh run train Mjlab-Getup-Flat-Unitree-Go1 --gpu-ids all --env.scene.num-envs 4096
```

## Useful Environment Variables

Override image:

```bash
MJLAB_PLAYGROUND_IMAGE=mjlab-playground:cuda128-dev
```

Override visible GPUs:

```bash
MJLAB_PLAYGROUND_GPUS=device=0
```

Disable Docker GPU arguments:

```bash
MJLAB_PLAYGROUND_GPUS=none
```

Override published port:

```bash
MJLAB_PLAYGROUND_PORT=18080
```

Override container workdir:

```bash
MJLAB_PLAYGROUND_CONTAINER_WORKDIR=/workspace/mjlab_playground
```
