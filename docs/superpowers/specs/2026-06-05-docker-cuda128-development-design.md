# Docker CUDA 12.8 Development Environment Design

## Purpose

`mjlab_playground` needs its own container workflow for active development and
repeatable training on a local workstation and remote HPC. The image should
provide a stable CUDA, Python, MuJoCo, MuJoCo-Warp, PyTorch, and `mjlab`
dependency environment while allowing `mjlab_playground` task and module code
to change without rebuilding the image.

This is a development/training environment, not a frozen experiment artifact
and not a test image.

## Goals

- Build a `mjlab-playground:cuda128-dev` image on the workstation.
- Use `nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04` as the base image.
- Bake Python 3.13, `uv`, runtime system packages, and locked Python runtime
  dependencies into `/app/.venv`.
- Default to the pinned `mjlab` dependency declared in `pyproject.toml` and
  resolved in `uv.lock`.
- Mount the live `mjlab_playground` checkout into the container for active
  development.
- Validate the image on the workstation before saving and transferring it to
  HPC.
- Avoid dependency resolution and package downloads during normal HPC use.
- Keep launch scripts neutral so users can run arbitrary experiments inside the
  container.

## Non-Goals

- Do not bake pytest or the dev dependency group into the default image.
- Do not add task-specific launcher subcommands such as `getup-smoke` or
  `train-smoke`.
- Do not change `pyproject.toml` from the pinned `mjlab` git source to a local
  path dependency by default.
- Do not require a sibling `mjlab` checkout for normal use.
- Do not build the image independently on HPC as the normal workflow.

## Image Architecture

The Dockerfile should live at the repository root. It should mirror the proven
`mjlab` Docker pattern:

- `FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04`
- copy `uv` into the image
- install required runtime system packages, including Git and EGL support
- install managed Python 3.13
- create the locked runtime environment under `/app/.venv`
- use `uv sync --locked --no-install-project --no-editable --no-dev` for the
  dependency layer
- copy the project into `/app`
- run `uv sync --locked --no-editable --no-dev` so the image contains an
  installed `mjlab-playground` distribution and its `mjlab.tasks` entry point

Runtime environment defaults:

```bash
PATH=/app/.venv/bin:$PATH
UV_PROJECT_ENVIRONMENT=/app/.venv
VIRTUAL_ENV=/app/.venv
UV_NO_SYNC=1
MUJOCO_GL=egl
```

The image tag is a moving development tag:

```bash
mjlab-playground:cuda128-dev
```

Reproducibility comes from the git commit plus the baked `pyproject.toml` and
`uv.lock` inside the image.

## Source And Dependency Model

By default, `mjlab` comes from the pinned git source in `pyproject.toml` and the
resolved lockfile. This keeps fresh workstation and HPC checkouts portable
without requiring a sibling local `mjlab` path.

The live `mjlab_playground` source is mounted at runtime:

```bash
/workspace/mjlab_playground
```

The launcher should set:

```bash
PYTHONPATH=/workspace/mjlab_playground/src
```

This lets source edits under `src/mjlab_playground` take effect immediately
while still using the baked dependencies from `/app/.venv`.

A future optional sibling `mjlab` override can mount:

```bash
/home/android/Code/mujocolab/mjlab -> /workspace/mjlab
```

and prepend `/workspace/mjlab/src` to `PYTHONPATH`. That override should not be
required for the default workflow.

## Rebuild Rules

Rebuild the image when any dependency or image input changes:

- `pyproject.toml`
- `uv.lock`
- `Dockerfile`
- Python version
- CUDA base image
- system package list
- pinned `mjlab` revision
- MuJoCo, MuJoCo-Warp, PyTorch, or other runtime dependency pins

Do not rebuild for ordinary source-only edits while the repo is mounted:

- `src/mjlab_playground/**`
- task configs
- robot assets
- training scripts or docs that do not affect image inputs

If the optional sibling `mjlab` source override is used, pure `mjlab` source
edits do not require an immediate rebuild. Rebuild once that `mjlab` revision
should become the pinned reproducible baseline.

## Launchers

Add neutral launchers:

```text
scripts/launch_docker_ws.sh
scripts/launch_docker_hpc.sh
```

Both launchers should default to:

```bash
MJLAB_PLAYGROUND_IMAGE=mjlab-playground:cuda128-dev
MJLAB_PLAYGROUND_CONTAINER_WORKDIR=/workspace/mjlab_playground
MJLAB_PLAYGROUND_GPUS=all
MJLAB_PLAYGROUND_PORT=8080
```

Both should:

- mount the current checkout at `/workspace/mjlab_playground`
- use `/app/.venv` for the environment
- set `UV_NO_SYNC=1`
- keep `logs/`, `wandb/`, `artifacts/`, `.container_home/`, and
  `.container_cache/` on the host
- run as root in the container for simplicity
- repair ownership for common output directories after the container exits
- expose a `print-config` command showing image, mounts, workdir, GPUs, port,
  cache paths, and optional sibling `mjlab` override state

Supported commands should stay generic:

```text
shell
bash
nvidia-smi
python
run
print-config
```

Do not add task-specific command modes. Users should run task validation and
training manually with `run ...` or from `shell`.

## Workstation Workflow

Build on the workstation:

```bash
docker build --pull=false -t mjlab-playground:cuda128-dev .
```

Validate manually on the workstation before transfer:

```bash
./scripts/launch_docker_ws.sh nvidia-smi
./scripts/launch_docker_ws.sh run list-envs --keyword Getup
./scripts/launch_docker_ws.sh run python -c "import mjlab; import mjlab_playground; import mjlab.tasks"
```

For training validation, enter the shell and run a short command chosen for the
current task:

```bash
./scripts/launch_docker_ws.sh shell
train Mjlab-Getup-Flat-Unitree-Go1 --env.scene.num-envs 128
```

The exact short training command is intentionally not hardcoded into the
launcher because `mjlab_playground` will contain changing custom tasks.

## Transfer And HPC Workflow

Save the validated workstation image:

```bash
docker save mjlab-playground:cuda128-dev \
  -o /media/android/data/docker_images/mjlab_playground_cuda128_dev.tar
```

Transfer the image archive and the matching `mjlab_playground` checkout to HPC.
The normal HPC path is to load the image rather than rebuild it:

```bash
docker load -i /path/to/mjlab_playground_cuda128_dev.tar
```

Then run from the HPC checkout:

```bash
./scripts/launch_docker_hpc.sh print-config
./scripts/launch_docker_hpc.sh shell
```

The HPC launcher initially mirrors the validated `mjlab` HPC launcher model
that uses Docker `--gpus`. If the target HPC runtime requires rootless Docker or
manual `/dev/nvidia*` passthrough, that should be treated as a follow-up design
revision after observing the host constraints.

## Documentation

Add a dedicated workflow guide:

```text
docs/docker_cuda128_workflow.md
```

Add a short pointer from `README.md`.

The guide should explain:

- build, validation, save, transfer, load, and launch commands
- the image/source/runtime mental model
- rebuild rules
- how to update the pinned `mjlab` dependency
- why pytest and dev dependencies are not installed in the container image
- how to run manual validation commands inside the generic launcher
- how to use Docker GPU visibility and `mjlab` training GPU flags for
  multi-GPU experiments

## Validation Scope

The implementation should be considered valid when:

- `docker build --pull=false -t mjlab-playground:cuda128-dev .` succeeds on the
  workstation
- `./scripts/launch_docker_ws.sh print-config` reports the intended image,
  mount, workdir, env, and GPU settings
- `./scripts/launch_docker_ws.sh nvidia-smi` shows GPU visibility
- `./scripts/launch_docker_ws.sh run list-envs --keyword Getup` shows the
  `mjlab_playground` tasks
- a manual import command confirms `mjlab`, `mjlab_playground`, and
  `mjlab.tasks` import from the intended locations
- a short manual training run can be started from the container shell

HPC validation should happen after transfer and load:

- `./scripts/launch_docker_hpc.sh print-config`
- `./scripts/launch_docker_hpc.sh nvidia-smi`
- `./scripts/launch_docker_hpc.sh run list-envs --keyword Getup`

## Open Follow-Up

Optional local sibling `mjlab` mounting is intentionally left as an extension
point, not part of the default requirement. If day-to-day work needs frequent
simultaneous `mjlab` and `mjlab_playground` edits inside the same container, add
an explicit launcher flag or environment variable for that override.
