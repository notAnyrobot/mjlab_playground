# mjlab playground

A collection of tasks built with [mjlab](https://github.com/mujocolab/mjlab), starting with ports from [MuJoCo Playground](https://playground.mujoco.org/).

## Tasks

| Task ID | Robot | Description | Preview |
|---------|-------|-------------|---------|
| **Getup** | | | |
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/go1_getup_teaser.gif" width="200"/> |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/t1_getup_teaser.gif" width="200"/> |

## Getting Started

```bash
git clone https://github.com/mujocolab/mjlab_playground.git && cd mjlab_playground
uv sync
```

## Docker

For containerized workstation and HPC experiments, use the CUDA 12.8 development
image `mjlab-playground:cuda128-dev`. See
[`docs/docker_cuda128_workflow.md`](docs/docker_cuda128_workflow.md) for the
build, validation, transfer, and launcher workflow.

Train a task:

```bash
uv run train <task-id> --env.scene.num-envs 4096
```

Play back a trained policy from a local checkpoint:

```bash
uv run play <task-id> --checkpoint-file path/to/model.pt
```

For example:

```bash
uv run play Mjlab-Getup-Flat-Unitree-Go1 --checkpoint-file logs/rsl_rl/go1_getup/2026-06-05_12-33-19/model_700.pt
```

## Reference motion viewer

The canonical viewer command loads reference motions independently of their
presentation and selects a suitable interactive viewer automatically:

```bash
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer \
  --motion-files path/to/motion.npz
```

`--viewer auto` is the default. It selects browser-based Viser on macOS and on
Linux without a display, and native MuJoCo on Linux when a display is available.
Use `--viewer native` or `--viewer viser` to require one implementation; an
explicit mode fails clearly if it cannot start and is never silently replaced.
Native interactive playback under ordinary Python is not a supported macOS path;
use Viser there. Viser starts a local server and requires an available browser.

Pass `--record-video` during interactive playback to enable the one-shot
"Record selected clip" action (`\` in the native viewer or the Viser control).
The selected clip is recorded from its first through final frame in a background
child process while interactive pause, speed, motion selection, and playback
continue independently. Only one background recording runs at a time. Closing
the viewer waits for an active recording; a second interrupt cancels it and
removes incomplete output.

For deterministic batch recording without an interactive viewer, use:

```bash
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer \
  --motion-files path/to/motions \
  --headless --record-video
```

This records every loaded clip at its reference-motion FPS. Use `--output-dir`
with `--record-video` to override the default sibling
`renderings/<timestamp>/` directory. Run the canonical command with `--help`
for source-format, robot, device, smoke-test, and other options.

### Getup training

On a single NVIDIA 5090, the Go1 getup task converges in ~2 minutes and T1 in ~8 minutes, but we continue training with a curriculum that progressively tightens action rate, joint velocity, and power penalties to produce smoother, safer policies.

<p align="center">
  <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/training_curves.png" width="80%"/>
</p>

## Citation

If you use this repository in your research, consider citing mjlab:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

This repository is released under an [Apache-2.0 License](LICENSE).
