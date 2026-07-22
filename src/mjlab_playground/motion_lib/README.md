# Motion Library

`mjlab_playground.motion_lib` turns robot-specific reference motion into data that
can be inspected, serialized, sampled, and queried by motion-imitation systems.
The current pipeline targets the Astro robot and uses PyRoki-retargeted motion as
its source format.

Use this module when you need to:

- load PyRoki `.npz` clips as canonical `ReferenceMotion` values;
- resample generalized-coordinate trajectories to a target FPS;
- enrich trajectories with MuJoCo-derived body poses and velocities;
- publish versioned, device-neutral reference motion `.npz` artifacts;
- inspect or record motions with native MuJoCo or browser-based Viser; or
- sample clip IDs and clip-local times for AMP- or mimic-style consumers.

For architecture, class contracts, extension seams, and the test map, see the
[motion library maintainer reference](../../../docs/motion_lib.md).

## Pipeline at a glance

```text
PyRoki source clips
    -> MotionLib.load()
    -> MotionLib.resample()
    -> MotionLib.enrich()
    -> ReferenceMotion.from_clips()
    -> one versioned reference motion artifact
    -> viewer, sampler, or training consumer
```

`ReferenceMotion` is the canonical value throughout the pipeline. Source clips
contain generalized coordinates. Rich clips additionally contain derived body
state. Versioned artifacts preserve rich tensors plus clip and axis metadata.

## Quick start

Run commands from the repository root after completing the
[project setup](../../../README.md#getting-started).

### 1. Build a versioned reference motion artifact

This example loads one bundled PyRoki clip, resamples it from 30 Hz to 50 Hz,
enriches it with the Astro MuJoCo model, performs one-clip reference motion
assembly, and publishes one versioned artifact:

```bash
uv run python -m mjlab_playground.motion_lib.scripts.run_motion_lib \
  --motion-files assets/motions/astro/sfu/pyroki-retargeted-astro/0005_0005_Jogging001_poses_keypoints_retargeted.npz \
  --format pyroki \
  --source-fps 30 \
  --output-fps 50 \
  --robot astro \
  --device cpu \
  --output-file /tmp/astro-jogging-reference-motion.npz
```

The command prints the path it writes. Existing output files are protected by
default; pass `--overwrite` only when replacement is intentional.

### 2. Inspect the artifact

```bash
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer \
  --motion-files /tmp/astro-jogging-reference-motion.npz \
  --format mjlab \
  --fps 50 \
  --robot astro \
  --viewer auto
```

`--viewer auto` selects Viser on macOS and on Linux without a display. It
selects native MuJoCo on Linux when a display is available. Use `--smoke-test`
instead of launching an interactive viewer when you only need to validate that
the motion can be loaded and applied to the robot scene.

### 3. Build and use the complete Astro SFU artifact

Build all 35 trusted direct-child PyRoki clips in sorted filename order and
publish one assembled artifact:

```bash
uv run python -m mjlab_playground.motion_lib.scripts.run_motion_lib \
  --motion-files assets/motions/astro/sfu/pyroki-retargeted-astro \
  --format pyroki \
  --source-fps 30 \
  --output-fps 50 \
  --robot astro \
  --device cpu \
  --output-file assets/motions/astro/sfu/mjlab-astro.npz
```

Publication is atomic and no-clobber by default. Add `--overwrite` only for an
intentional atomic replacement. The runner publishes no intermediate one-clip
artifacts; the trusted PyRoki inputs and final versioned reference motion
artifact are the durable inputs and output.

The verified `mjlab-astro.npz` artifact uses schema v1 and contains 35 clips,
39,225 packed frames at 50 FPS, 29 DOFs, and 31 bodies. Loading it produces one
packed `ReferenceMotion`; iterate its logical clips without unpacking or copying
its frame tensors:

```python
from mjlab_playground.motion_lib.motion_loader import MotionLoader

motions = MotionLoader.load(
    "assets/motions/astro/sfu/mjlab-astro.npz",
    motion_format="mjlab",
    fps=50,
    device="cpu",
)
package, = motions
clips = tuple(package.iter_clip_spans())

assert len(clips) == 35
first_frame = package.get_frame(clips[0].to_packed_frame(0))
```

Browse the package interactively as 35 clip-local selections:

```bash
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer \
  --motion-files assets/motions/astro/sfu/mjlab-astro.npz \
  --format mjlab \
  --fps 50 \
  --robot astro \
  --device cpu \
  --viewer auto
```

The status starts at `Clip 1/35`. Use Previous/Next to select clips with
circular wrapping. Selection resets to the clip's first frame; pause, speed,
progress, and looping remain clip-local and never cross a packed clip boundary.

Add `--record-video --output-dir <FRESH_OUTPUT_DIRECTORY>` to enable the
interactive **Record selected clip** action. It records the selected full clip
in a background child while interactive playback remains usable. To record the
entire package deterministically without opening a viewer, add `--headless`;
public headless mode emits one video per logical clip, in stored clip order.
Use synthetic automated coverage for whole-package regression checks rather
than routinely rendering all 35 real videos.

## Command-line tools

### `run_motion_lib`

Runs the complete source-to-artifact workflow: load source clips, transform each
clip through `MotionLib`, perform reference motion assembly, and atomically
publish one versioned reference motion artifact.

```bash
uv run python -m mjlab_playground.motion_lib.scripts.run_motion_lib --help
```

Important behavior:

- Inputs are one `.npz` file or a flat directory of direct-child `.npz` files.
- PyRoki is the implemented source format in v1; `proto` is reserved.
- Astro is the only implemented robot in v1.
- `--output-file` is required and names the singular published artifact.
- Publication is no-clobber by default; `--overwrite` explicitly enables atomic
  replacement.
- The runner creates no intermediate one-clip artifacts.

### `launch_motion_viewer`

Loads PyRoki or `mjlab` reference motions and presents them through the same
source-agnostic playback session.

```bash
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer --help
```

Common modes:

```bash
# Interactive playback.
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer \
  --motion-files <MOTION_FILE_OR_DIRECTORY> \
  --format mjlab \
  --viewer auto

# Validate loading and one-frame scene application without opening a viewer.
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer \
  --motion-files <MOTION_FILE_OR_DIRECTORY> \
  --format mjlab \
  --smoke-test

# Record every loaded clip without an interactive viewer.
uv run python -m mjlab_playground.motion_lib.scripts.launch_motion_viewer \
  --motion-files <MOTION_FILE_OR_DIRECTORY> \
  --format mjlab \
  --record-video \
  --headless \
  --output-dir <OUTPUT_DIRECTORY>
```

During interactive playback, `--record-video` enables one-shot background
recording of the selected full clip while playback continues. Public headless
mode records every logical clip; there is no public single-clip headless
selector. `--viewer` belongs to this command, not to `run_motion_lib`.

### `convert_pyroki_to_gmr`

Converts PyRoki `.npz` data to the GMR `.pkl` compatibility format. This is a
format conversion utility; GMR is not a `MotionLib` source format.

```bash
uv run python -m mjlab_playground.motion_lib.scripts.convert_pyroki_to_gmr \
  --input assets/motions/astro/sfu/pyroki-retargeted-astro/0005_0005_Jogging001_poses_keypoints_retargeted.npz \
  --output /tmp/astro-jogging.pkl \
  --fps 30
```

```bash
uv run python -m mjlab_playground.motion_lib.scripts.convert_pyroki_to_gmr --help
```

The converter refuses to replace existing output unless `--force-remake` is
provided.

## Python entry points

The package root exports the stable values, pipeline, and sampling interfaces:

```python
from mjlab_playground.motion_lib import (
    MimicMotionManager,
    MotionLib,
    MotionLibCfg,
    MotionManager,
    MotionManagerCfg,
    ReferenceFrame,
    ReferenceMotion,
)
```

Persistence, loading adapters, resampling internals, viewers, and recording
helpers are intentionally imported from their defining submodules. See the
[maintainer reference](../../../docs/motion_lib.md#class-by-class-reference) for
those boundaries.

## Common failures

- **Output already exists:** choose a new `--output-file`, or pass `--overwrite`
  to `run_motion_lib` only when atomic replacement is intentional.
- **`proto` is not implemented:** choose `--format pyroki` for source processing
  or use `--format mjlab` with the viewer for versioned/legacy rich artifacts.
- **Native viewer on macOS:** use `--viewer viser` or the default `auto` mode.
- **Reference motion assembly rejects a clip:** verify that transformed clips
  have matching FPS, DOF names, body names, dtype, device, and contact presence.
- **Unexpected files are ignored:** directory inputs are intentionally flat;
  nested directories are not scanned recursively.
