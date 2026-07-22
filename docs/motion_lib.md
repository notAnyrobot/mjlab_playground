# Motion Library Maintainer Reference

This page documents the architecture and maintenance contracts of
`mjlab_playground.motion_lib`. Start with the
[task-oriented module README](../src/mjlab_playground/motion_lib/README.md) when
you need runnable examples.

This is a curated API guide, not a generated API catalog. It describes stable
public interfaces and important internal seams without copying every signature
or docstring. Generated Sphinx/autodoc pages can be added later when this
repository has a documentation build and the public API is sufficiently stable.

## Scope

The motion library owns the path from robot-specific source motion to canonical
reference motion values and their runtime use:

```text
source adapter -> canonical value -> resampler -> simulator enrichment
                                             \-> viewer
rich clips -> reference motion assembly -> versioned serializer -> one artifact
assembled reference motion -> sampler -> MotionLib.query() -> consumer batch
```

The module deliberately separates these responsibilities:

- `MotionLoader` adapts storage formats to `ReferenceMotion`.
- `MotionLib` owns source loading, resampling, simulator enrichment, and query.
- `ReferenceMotion` owns frame values and compatible in-memory clip assembly.
- `ReferenceMotionNpzWriter` owns durable versioned serialization.
- `run_motion_lib` owns source-to-artifact orchestration.
- `MotionManager` and `MimicMotionManager` own sampling policy and track state.
- `MotionViewer` owns playback semantics behind injected scene, presentation,
  and recording interfaces.
- Modules under `scripts/` are composition roots, not reusable domain APIs.

These boundaries are recorded in the accepted ADRs:

- [Split Reference Motion Viewer From Launch Runner](adr/0001-split-reference-motion-viewer-from-launch-runner.md)
- [Make the Reference Motion Viewer Presentation Agnostic](adr/0002-make-reference-motion-viewer-presentation-agnostic.md)
- [Use Versioned NPZ Reference Motion Artifacts](adr/0003-use-versioned-npz-reference-motion-artifacts.md)
- [Use Reference Motion Clip Spans for Packed Logical Clips](adr/0004-use-reference-motion-clip-spans.md)
- [Run Source to Reference Motion Artifact Production From the Source Runner](adr/0005-run-source-to-reference-motion-artifact.md)

## Domain model

The repository glossary in [`CONTEXT.md`](../CONTEXT.md) is authoritative. The
most important concepts for this module are:

- **Source motion clip:** the smallest trusted input, typically generalized
  coordinates without derived body state.
- **Reference frame:** one desired robot-coordinate timestep, not current
  simulator state.
- **Reference motion:** one or more packed frame trajectories plus optional
  clip and axis metadata.
- **Reference motion clip span:** a lightweight logical description of one clip
  inside a packed reference motion.
- **Rich reference clip:** a reference clip containing generalized-coordinate
  fields and simulator-derived body fields.
- **Reference motion assembly:** deterministic concatenation of compatible rich
  clips; it does not resample, enrich, repair, or serialize them.
- **Versioned reference motion artifact:** the device-neutral `.npz`
  serialization of one canonical `ReferenceMotion`.
- **Motion scene:** an injected robot scene that applies reference frames and
  exposes resulting robot state.

Use these terms consistently. In particular, do not use "package" to conflate
in-memory assembly with on-disk serialization.

## Supported formats and platforms

| Name | Role | Current support |
|---|---|---|
| `pyroki` | Source-format adapter | Implemented. Requires `base_frame_pos`, `base_frame_wxyz`, and `joint_angles`. |
| `mjlab` | Versioned or legacy rich `.npz` loader | Implemented in `MotionLoader`, runtime consumers, and the viewer. New durable output uses versioned v1. |
| `proto` | Reserved source-format name | Not implemented. Public CLIs may expose the reserved choice but fail clearly. |
| GMR `.pkl` | Compatibility output | Written by `convert_pyroki_to_gmr`; it is not accepted by `MotionLib`. |

`MotionLibCfg` narrows the transformation pipeline further: v1 accepts only
`source_format="pyroki"` and `robot="astro"`. The lower-level loader has the
wider format switch because viewers and runtime consumers must also consume
`mjlab` artifacts.

Viewer presentation support is:

- Viser on macOS and Linux;
- native MuJoCo on Linux with a display; and
- `auto` resolution to Viser on macOS or headless Linux and native MuJoCo on
  Linux with a display.

## Public package surface

`motion_lib/__init__.py` is intentionally small.

| Export | Purpose |
|---|---|
| `MotionLib`, `MotionLibCfg` | Transformation and query pipeline. |
| `ReferenceFrame`, `ReferenceMotion`, `ReferenceMotionClipSpan` | Canonical reference values and logical clip navigation. |
| `ReferenceMotionState` | Deprecated compatibility alias for `ReferenceMotion`. |
| `MotionManager`, `MotionManagerCfg` | Stateless expert-sample policy. |
| `MimicMotionManager` | Persistent per-environment reference tracks. |
| `ReferenceMotionSample` | Sampled clip IDs and clip-local times. |
| `ClipWeighting`, `TimeSampling` | Sampling policy type aliases. |

Loaders, resamplers, the NPZ writer, viewer classes, adapters, and recording
helpers are not re-exported. Import those from their defining submodules so
internal boundaries remain visible at call sites.

## Data flow and ownership

### Source processing

1. `MotionLib.load(path)` delegates to `MotionLoader` using configuration-owned
   source format, FPS, and device. It returns a list because one flat directory
   can contain several source clips.
2. `MotionLib.resample(motion)` accepts one `ReferenceMotion`. It interpolates
   generalized-coordinate fields and derives generalized-coordinate velocity
   fields at `output_fps`.
3. `MotionLib.enrich(motion)` accepts one already-resampled clip. It applies each
   frame to one cached `MujocoSceneAdapter` and reads body pose and velocity
   fields back from the Astro scene.
4. The source runner passes every rich clip to `ReferenceMotion.from_clips()` in
   loader order, including when only one clip was loaded.
5. `ReferenceMotionNpzWriter.write(...)` validates and atomically publishes the
   assembled canonical value exactly once.

The scalar `resample` and `enrich` interfaces make orchestration explicit. The
runner owns loops over clips, cross-module sequencing, failure context, and the
singular output path. It publishes no intermediate one-clip artifacts.

### Runtime sampling

1. `MotionManager` samples a `ReferenceMotionSample` containing `motion_ids` and
   clip-local `motion_times`.
2. `MotionLib.query(...)` validates those tensors, maps each time into the
   selected packed clip, and interpolates fields.
3. Linear values use linear interpolation. Canonical `wxyz` quaternions use
   spherical interpolation. Floating contact values interpolate; non-floating
   contact labels select the lower frame.

If clip metadata is absent, both sampling and query treat the full packed tensor
as one clip.

#### Sampling input

Runtime sampling operates on one packed `ReferenceMotion`. Multi-clip values
provide contiguous `clip_starts`, per-clip frame-count `clip_lengths`, and
per-clip `clip_fps` metadata:

```python
import torch

from mjlab_playground.motion_lib import ReferenceMotion

clip_lengths = torch.tensor([6, 11], dtype=torch.long)
clip_starts = torch.tensor([0, 6], dtype=torch.long)
frame_count = int(clip_lengths.sum())

reference_motion = ReferenceMotion(
    name="training-reference",
    fps=10.0,
    root_pos=torch.zeros(frame_count, 3),
    root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(frame_count, 1),
    dof_pos=torch.zeros(frame_count, 29),
    clip_starts=clip_starts,
    clip_lengths=clip_lengths,
    clip_fps=torch.full((clip_lengths.numel(),), 10.0),
)
```

`clip_starts` index into packed frame tensors, while `clip_fps` converts
clip-local seconds into frame positions. When these fields are omitted,
`MotionManager`, `MimicMotionManager`, and `MotionLib.query()` treat the full
frame tensor as one clip. `ReferenceMotionState` remains only as a deprecated
compatibility alias; new code should use `ReferenceMotion`.

#### AMP-style expert batches

Use `MotionManager.sample_batch()` for temporary expert batches. Sampling does
not mutate persistent environment tracks:

```python
from mjlab_playground.motion_lib import (
    MotionLib,
    MotionLibCfg,
    MotionManager,
    MotionManagerCfg,
)

motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
manager = MotionManager(
    reference_motion,
    MotionManagerCfg(
        clip_weighting="duration",
        time_sampling="uniform",
        history_seconds=0.2,
        future_seconds=0.1,
    ),
)

expert_sample = manager.sample_batch(256)
expert_state = motion_lib.query(
    reference_motion,
    motion_ids=expert_sample.motion_ids,
    motion_times=expert_sample.motion_times,
)
```

The sampler output is motion IDs and motion times only. Consumers pass those
tensors to `MotionLib.query()` when they need root, joint, body, or contact
tensors; MotionLib owns query and interpolation.

#### Persistent mimic tracks

Use `MimicMotionManager` when each environment needs persistent reference
playback state. `sample_envs()` assigns selected tracks, `advance_envs()` moves
active tracks by elapsed seconds, and `done_envs()` reports tracks that would
exceed their valid sampling windows:

```python
import torch

from mjlab_playground.motion_lib import MimicMotionManager, MotionManagerCfg

mimic_manager = MimicMotionManager(
    reference_motion,
    num_envs=4096,
    cfg=MotionManagerCfg(
        clip_weighting="uniform",
        time_sampling="adaptive",
        future_seconds=0.2,
    ),
)

reset_env_ids = torch.tensor([0, 7, 42], dtype=torch.long)
reset_sample = mimic_manager.sample_envs(reset_env_ids)
reset_state = motion_lib.query(
    reference_motion,
    motion_ids=reset_sample.motion_ids,
    motion_times=reset_sample.motion_times,
)

mimic_manager.advance_envs(dt=1.0 / 50.0)
done_mask = mimic_manager.done_envs(lookahead=0.2)
query_mask = (mimic_manager.motion_ids >= 0) & ~done_mask
active_state = motion_lib.query(
    reference_motion,
    motion_ids=mimic_manager.motion_ids[query_mask],
    motion_times=mimic_manager.motion_times[query_mask],
)
```

For adaptive sampling, call
`mimic_manager.report_env_outcomes(env_ids, failed=...)` after task outcomes are
known. This updates manager-owned failure pressure; it does not query motion
tensors or reset environments.

#### V1 boundaries

The managers own clip weighting, valid time windows, time selection, adaptive
failure pressure, and mimic track state. MotionLib owns query and interpolation
over the packed reference tensors. Out of scope for v1: rewind sampling,
contact-label resampling, viewer integration, and packaged artifact metadata
beyond runtime clip metadata.

### Viewing and recording

The launch runner loads motions, constructs a robot scene and presentation
adapter, then injects them into `MotionViewer`. `MotionViewer` controls playback
state and frame application without knowing the source format or concrete UI.
It flattens each loaded value's logical spans in outer-reference-motion order
and then stored clip order. Selection, timing, progress, looping, and status are
clip-local; frame application maps each local index back into the original
packed parent, so playback never crosses a clip seam.

Interactive background recording captures only the selected span in a child
process while playback continues. Public headless recording remains a batch
operation over every logical span. Both modes use canonical span identity and
mapping plus the same output plan, but retain separate lifecycle semantics.

## Class-by-class reference

### `motion_loader.py`

#### `ReferenceFrame`

A frozen value for one robot-coordinate timestep. `root_pos`, canonical `wxyz`
`root_rot`, and `dof_pos` are required. Velocity, body-state, body-contact, and
source-foot-contact tensors are optional so the same value can represent source,
resampled, or rich data.

#### `ReferenceMotion`

The canonical tensor-backed motion value. Required packed frame tensors are
`root_pos`, `root_rot`, and `dof_pos`; optional tensors add velocity, body state,
and contacts. Operational metadata describes clip spans and names, while
`dof_names` and `body_names` define tensor axes.

Important constructors and accessors:

- `from_frames(...)` stacks consistent `ReferenceFrame` values. Every optional
  field must be present on all frames or on none.
- `from_clips(...)` assembles compatible rich, one-clip values. It requires at
  least three frames per clip, common FPS and axis order, and consistent
  `body_contacts` presence. Source `foot_contacts` and package-of-packages input
  are rejected.
- `get_frame(index)` returns one packed frame and checks bounds.
- `iter_clip_spans()` lazily validates operational clip metadata and yields
  logical descriptors in stored order. Metadata-free values yield one implicit
  full-length span.
- `clip_name(clip_id)` lazily decodes one UTF-8 identity from compact CPU
  metadata.

`ReferenceMotionState` is a deprecated alias. New code should use
`ReferenceMotion`.

#### `ReferenceMotionClipSpan`

A frozen logical descriptor over one packed parent. It exposes the parent,
integer clip ID, packed start frame, frame count, FPS, and lazily decoded name.
`to_packed_frame(local_frame_index)` validates clip-local bounds before mapping
to the packed parent. It never materializes a clip-local `ReferenceMotion`,
tensor slice, or tensor view.

Operational span validation belongs to `ReferenceMotion.iter_clip_spans()` and
therefore remains lazy. Requesting spans rejects partial, empty, misaligned,
noncontiguous, out-of-bounds, nonpositive, or incoherently named metadata;
ordinary `ReferenceMotion` construction remains passive.

#### `MotionLoader`

The abstract format-adapter factory. `load(...)` constructs a concrete adapter
through `from_format(...)`, then returns a list of canonical motions. File inputs
must have the expected suffix; directory inputs scan sorted direct children and
do not recurse.

#### `PyrokiMotionLoader`

Loads PyRoki `.npz` fields, validates aligned frame counts, normalizes canonical
`wxyz` root quaternions, and moves tensors to the selected device. It can load
optional source contact labels at the adapter level, but `MotionLib.load()` does
not expose that parameter because the v1 enrichment pipeline rejects source foot
contacts.

#### `ReferenceMotionNpzLoader`

Loads `mjlab` `.npz` artifacts. It accepts versioned v1 artifacts and the legacy
unversioned rich single-clip layout. Versioned loading validates schema keys,
frame and axis metadata, clip spans, common FPS, minimum clip lengths, compact
UTF-8 clip names, and tensor shapes before constructing `ReferenceMotion`.

### `motion_lib.py`

#### `MotionLibCfg`

Frozen public pipeline configuration. It validates positive integer-valued
`source_fps` and `output_fps`, normalizes `device` to `torch.device`, and enforces
the v1 PyRoki/Astro boundary.

#### `MotionLib`

The public transformation and query facade:

- `load(...)` returns source generalized-coordinate clips.
- `resample(...)` transforms exactly one source clip and names errors with the
  motion identity.
- `enrich(...)` transforms exactly one already-resampled clip and reuses a
  lazily created MuJoCo scene adapter across calls.
- `query(...)` interpolates package-shaped reference tensors by clip ID and
  clip-local seconds.

`resample` rejects already-rich clips and contact-bearing clips. `enrich`
rejects the wrong FPS, existing rich body fields, and source foot contacts. The
class does not assemble or serialize artifacts.

### `motion_resampler.py`

#### `MotionResamplingCfg`

Internal frozen configuration containing the positive integer-valued output
FPS.

#### `ReferenceMotionResampler`

Resamples one source clip using a deterministic target timeline. Positions and
joint values interpolate linearly; root quaternions use normalized spherical
interpolation. It derives root linear velocity, root angular velocity, and DOF
velocity from generalized-coordinate tensors. It does not load, enrich, write,
or preserve source contact labels. V1 supports equal-rate output and upsampling;
it rejects downsampling.

### `motion_manager.py`

#### `MotionManagerCfg`

Frozen sampling policy. `clip_weighting` is `uniform`, `duration`, or
`explicit`; `time_sampling` is `start`, `uniform`, or `adaptive`. History and
future windows restrict valid clip-local times. Adaptive settings control the
number of bins and the uniform exploration floor.

#### `ReferenceMotionSample`

A small value containing sampled `motion_ids` and `motion_times`. It deliberately
does not contain queried reference tensors.

#### `MotionManager`

Stateless AMP-style sampling over clip metadata. It computes valid time windows,
normalizes clip probabilities, and samples temporary expert batches. Adaptive
sampling tracks failure pressure per clip and time bin. It never mutates mimic
environment tracks and never queries motion tensors.

#### `MimicMotionManager`

Extends `MotionManager` with persistent `motion_ids` and `motion_times` for a
fixed number of environments. `sample_envs`, `advance_envs`, `done_envs`, and
outcome reporting manage reference playback state; consumers still call
`MotionLib.query()` separately.

See [Runtime sampling](#runtime-sampling) for construction and query examples.

### `reference_motion_npz_writer.py`

#### `ReferenceMotionNpzWriter`

Validates a train-ready rich `ReferenceMotion`, converts tensors to CPU NumPy
arrays, and publishes a schema-v1 `.npz` atomically. Required rich fields are
root, DOF, body pose, and body velocity tensors. `body_contacts` is optional;
source `foot_contacts` is forbidden.

The writer refuses replacement by default. For no-clobber publication it writes
and fsyncs a temporary file, then links it into place. Explicit overwrite uses
`os.replace`. Temporary files are removed after success or failure.

The writer is a pure serializer: it accepts any valid canonical one-clip or
multi-clip `ReferenceMotion` without inspecting clip count or source origin. It
does not load inputs, perform reference motion assembly, or expose a command-line
entry point.

### `_reference_motion_npz_schema.py`

#### `ReferenceMotionNpzSchema`

A private frozen descriptor that centralizes schema-version keys. Version 1
requires generalized-coordinate fields, derived velocity and body fields, FPS,
contiguous clip spans, compact clip names, and DOF/body axis names. Shared
validators keep loader, writer, and assembly errors aligned.

### `mujoco_scene_adapter.py`

#### `MujocoSceneAdapterCfg`

Internal scene construction data: robot configuration, output FPS, and device.

#### `MujocoSceneAdapter`

Owns one mjlab MuJoCo scene and robot entity. It applies a `ReferenceFrame` in
the correct simulation update order, exposes the resulting robot state, keeps
display state synchronized, and configures root tracking. Both `MotionLib`
enrichment and reference viewing reuse this scene boundary.

### `motion_viewer.py`

#### Session values and protocols

| Class | Responsibility |
|---|---|
| `PlaybackAction` | Typed playback commands for pause, clip selection, speed, selected-clip recording, and stop. |
| `RecordingStatus` | Disabled, idle, running, succeeded, or failed recording state. |
| `ViewerTick` | Adapter-to-session input for elapsed time and queued actions. |
| `ViewerSnapshot` | Session-to-adapter presentation state. |
| `MotionScene` | Protocol for applying one reference frame to the authoritative scene. |
| `ViewerAdapter` | Protocol whose `run(update)` owns a concrete interactive event loop. |
| `BackgroundRecorder` | Protocol for one asynchronous selected-clip recording. |
| `DeterministicRecordingOperation` | Protocol for synchronous headless recording. |
| `ViewerError` | Session-level viewer failure. |

#### `PlaybackController`

Owns selected clip, frame, pause state, speed, and frame advancement. It is UI
independent and keeps playback policy testable without opening a viewer.

#### `TerminalStatusReporter`

Writes status updates without making the core viewer depend on one UI.

#### `MotionViewer`

The source- and presentation-agnostic session. It applies frames to the injected
scene, handles actions, updates adapter snapshots, coordinates graceful stop,
and separates interactive playback from selected-clip background recording. Its
`record(...)` path drives deterministic headless recording without launching an
interactive adapter.

`MotionViewer.run()` remains the primary presentation-agnostic seam. Loaded
canonical motions and the injected scene remain its construction inputs;
presentation adapters and recording adapters do not own package indexing.

### `recording.py`

#### Planning and request values

| Class | Responsibility |
|---|---|
| `RecordingResult` | Output path, frame count, FPS, written flag, and optional skip reason. |
| `RecordingTarget` | One loaded-reference-motion index, internal clip ID, source artifact, and destination path. |
| `RecordingOutputPlan` | Deterministic output directory and target set. |
| `RecordingRequestTarget` | Serializable child-process target identity and path. |
| `RecordingRequest` | One deterministic multi-target recording request. |

#### Recording implementations

| Class | Responsibility |
|---|---|
| `BackgroundProcess` | Minimal child-process lifecycle protocol. |
| `SubprocessBackgroundRecorder` | Starts, polls, waits for, or cancels one child recording process. |
| `RecordingScene` | Protocol for applying a frame before rendering. |
| `FrameRenderer` | Protocol for rendering one frame. |
| `DeterministicRecordingError` | Labeled synchronous recording failure. |
| `DeterministicRecorder` | Traverses complete clips deterministically and writes video outputs. |
| `_MjlabOffscreenFrameRenderer` | Private adapter over mjlab's offscreen renderer. |
| `RecordingAttachment` | Buffers rendered frames for one video and writes or aborts the output. |

`create_mjlab_deterministic_recorder(...)` wires the scene to mjlab offscreen
rendering. `create_subprocess_background_recorder(...)` builds the interactive
background path. `plan_recording_outputs(...)` centralizes collision-free output
names and directory selection. It expands every loaded logical span in
outer-reference-motion then clip-ID order, derives filenames from span names,
and rejects duplicate destinations before renderer construction.

The interactive child selector is deliberately internal: its request retains
the source artifact identity and internal clip ID needed to reproduce the
parent selection, but the public launch CLI exposes no single-clip headless
option. The public `--headless` contract always records the complete loaded
dataset or package.

### `viewers/`

#### `NativeCameraConfig`

Small immutable camera defaults for the native MuJoCo presentation.

#### `NativeMujocoViewerAdapter`

Runs MuJoCo passive-viewer playback on supported Linux displays. It translates
keyboard input into `PlaybackAction`, applies session snapshots, and owns native
viewer lifecycle without owning reference motion semantics.

#### `ViserPlaybackControls`

Builds browser controls and translates callbacks into typed playback actions.

#### `ViserViewerAdapter`

Runs the browser-based Viser presentation around the same authoritative MuJoCo
scene. It owns server, scene, GUI, and action-queue lifecycle while delegating
playback policy to `MotionViewer`.

`resolve_viewer_mode(...)` is a pure platform/display decision. The
`create_viewer_adapter(...)` factory keeps concrete classes out of the launch
runner.

### `scripts/`

Scripts are composition roots and may depend on concrete adapters and robot
configuration. Reusable behavior should remain in the modules above.

| Module | Responsibility |
|---|---|
| `run_motion_lib` | Loads source clips, transforms each through the scalar `MotionLib` stages, assembles them through `ReferenceMotion.from_clips()`, and publishes one versioned reference motion artifact through the writer. |
| `launch_motion_viewer` | Loads motions, constructs the Astro scene and selected presentation, and wires recording or smoke-test paths. |
| `convert_pyroki_to_gmr` | Converts PyRoki arrays to the GMR pickle compatibility layout and changes root quaternion order to GMR `xyzw`. |

## Versioned artifact contract

Schema v1 uses named, pickle-free NumPy arrays. The private descriptor is the
source of truth for exact keys. Maintainers must preserve these invariants:

- `schema_version` is `1`.
- Root and body quaternions are finite, normalized, and stored in canonical
  MuJoCo-facing `wxyz` order.
- FPS is positive and integer-valued; every v1 clip uses the artifact's common
  FPS.
- Clip spans start at frame zero, are contiguous, and cover every packed frame.
- Every rich clip contains at least three frames.
- Clip names are non-empty UTF-8 byte spans represented by CPU
  `clip_name_bytes` and `clip_name_offsets` tensors.
- `dof_names` and `body_names` are ordered, non-empty UTF-8 strings aligned to
  their tensor axes.
- Every required tensor has a common frame count. Body and DOF axes agree with
  their metadata.
- Durable arrays are written from CPU NumPy data. The load/assembly `device`
  controls working tensor placement, not artifact portability.
- A one-clip artifact and an assembled multi-clip artifact use the same schema.

Legacy unversioned rich `.npz` remains readable for compatibility. Do not extend
legacy layout semantics; new durable output must be versioned.

### Artifact lifecycle

For a multi-clip artifact build, trusted source clips and the final versioned
artifact are durable. The source runner transforms every clip before reference
motion assembly and publishes no intermediate artifacts. Source discovery is
deterministic, sorted, and limited to direct children; assembly preserves that
loader order. Publication is atomic and no-clobber by default, with explicit
overwrite permission for intentional replacement. Schema v1 remains the one
authoritative format, with no manifest, sidecar, provenance extension,
dependency, or derived PyTorch cache.

The validated Astro SFU artifact at
`assets/motions/astro/sfu/mjlab-astro.npz` contains 35 clips and 39,225 frames at
50 FPS, with 29 DOFs and 31 bodies.

## Extension guidance

### Add a source format

1. Implement a `MotionLoader` adapter that produces canonical
   `ReferenceMotion` values.
2. Extend `MotionFormat` and `MotionLoader.from_format`.
3. Decide whether the format belongs in `MotionLibCfg`, the viewer, or both.
4. Keep source-specific quaternion and field conversion inside the adapter.
5. Add loader tests, CLI option tests, and one end-to-end boundary test.

Do not leak source keys into `MotionViewer`, `MotionManager`, or consumers.

### Add a robot

1. Add a robot configuration factory and extend the robot literal/CLI choices.
2. Construct it behind `MujocoSceneAdapterCfg` rather than branching inside
   playback or recording.
3. Verify DOF and body axis names/order, canonical quaternion order, and frame
   application semantics.
4. Add simulator-safe tests before claiming enrichment support.

Avoid silent changes to units, coordinate frames, joint order, gains, or action
scaling.

### Add a viewer presentation

1. Implement the `ViewerAdapter.run(update)` seam.
2. Translate UI input into `PlaybackAction` values.
3. Present `ViewerSnapshot` state without owning playback policy.
4. Extend `ViewerMode`, resolution, and the adapter factory.
5. Reuse the injected authoritative `MotionScene`.

### Add sampling behavior

Keep clip/time selection in motion managers and interpolation in
`MotionLib.query`. A new policy should return `ReferenceMotionSample`; it should
not materialize reference tensors or mutate unrelated environment state.

### Evolve the artifact schema

Add a new schema descriptor and explicit loader/writer handling rather than
silently changing v1 keys. State compatibility and migration behavior in an ADR
when the decision is hard to reverse, surprising, and represents a real
trade-off.

## Failure behavior

Errors should identify the boundary and offending motion or path:

- loaders prefix file-specific failures with the source path;
- scalar transformation stages include the motion identity;
- runner assembly errors report deterministic loader order;
- viewer smoke tests label load, scene construction, and frame application
  phases; and
- recording errors remove incomplete outputs where possible.

Do not catch failures only to continue with partial datasets. No-clobber and
atomic publication are data-safety contracts, not convenience behavior.

## Test map

| Responsibility | Primary tests |
|---|---|
| Canonical values and frame/clip assembly | `tests/test_reference_motion.py` |
| PyRoki and `mjlab` loading | `tests/test_motion_loaders.py` |
| Pipeline stages and package query | `tests/test_motion_lib.py` |
| Resampling math and contracts | `tests/test_reference_motion_resampler.py`, `tests/test_motion_math_util.py` |
| Versioned schema and clip-count-agnostic serialization | `tests/test_reference_motion_npz_writer.py` |
| Source-to-artifact orchestration, assembly, and publication | `tests/test_run_motion_lib.py` |
| Sampling and mimic tracks | `tests/test_reference_motion_sampler.py` |
| Playback/session semantics | `tests/test_motion_viewer_session.py` |
| Viewer launch, selection, and smoke test | `tests/test_launch_motion_viewer.py` |
| Recording | `tests/test_motion_recording.py` |
| MuJoCo scene application | `tests/test_mujoco_scene_adapter.py` |
| PyRoki-to-GMR compatibility conversion | `tests/test_convert_pyroki_to_gmr.py` |

When changing a seam, run its focused tests first, then the neighboring pipeline
tests. Visual behavior should be validated through `--smoke-test`, interactive
viewer inspection, or deterministic recording as appropriate; unit tests alone
cannot establish presentation quality.

## Documentation maintenance

When changing a public command or contract:

1. Update CLI help and code-level validation together.
2. Update the task-oriented README if a user command or expected output changes.
3. Update this reference if ownership, class responsibility, invariants, or an
   extension seam changes.
4. Update `CONTEXT.md` only when domain language changes, not for implementation
   details.
5. Add or revise an ADR only for a hard-to-reverse, surprising trade-off.
6. Verify every documented CLI with `--help` and check local links before
   merging.
