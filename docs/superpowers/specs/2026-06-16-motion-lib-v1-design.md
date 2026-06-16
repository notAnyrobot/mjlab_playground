# mjlab_playground MotionLib V1 Design

## Status

Approved design for the first implementation slice of `mjlab_playground.motion_lib`.

## Goal

Build a standalone reference motion library for `mjlab_playground` that aligns with the useful `MotionLib` boundary from ProtoMotions while exposing a small, repo-local API for `mjlab` consumers.

The first version targets PyRoki-to-ProtoMotions retargeted datasets packaged as ProtoMotions `.pt` motion libraries.

## Non-Goals

- No reuse of `mjlab` tracking-task `MotionLoader`.
- No `.npz`, raw `.motion`, YAML, or directory loading in v1.
- No save, repack, filtering, or conversion pipeline in v1.
- No per-environment `MotionManager` in v1.
- No construction-time body slicing in v1.
- No local-rotation-derived `dof_pos` interpolation in v1.

## Architecture

`mjlab_playground.motion_lib` contains a standalone immutable tensor store and query engine:

- `MotionLibConfig`: small loader config for a ProtoMotions `.pt` file and target device.
- `MotionState`: repo-local dataclass returned by query APIs.
- `MotionLib`: strict loader, metadata store, batched query engine, and minimal stateless sampling helper.
- Schema validation helpers: fail before training if a package is malformed or incomplete.
- Interpolation helpers: linear interpolation for vector fields and slerp for rotations.

The module follows the ProtoMotions boundary: the library owns reference tensors, metadata, interpolation, and provenance. Mutable curriculum weights, fixed motion IDs, exclusions, per-env `motion_ids`, per-env `motion_times`, and playback advancement belong in a future manager.

## Public API

```python
@dataclass(frozen=True, kw_only=True)
class MotionLibConfig:
    motion_file: str | Path
    device: str | torch.device = "cpu"
```

```python
@dataclass(frozen=True)
class MotionState:
    rigid_body_pos: torch.Tensor
    rigid_body_rot: torch.Tensor  # wxyz, matching mjlab math APIs
    rigid_body_vel: torch.Tensor
    rigid_body_ang_vel: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    rigid_body_contacts: torch.Tensor | None = None
```

`MotionLib` exposes:

```python
MotionLib(config: MotionLibConfig)
MotionLib.empty(device: str | torch.device = "cpu")

num_motions() -> int
get_motion_length(motion_ids: torch.Tensor) -> torch.Tensor
get_motion_num_frames(motion_ids: torch.Tensor) -> torch.Tensor
get_motion_state(motion_ids: torch.Tensor, motion_times: torch.Tensor) -> MotionState
get_motion_state_exact_frame(
    motion_ids: torch.Tensor,
    frame_indices: torch.Tensor,
) -> MotionState

sample_motion_ids(
    num_samples: int,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor
sample_motion_times(
    motion_ids: torch.Tensor,
    *,
    truncate_time: float | None = None,
) -> torch.Tensor
sample_motion_ids_and_times(
    num_samples: int,
    *,
    truncate_time: float | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Internal tensor names may mirror ProtoMotions package fields, but consumers use `MotionState` field names.

## Data Model

Required ProtoMotions `.pt` fields:

- `gts`: rigid body positions, shape `[total_frames, num_bodies, 3]`
- `grs`: rigid body rotations in ProtoMotions package convention, shape `[total_frames, num_bodies, 4]`
- `gvs`: rigid body linear velocities, shape `[total_frames, num_bodies, 3]`
- `gavs`: rigid body angular velocities, shape `[total_frames, num_bodies, 3]`
- `dps`: DOF positions, shape `[total_frames, num_dofs]`
- `dvs`: DOF velocities, shape `[total_frames, num_dofs]`
- `length_starts`: flat-frame start offset for each motion, shape `[num_motions]`
- `motion_lengths`: duration in seconds for each motion, shape `[num_motions]`
- `motion_dt`: timestep for each motion, shape `[num_motions]`
- `motion_num_frames`: frame count for each motion, shape `[num_motions]`
- `motion_weights`: initial sampling weights, shape `[num_motions]`
- `motion_files`: source provenance, length `num_motions`

Optional fields:

- `contacts`: returned as `MotionState.rigid_body_contacts` when present.
- `lrs`: preserved on the library as optional metadata, but not used for `dof_pos` interpolation in v1.

The library loads full tensors onto the configured device. It does not slice body tensors during construction.

## Loading And Validation

The loader accepts only `.pt` files in v1. It uses `torch.load(..., map_location=device, weights_only=False)` and validates before storing the package.

Validation rules:

- Required fields must exist.
- Tensor fields must have compatible first dimension `total_frames`.
- Body tensor fields must agree on body count.
- DOF tensor fields must agree on DOF count.
- Metadata fields must have length `num_motions`.
- `length_starts` must equal the cumulative starts derived from `motion_num_frames`.
- `motion_lengths`, `motion_dt`, and `motion_num_frames` must be positive.
- `motion_weights` must be finite, non-negative, and have positive total mass for sampling.
- `motion_files` must have one entry per motion.

After validation, rigid body rotations are converted once from ProtoMotions package order into `wxyz`, matching `mjlab` math APIs. `MotionState.rigid_body_rot` and stored `grs` exposed through query results are therefore `wxyz`.

## Query Flow

`get_motion_state(motion_ids, motion_times)` performs a batched continuous-time query:

1. Validate `motion_ids` and `motion_times` shapes.
2. Clamp `motion_times` to each selected clip's valid range.
3. Compute neighboring frame indices and interpolation blend.
4. Convert clip-local frame indices to flat indices with `length_starts[motion_ids]`.
5. Gather frame 0 and frame 1 tensors.
6. Linearly interpolate positions and velocities.
7. Slerp `rigid_body_rot`.
8. Interpolate `dof_pos` and `dof_vel` linearly.
9. Return a `MotionState`.

`get_motion_state_exact_frame(motion_ids, frame_indices)` gathers exact frames with no blending. Frame indices are validated against the selected motion lengths.

Out-of-range continuous times clamp to the first or last valid frame. The query path does not wrap times.

## Sampling Helpers

V1 includes minimal stateless sampling helpers on `MotionLib`:

- `sample_motion_ids(...)` samples IDs from stored `motion_weights` or caller-provided weights.
- `sample_motion_times(...)` samples uniformly within selected clip lengths.
- `sample_motion_ids_and_times(...)` is a convenience wrapper.

These helpers return fresh tensors only. They do not mutate weights, store active IDs, store active times, implement fixed IDs, handle exclusions, or checkpoint sampling state. That behavior belongs to a future manager.

`truncate_time` subtracts from selected clip lengths before time sampling. It fails if any selected clip would have no valid sampling interval.

## Error Handling

The module should raise `ValueError` with the field or operation name for:

- Non-`.pt` input paths.
- Missing required fields.
- Shape mismatches.
- Invalid `length_starts`.
- Invalid motion metadata.
- Empty or all-zero sampling weights.
- `truncate_time` longer than selected clips.
- Querying or sampling an empty library.

`FileNotFoundError` is used for missing `motion_file` paths.

## Test Plan

Focused tests should cover:

- Loading a synthetic strict ProtoMotions `.pt` package.
- Missing required fields fail with useful errors.
- `length_starts` validation.
- Full tensor loading without construction-time body slicing.
- `grs` conversion to `wxyz`.
- Exact frame indexing from `(motion_id, frame_idx)` to flat row.
- Continuous interpolation for positions and velocities.
- Rotation slerp.
- Query-time clamping for negative and too-large times.
- Sampling ID shape/range and invalid weight rejection.
- Sampling time bounds and `truncate_time`.
- Optional `contacts` return behavior.
- Optional `lrs` preservation without affecting `dof_pos`.
- Empty-library behavior.

## Deferred V2 Work

The following are intentionally deferred:

- Raw `.motion`, YAML, directory, and `.npz` loaders.
- Save/repack support.
- Filtering or body subsetting at load time.
- Windowed query APIs for AMP/history features.
- A stateful `MotionManager` for per-env playback, mutable weights, fixed IDs, exclusions, subsets, curriculum updates, and checkpoint state.
