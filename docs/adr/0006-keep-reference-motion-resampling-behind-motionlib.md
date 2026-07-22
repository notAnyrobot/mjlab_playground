---
status: accepted
---

# Keep Reference Motion Resampling Behind MotionLib

`MotionLib.resample()` owns reference motion resampling directly. The standalone
`MotionResamplingCfg`, `ReferenceMotionResampler`, and `motion_resampler.py`
module are removed because `MotionLibCfg` already owns the target FPS and
`MotionLib` was the only production caller. `ReferenceMotion` remains a passive
canonical value rather than gaining transformation behavior.

Resampling validation is an operation-specific input boundary, not general
`ReferenceMotion` validation. Version 1 accepts one metadata-free generalized-
coordinate source clip, supports equal-rate output and upsampling, and rejects
downsampling, rich body or contact fields, and clip or axis metadata that would
otherwise be discarded or interpolated across clip seams.

## Considered Options

- Retain a configurable standalone resampler: rejected because its one-field
  configuration duplicates `MotionLibCfg.output_fps` and adds an internal
  delegation layer without an independent adapter or policy seam.
- Move resampling onto `ReferenceMotion`: rejected because the canonical value
  intentionally remains passive while loaders, transformations, assembly,
  serialization, and runtime consumers enforce their own narrower contracts.
- Let `MotionLib.resample()` accept packed or already-rich motions: rejected
  because resampling packed tensors can cross clip seams and silently lose
  metadata, while resampling rich or contact-bearing fields has no accepted v1
  semantics.

## Consequences

- Callers configure source and output FPS once through `MotionLibCfg` and invoke
  the public `MotionLib.resample()` seam for each source clip.
- Position and DOF values interpolate linearly; root quaternions use normalized
  shortest-path `wxyz` interpolation; generalized-coordinate velocities are
  recomputed at the output FPS.
- Resampling failures retain motion identity and identify the unsupported field
  or FPS relationship.
- Resampling behavior is covered through `tests/test_motion_lib.py`; no private
  resampler configuration, class, module, or standalone test surface remains.
