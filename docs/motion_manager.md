# Motion Manager Usage

The motion manager is the runtime policy layer over a package-shaped
`ReferenceMotion`. It chooses clip-local `motion_ids` and `motion_times`;
`MotionLib` owns query and interpolation.

## Reference Motion

`ReferenceMotion` is the common batched pose/state container. Ordinary
loaded clips and sampled query results do not need clip metadata. Multi-clip
packages provide packed frame tensors plus contiguous clip metadata:

```python
import torch

from mjlab_playground.motion_lib import ReferenceMotion

clip_lengths = torch.tensor([6, 11], dtype=torch.long)
clip_starts = torch.tensor([0, 6], dtype=torch.long)
frame_count = int(clip_lengths.sum())

motion = ReferenceMotion(
    name="training-package",
    fps=10.0,
    root_pos=torch.zeros(frame_count, 3),
    root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(frame_count, 1),
    dof_pos=torch.zeros(frame_count, 29),
    clip_starts=clip_starts,
    clip_lengths=clip_lengths,
    clip_fps=torch.full((clip_lengths.numel(),), 10.0),
)
```

`clip_starts` index into the packed frame tensors, `clip_lengths` are frame
counts, and `clip_fps` carries the per-clip FPS used to turn seconds into frame
positions.

When package metadata is omitted, `MotionLib.query` and the samplers treat the
reference motion as one clip spanning the full batch.

`ReferenceMotionState` remains available only as a deprecated compatibility
alias for downstream callers. New code should import and use `ReferenceMotion`.

## AMP-Style Sampling

Use `MotionManager.sample_batch` for temporary expert batches. This
does not mutate persistent per-environment mimic tracks.

```python
from mjlab_playground.motion_lib import (
    MotionLib,
    MotionLibCfg,
    MotionManager,
    MotionManagerCfg,
)

motion_lib = MotionLib(MotionLibCfg(output_fps=10.0))
sampler = MotionManager(
    motion,
    MotionManagerCfg(
        clip_weighting="duration",
        time_sampling="uniform",
        history_seconds=0.2,
        future_seconds=0.1,
    ),
)

expert_sample = sampler.sample_batch(256)
expert_state = motion_lib.query(
    motion,
    motion_ids=expert_sample.motion_ids,
    motion_times=expert_sample.motion_times,
)
```

The sampler output is motion IDs and motion times only. Pass those tensors to
`MotionLib.query` when a task needs reference root, joint, body, or contact
tensors.

## Mimic Sampling

Use `MimicMotionManager` for persistent per-environment playback state.
`sample_envs` resamples selected env tracks, `advance_envs` advances active
tracks by elapsed seconds, and `done_envs` reports tracks that would exceed the
valid sampling window.

```python
import torch

from mjlab_playground.motion_lib import (
    MimicMotionManager,
    MotionManagerCfg,
)

mimic_sampler = MimicMotionManager(
    motion,
    num_envs=4096,
    cfg=MotionManagerCfg(
        clip_weighting="uniform",
        time_sampling="adaptive",
        future_seconds=0.2,
    ),
)

reset_env_ids = torch.tensor([0, 7, 42], dtype=torch.long)
reset_sample = mimic_sampler.sample_envs(reset_env_ids)
reset_state = motion_lib.query(
    motion,
    motion_ids=reset_sample.motion_ids,
    motion_times=reset_sample.motion_times,
)

mimic_sampler.advance_envs(dt=1.0 / 50.0)
done_mask = mimic_sampler.done_envs(lookahead=0.2)
query_mask = (mimic_sampler.motion_ids >= 0) & ~done_mask
active_state = motion_lib.query(
    motion,
    motion_ids=mimic_sampler.motion_ids[query_mask],
    motion_times=mimic_sampler.motion_times[query_mask],
)
```

For adaptive mimic sampling, call `report_env_outcomes(env_ids, failed=...)`
after task outcomes are known. Reporting updates sampler-owned failure pressure;
it does not query motion tensors or reset environments by itself.

## V1 Boundaries

The sampler owns clip weighting, valid time windows, random time selection, and
mimic track state. MotionLib owns query and interpolation over the packed
reference motion tensors.

Out of scope for v1: rewind sampling, contact-label resampling, viewer integration,
and packaged artifact metadata beyond the clip metadata required for runtime
sampling.
