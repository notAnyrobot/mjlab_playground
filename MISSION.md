# Mission: Reference Motion Library Design

## Why
Build a `mjlab_playground.motion_lib` module that can load, manage, and sample reference motion data without blindly copying ProtoMotions. The goal is to understand ProtoMotions' separation between stored motion data and environment playback state, then adapt the useful parts to `mjlab_playground`.

## Success looks like
- Explain which responsibilities belong in a motion library versus a motion manager.
- Design a minimal `mjlab_playground` motion module with clear tensor storage, metadata, and query APIs.
- Avoid mixing immutable packaged motion data with mutable curriculum or per-environment sampling state.
- Write focused tests for frame indexing, time sampling, interpolation, and empty-library behavior.

## Constraints
- Use the local ProtoMotions source as the primary reference.
- Keep lessons short and tied to implementation decisions in this repo.
- Preserve unrelated dirty worktree changes.

## Out of scope
- Porting the full ProtoMotions environment stack.
- Training-policy or reward design beyond what is needed to place motion queries.
