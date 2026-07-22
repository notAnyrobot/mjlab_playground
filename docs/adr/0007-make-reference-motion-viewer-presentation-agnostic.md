---
status: accepted
---

# Make the Reference Motion Viewer Presentation Agnostic

The reference motion viewer must behave consistently on macOS and Ubuntu without making playback semantics depend on MuJoCo's native passive viewer. We will make `MotionViewer` a deep, source- and presentation-agnostic module that owns playback, graceful stop, status, and background-recording policy behind a small `run()` interface; orchestration modules construct and inject an already-loaded reference motion set, a motion scene, a viewer adapter, and a background recorder. Native MuJoCo and Viser adapters will share a single `run(update)` seam and the same authoritative MuJoCo scene, while reusing `mjlab.viewer.ViewerConfig`, `mjlab.viewer.OffscreenRenderer`, and `mjlab.viewer.viser.MjlabViserScene` instead of coupling to the RL-oriented `BaseViewer` implementations or recreating their rendering logic.

## Considered Options

- Keep `launch_passive` inside `MotionViewer`: rejected because it leaks MuJoCo model, data, callbacks, and lifecycle requirements into the core interface and fails under ordinary Python on macOS.
- Subclass `mjlab.viewer.BaseViewer`, `NativeMujocoViewer`, or `ViserPlayViewer`: rejected because those modules own RL environment and policy orchestration that reference-motion playback does not have.
- Expose separate `open`, `poll`, `render`, `sync`, and `close` methods: rejected because every adapter would have to reproduce ordering and failure semantics, producing a shallow module.

## Consequences

- Launch scripts remain composition roots and depend only on stable factories, not concrete adapter classes.
- `--viewer auto|native|viser` selects presentation; `auto` chooses Viser on macOS, native MuJoCo on Linux with a display, and Viser on Linux without a display.
- Interactive recording becomes a one-shot background request that reuses deterministic headless recording and does not depend on interactive pause, speed, position, or refresh cadence.
- Existing CLI inputs remain stable, but the old MuJoCo-specific Python orchestration interface is intentionally replaced rather than wrapped.
