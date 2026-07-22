---
status: accepted
---

# Use Versioned NPZ Reference Motion Artifacts

> Superseding orchestration note: ADR 0005 replaces only this ADR's thin
> runnable-writer entry point with the source runner's one-command workflow. The
> schema, assembly, serializer, and atomic-publication decisions remain accepted.

New rich reference motion artifacts will use a self-describing, device-neutral `.npz` schema for both individual clips and assembled multi-clip datasets. `ReferenceMotion` owns compatible-clip assembly, `ReferenceMotionNpzWriter` serializes any valid canonical value, and the writer module exposes only a thin runnable orchestration entry point; `MotionLib` remains responsible for motion loading, resampling, enrichment, and querying rather than persistence. NPZ is authoritative because its named arrays provide an explicit, portable, pickle-free schema; `.pt` may be introduced only as a derived runtime cache if benchmarks justify it.

## Considered Options

- Make `MotionLib` own packaging and writing: rejected because it mixes motion transformation with persistence and produces a shallow forwarding interface.
- Use `.pt` as the authoritative artifact: rejected because device remapping does not remove CPU staging, while the format couples the durable dataset to PyTorch serialization.
- Store clip metadata in a YAML sidecar: rejected because it creates a second authoritative artifact and does not reduce VRAM use.

## Consequences

- Schema version 1 stores explicit clip spans and names, body names, DOF names, common FPS, canonical `wxyz` quaternions, and the rich reference tensors required to reconstruct `ReferenceMotion`.
- Clip names remain compact, lazily decoded CPU metadata; the loader or assembly caller selects tensor placement with a device argument that defaults to CPU.
- Legacy unversioned rich `.npz` files remain readable but unchanged and are not accepted as inputs to the versioned assembly workflow.
- The writer writes atomically and refuses replacement by default; explicit overwrite permission is required.
