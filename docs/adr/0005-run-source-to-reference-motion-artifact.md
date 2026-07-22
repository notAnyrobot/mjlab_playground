---
status: accepted
---

# Run Source to Reference Motion Artifact Production From the Source Runner

The source runner is the single production orchestration path from source motion
clips to one versioned reference motion artifact. It loads clips through one
configured `MotionLib`, resamples and enriches each clip independently, performs
reference motion assembly through `ReferenceMotion.from_clips()`, and passes the
assembled canonical value to `ReferenceMotionNpzWriter.write()` exactly once.
The runner requires an explicit output file, protects it by default, and accepts
explicit overwrite permission for atomic replacement.

This decision refines ADR 0003 without changing its artifact architecture.
`MotionLib` still owns transformation, `ReferenceMotion.from_clips()` still owns
compatible in-memory assembly, and `ReferenceMotionNpzWriter` remains a pure,
clip-count-agnostic serializer. ADR 0003's schema-v1, device-neutral storage,
atomic publication, legacy-read compatibility, assembly validation, and common
one-clip/multi-clip artifact decisions remain accepted.

## Considered Options

- Retain the two-command workflow with staged one-clip artifacts: rejected
  because disposable files become a public workflow, orchestration is split
  across commands, and a partial build can be mistaken for a completed dataset.
- Keep the writer module runnable and move source transformation into it:
  rejected because loading source formats and configuring simulator enrichment
  are runner responsibilities, not serialization responsibilities.
- Add assembly or persistence methods to `MotionLib`: rejected because they
  would duplicate `ReferenceMotion.from_clips()` and couple transformation to a
  durable storage format.
- Make the source runner the composition root: accepted because it already owns
  source-pipeline configuration and can sequence transformation, assembly, and
  publication without weakening the domain boundaries.

## Consequences

- One runner invocation produces exactly one versioned reference motion artifact
  from either one source file or a flat source directory.
- Every clip is transformed before assembly begins; failures publish neither a
  final artifact nor intermediate one-clip artifacts.
- Loader order determines clip order in the assembled reference motion.
- The runner CLI uses required `--output-file` and optional `--overwrite`;
  removed `--output-dir` and writer-CLI behavior have no compatibility aliases.
- The writer module exposes only serialization behavior and is not runnable.
- Existing artifact schema and runtime consumers remain unchanged.
