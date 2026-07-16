# Motion Library Documentation Design

## Goal

Document `mjlab_playground.motion_lib` for two distinct audiences without changing runtime behavior:

- Give users a short, task-oriented module README with commands they can run immediately.
- Give maintainers a curated reference for the module's architecture, classes, contracts, extension seams, and tests.

The documentation should follow official mjlab's useful pattern of leading with concrete tasks and commands, then explaining concepts and implementation boundaries in greater depth.

## Deliverables

### Module README

Create `src/mjlab_playground/motion_lib/README.md` as the user entry point. It will:

- Explain what the motion library is and when to use it.
- Introduce the `load -> resample -> enrich -> serialize or view` workflow.
- Provide runnable quick-start commands.
- Cover the three modules under `motion_lib/scripts/`:
  - `run_motion_lib`
  - `launch_motion_viewer`
  - `convert_pyroki_to_gmr`
- Cover the runnable `reference_motion_npz_writer` module for multi-clip assembly.
- State operational constraints that commonly surprise users, including no-clobber exports and the fact that `--viewer` belongs to `launch_motion_viewer`.
- Link to `docs/motion_lib.md` for maintainer details.

The README will not duplicate installation guidance already owned by the repository root README.

### Maintainer Reference

Create `docs/motion_lib.md` as a curated maintainer reference. It will cover:

- Scope, terminology, architecture, and end-to-end data flow.
- Supported source formats and versioned reference motion artifacts.
- Public exports and class-by-class responsibilities for:
  - reference values and loaders;
  - `MotionLib` and its configuration;
  - motion managers and sampling state;
  - resampling and simulator enrichment;
  - versioned NPZ serialization and assembly;
  - viewing, scene adaptation, recording, and presentation adapters;
  - command-line composition roots.
- Important contracts and invariants, including canonical quaternion order, clip and axis metadata, valid clip length, stage boundaries, device-neutral serialization, and no-clobber publication.
- Extension guidance for adding formats, robots, viewers, and sampling behavior.
- A map from responsibilities to focused test files.
- Links to the accepted motion-library ADRs.

This page will explain public and important internal APIs selectively. It will not attempt to reproduce every signature or docstring.

## API Documentation Boundary

`docs/motion_lib.md` is a maintainer-oriented API guide, not a generated API catalog. A Sphinx or autodoc reference should be added later only when the repository has a documentation build and the public motion-library API is stable enough for generated reference pages to be useful.

This boundary keeps the first documentation increment readable and maintainable while leaving a clear path toward the layered guide-plus-API structure used by official mjlab.

## Existing Documentation Layout

User-facing documentation remains directly under `docs/`. Agent workflow material remains under `docs/agents/`; no existing agent documentation moves as part of this change. Root `AGENTS.md` and `CONTEXT.md` remain in place because repository tools and skills use them as entry points.

## Verification

Implementation verification will:

- Run each documented command module with `--help` using the repository environment.
- Check that all documented module names, options, paths, and cross-links exist.
- Check Markdown whitespace and review the final diff.
- Confirm that only the two requested documentation files are added during implementation.

No runtime tests are required because the implementation changes documentation only; existing CLI help output is the source of truth for command syntax.

## Out of Scope

- Runtime code, public exports, or behavior changes.
- New Sphinx, MkDocs, or generated API tooling.
- Changes to the domain glossary or accepted ADRs.
- Moving or restructuring existing agent documentation.
- Adding tutorials for task-specific RL integration beyond the current motion-library interfaces.
