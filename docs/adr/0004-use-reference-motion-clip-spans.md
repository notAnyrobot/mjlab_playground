---
status: accepted
---

# Use Reference Motion Clip Spans for Packed Logical Clips

The presentation-agnostic reference motion viewer decision keeps playback policy
inside a deep `MotionViewer` module and presentation details behind its existing
viewer and recording adapter seams. Packaged reference motions add logical clip
boundaries that both playback and deterministic recording must interpret. We will
represent those boundaries through the canonical
`ReferenceMotion.iter_clip_spans()` seam. Each lightweight span retains its packed
parent and owns clip identity, timing, lazy name access, and validated
clip-local-to-packed frame mapping without creating a clip-local reference motion
or tensor slice.

This refines the presentation-agnostic viewer architecture rather than moving
package knowledge into it. The seam provides leverage because interactive
playback, deterministic recording, and future callers share one boundary and
indexing contract. It provides locality because validation and frame mapping stay
with the canonical reference motion metadata instead of being repeated across
presentation and orchestration modules.

## Considered Options

- Add a viewer-private clip adapter: rejected because recording and other callers
  would need parallel package-indexing rules, weakening the viewer's deep module
  boundary.
- Materialize one display value per logical clip: rejected because it would copy
  or slice packed tensors and make presentation needs redefine the canonical
  loaded-value contract.
- Validate operational clip metadata during `ReferenceMotion` construction:
  rejected because canonical values intentionally remain passive and may be
  constructed for uses that do not navigate packaged clips.

## Consequences

- Metadata-free values produce one implicit full-length span; packaged values
  produce spans in stored metadata order over the original packed parent.
- Empty, partial, mismatched, noncontiguous, out-of-bounds, nonpositive, or
  incoherently named operational metadata is rejected only when span iteration is
  requested.
- Compact clip names remain CPU metadata and are exposed lazily by each span.
- Viewer and recording implementations reuse the span mapping. The existing
  `MotionViewer.run()` and viewer adapter seams remain unchanged; the internal
  background-recorder request is refined to carry the canonical outer reference
  motion index together with the selected span's package-local clip ID.
- Schema version 1 is unchanged because clip spans are runtime descriptors over
  its existing metadata, not persisted fields.
