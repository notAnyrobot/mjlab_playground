# Reference Motion Library Design Resources

## Knowledge

- [ProtoMotions `motion_lib.py`](/home/android/Code/NVlabs/ProtoMotions/protomotions/components/motion_lib.py)
  Primary implementation reference for packaged motion tensor storage, YAML or directory loading, frame indexing, interpolation, contact handling, and `.pt` save/load behavior.
- [ProtoMotions `motion_manager.py`](/home/android/Code/NVlabs/ProtoMotions/protomotions/envs/motion_manager/motion_manager.py)
  Primary implementation reference for per-environment motion IDs, motion times, weighted sampling, subsets, exclusions, fixed IDs, and checkpointed curriculum weights.
- [ProtoMotions `motion_manager/config.py`](/home/android/Code/NVlabs/ProtoMotions/protomotions/envs/motion_manager/config.py)
  Configuration surface for reset-time behavior: initial-start probability, subset selection, exclusions, realignment, and mimic resampling.
- [ProtoMotions `mimic_motion_manager.py`](/home/android/Code/NVlabs/ProtoMotions/protomotions/envs/motion_manager/mimic_motion_manager.py)
  Small subclass showing where playback-time advancement and clip-done checks live for mimic environments.
- [ProtoMotions `base_env/env.py`](/home/android/Code/NVlabs/ProtoMotions/protomotions/envs/base_env/env.py)
  Environment integration reference: compatibility validation, manager creation, reset sequencing, reference-state queries, and checkpoint state handoff.

## Wisdom (Communities)

- Local source review with the `mjlab_playground` implementation.
  Use for: checking whether a ProtoMotions pattern should be copied, thinned, or replaced with a simpler repo-native boundary.

## Gaps

- No external paper or design doc was found in this pass. The current lesson is intentionally grounded in source code behavior.
