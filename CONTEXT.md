# mjlab_playground

This context covers repository-local robotics motion data and reinforcement-learning extension work.

## Language

**PyRoki format**:
Retargeted robot motion data emitted by PyRoki as per-motion `.npz` artifacts. In this repo, the term refers to files containing root pose and joint-angle arrays for one robot motion.
_Avoid_: Proto format, ProtoMotions format, proto motion data

**ProtoMotions motion data**:
Motion artifacts produced for ProtoMotions consumption, such as `.motion` files or packaged motion-library `.pt` files.
_Avoid_: PyRoki format
