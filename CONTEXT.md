# mjlab_playground

This context covers repository-local robotics motion data and reinforcement-learning extension work.

## Language

**PyRoki format**:
Retargeted robot motion data emitted by PyRoki as per-motion `.npz` artifacts. In this repo, the term refers to files containing root pose and joint-angle arrays for one robot motion.
_Avoid_: Proto format, ProtoMotions format, proto motion data

**ProtoMotions motion data**:
Motion artifacts produced for ProtoMotions consumption, such as `.motion` files or packaged motion-library `.pt` files.
_Avoid_: PyRoki format

**Source motion clip**:
The smallest trusted motion input before simulator enrichment. It may contain only generalized coordinates, such as root pose and joint angles, and is not assumed to contain body-state tensors.
_Avoid_: Full motion state, reference state

**Rich reference clip**:
A simulator-ready reference motion clip containing generalized coordinates plus derived body poses, body velocities, joint velocities, timing, and optional contacts.
_Avoid_: Raw clip, source clip

**Reference motion state**:
The frame or batched-frame query result consumed by tracking tasks. It represents the desired expert demonstration state, not the robot's current simulated state.
_Avoid_: Motion state, robot state

**Canonical reference quaternion**:
A normalized root or body quaternion stored in MuJoCo-facing <code>wxyz</code> order inside reference motion state. Source-format adapters own any <code>xyzw</code> to <code>wxyz</code> conversion before constructing reference motion state.
_Avoid_: Source quaternion, xyzw reference quaternion

**Reference motion clip**:
A time-ordered sequence of reference motion states for one expert demonstration. It may be sparse during visual validation or rich after simulator enrichment.
_Avoid_: Motion file, motion source

**Robot-specific reference motion clip**:
A reference motion clip whose generalized coordinates already target one concrete robot's kinematic layout, such as Astro root pose and Astro joint angles.
_Avoid_: Robot-agnostic motion clip

**Valid reference motion clip**:
A reference motion clip with at least three frames. One-frame and two-frame clips are rejected in v1 because they do not provide enough temporal context for the standard velocity convention.
_Avoid_: Single-frame reference, two-frame reference

**Recording attachment**:
An optional helper attached by a viewer runner to capture rendered frames and write video artifacts. It is not part of the source-agnostic motion viewer interface and may depend on offscreen rendering.
_Avoid_: Viewer recording mode, motion source recorder

**Reference motion viewer**:
A source-agnostic viewer for already-loaded reference motion clips. It plays, scrubs, switches, and optionally records clips without knowing which motion artifact format produced them.
_Avoid_: PyRoki viewer, motion file viewer, loader viewer

**Root-tracking camera**:
A viewer camera whose target follows the rendered robot root body during reference motion playback or recording while preserving user control over view angle and distance.
_Avoid_: Fixed world camera, source motion camera

**Reference motion resampler**:
An internal motion-library implementation detail that converts a reference motion clip to a target frame rate by interpolating generalized coordinates and deriving generalized-coordinate velocity fields. It does not load files, write files, or derive body-state tensors.
_Avoid_: Motion loader, simulator enricher, file converter

**Simulator enrichment**:
A preprocessing step owned by MotionLib that applies robot-specific generalized coordinates to a robot model or simulator to derive rich reference clip fields such as body poses and body velocities.
_Avoid_: Resampling, interpolation

**Generalized-coordinate velocity**:
A velocity field derived only from generalized-coordinate reference tensors, such as root position, root quaternion, and joint positions. These fields are produced by the reference motion resampler, not by simulator enrichment.
_Avoid_: Body velocity, simulator velocity

**MotionLib**:
The public motion-library module that loads source motion clips, aligns optional contact labels, internally resamples generalized coordinates, runs simulator enrichment, and returns rich reference clips for policy-training consumers.
_Avoid_: Resampler, viewer, file converter
