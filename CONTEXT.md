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

**Reference frame**:
One frame of expert demonstration data in robot coordinates. It represents desired reference root, joint, optional velocity, body-state, and contact values at one time step, not the robot's current simulated state.
_Avoid_: Motion state, robot state, simulator state

**Canonical reference quaternion**:
A normalized root or body quaternion stored in MuJoCo-facing <code>wxyz</code> order inside reference frames and reference motions. Source-format adapters own any <code>xyzw</code> to <code>wxyz</code> conversion before constructing reference motion data.
_Avoid_: Source quaternion, xyzw reference quaternion

**Reference motion**:
A tensor-backed collection of one or more reference-frame trajectories plus clip metadata such as clip starts, clip lengths, and clip FPS. A single source motion clip is represented as one-clip reference motion; packaged multi-clip data is represented by the same concept.
_Avoid_: Motion file, motion source, reference motion state

**Robot-specific reference motion clip**:
A reference motion trajectory whose generalized coordinates already target one concrete robot's kinematic layout, such as Astro root pose and Astro joint angles.
_Avoid_: Robot-agnostic motion clip

**Valid reference motion clip**:
A reference motion trajectory with at least three frames. One-frame and two-frame clips are rejected in v1 because they do not provide enough temporal context for the standard velocity convention.
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
A preprocessing step owned by MotionLib that applies already-resampled robot-specific generalized coordinates to a robot model or simulator to derive rich reference clip fields such as body poses and body velocities.
_Avoid_: Resampling, interpolation

**MuJoCo scene adapter**:
The motion-library module that owns a one-environment mjlab MuJoCo scene, robot entity, simulation step/update order, reference-frame application, display synchronization, root-tracking camera setup, and current robot-state reading. It does not own motion-level enrichment or package assembly.
_Avoid_: Motion viewer, source loader, resampler

**Reference body contract**:
The ordered robot body set used when simulator enrichment turns generalized-coordinate motion into policy-facing body trajectories and body contact labels. It follows the robot entity's resolved body order so body-indexed tensors and future metadata share one body axis.
_Avoid_: User-selected body names, viewer body list, AMP body subset

**Body contact labels**:
Optional per-frame contact labels indexed by the same body axis as the rich reference clip's body-state tensors. Policy consumers should read foot contacts through this body-indexed field once package metadata defines the body names and indices.
_Avoid_: Foot contacts, source contact labels

**Source foot contact labels**:
Optional per-frame left/right foot contact labels associated with source motion clips. They are source annotations, not the policy-facing contact contract. MotionLib v1 rejects them; MotionLib v2 may resample them and map them into body contact labels.
_Avoid_: Body contacts, simulator contacts

**Resampled contact probabilities**:
Deferred float contact labels produced by time-aware interpolation of source foot contact labels before robot-specific body-contact mapping.
_Avoid_: Hard contact labels, simulator-detected contacts, ProtoMotions OR contacts

**Packaged motion metadata**:
Dataset-level information, such as FPS, clip starts, clip lengths, body names, joint names, and schema version, stored with a future packaged motion artifact rather than repeated inside every per-clip train-ready `.npz`.
_Avoid_: Per-clip duplicated metadata, implicit body index contract

**Generalized-coordinate velocity**:
A velocity field derived only from generalized-coordinate reference tensors, such as root position, root quaternion, and joint positions. These fields are produced by the reference motion resampler, not by simulator enrichment.
_Avoid_: Body velocity, simulator velocity

**MotionLib**:
The public motion-library module that owns the source loading, resampling, and simulator enrichment pipeline behind explicit load, resample, and enrich methods. Its enrich method expects resampled clips and does not perform resampling itself.
_Avoid_: Resampler, viewer, file converter

**Motion sampling metadata**:
Dataset-level clip indexing information needed to sample reference motion time ranges, such as clip IDs, clip lengths, optional clip starts in a packaged flat tensor layout, clip weights, and valid start/end times.
_Avoid_: Source metadata, per-frame tensor payload

**Mimic motion track**:
Persistent per-environment reference playback state containing the selected reference motion clip ID and current time for a tracking task.
_Avoid_: AMP sample, expert batch

**Expert motion sample batch**:
Temporary motion IDs and motion times sampled from reference motion data to build discriminator expert observations for AMP or ASE-style training. It does not mutate mimic motion tracks.
_Avoid_: Mimic track, env playback state

**Motion sampler**:
The motion-library module that owns reference motion sampling policy, sampling weights, valid time windows, and optional per-environment mimic motion tracks. It does not load source files, resample clips, run simulator enrichment, or query motion tensors.
_Avoid_: MotionLib, MotionLoader, MotionCommand
