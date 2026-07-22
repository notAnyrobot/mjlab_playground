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
A tensor-backed collection of one or more reference-frame trajectories plus clip and axis metadata. A single source motion clip and an assembled multi-clip dataset use the same canonical concept.
_Avoid_: Motion file, motion source, reference motion state

**Reference motion clip span**:
A lightweight logical description of one clip inside a packed reference motion. It identifies the clip's boundaries, timing, and name while retaining the packed reference motion as the source of frames.
_Avoid_: Extracted reference motion, copied clip, display motion

**Reference motion assembly**:
The deterministic combination of compatible rich reference clips into one reference motion with contiguous clip metadata. It does not resample, enrich, repair, or serialize its inputs.
_Avoid_: MotionLib packaging, dataset conversion, implicit normalization

**Versioned reference motion artifact**:
A self-describing, device-neutral `.npz` serialization of one canonical reference motion, containing either one clip or multiple clips under the same schema.
_Avoid_: PyTorch checkpoint, `.pt` motion package, YAML motion manifest

**Legacy rich reference artifact**:
An existing unversioned, single-clip rich `.npz` artifact. It remains readable but is not an input to versioned reference motion assembly.
_Avoid_: Versioned reference motion artifact, packaged reference motion

**Robot-specific reference motion clip**:
A reference motion trajectory whose generalized coordinates already target one concrete robot's kinematic layout, such as Astro root pose and Astro joint angles.
_Avoid_: Robot-agnostic motion clip

**Valid reference motion clip**:
A reference motion trajectory with at least three frames. One-frame and two-frame clips are rejected in v1 because they do not provide enough temporal context for the standard velocity convention.
_Avoid_: Single-frame reference, two-frame reference

**Recording attachment**:
An internal helper used by the reference motion viewer to capture rendered frames and write video artifacts. It is not exposed to launch runners or other callers and may depend on offscreen rendering.
_Avoid_: Viewer recording mode, motion source recorder

**Background recording**:
A deterministic full-clip recording requested from an interactive reference motion viewer and executed independently of interactive playback. Interactive pause, speed, and frame position do not affect its traversal of the selected clip. It reuses headless recording internally but is distinct from headless batch mode, where no interactive viewer is running.
_Avoid_: Live recording, headless batch recording, paused viewer recording

**Reference motion viewer**:
A source- and presentation-agnostic viewer for already-loaded reference motion clips. It owns consistent playback, switching, status, and recording semantics while hiding the concrete interactive viewer and recording implementation from callers.
_Avoid_: PyRoki viewer, motion file viewer, loader viewer

**Root-tracking camera**:
A viewer camera whose target follows the rendered robot root body during reference motion playback or recording while preserving user control over view angle and distance.
_Avoid_: Fixed world camera, source motion camera

**Reference motion resampling**:
The transformation of a metadata-free source motion clip to a target frame rate by interpolating generalized coordinates and deriving generalized-coordinate velocity fields. It does not load files, write files, or derive body-state tensors.
_Avoid_: Reference motion resampler, simulator enrichment, file conversion

**Simulator enrichment**:
A preprocessing step owned by MotionLib that applies already-resampled robot-specific generalized coordinates to a robot model or simulator to derive rich reference clip fields such as body poses and body velocities.
_Avoid_: Resampling, interpolation

**Motion scene**:
An already-constructed robot scene that accepts reference frames and exposes the resulting robot state for viewing, recording, or simulator enrichment. Orchestration modules construct and inject it; reference motion viewers and MotionLib do not create it.
_Avoid_: Motion viewer, motion loader, viewer adapter

**MuJoCo scene adapter**:
The motion-library module that owns a one-environment mjlab MuJoCo scene, robot entity, simulation step/update order, reference-frame application, display synchronization, root-tracking camera setup, and current robot-state reading. It does not own motion-level enrichment or package assembly.
_Avoid_: Motion viewer, source loader, reference motion resampling

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
Artifact-level information including FPS, clip starts, clip lengths, clip names, body names, DOF names, and schema version. Every versioned reference motion artifact carries this metadata whether it contains one clip or many.
_Avoid_: Implicit axis order, external YAML metadata, joint names

**Generalized-coordinate velocity**:
A velocity field derived only from generalized-coordinate reference tensors, such as root position, root quaternion, and joint positions. These fields are produced during reference motion resampling, not by simulator enrichment.
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

**Astro getup curiosity state**:
The compact, training-only state used by random network distillation in the opt-in Astro getup experiment. It contains torso-frame projected gravity together with normalized torso and pelvis heights; it excludes policy-observation noise, history, actions, velocities, contacts, and joint posture.
_Avoid_: Actor observation, critic observation, full simulator state, RND embedding
