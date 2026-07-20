from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _identity_quat(frames: int, bodies: int | None = None) -> torch.Tensor:
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    if bodies is None:
        return quat.repeat(frames, 1)
    return quat.reshape(1, 1, 4).repeat(frames, bodies, 1)


def _rich_reference_motion(**overrides):
    from mjlab_playground.motion_lib import ReferenceMotion

    frames = 3
    fields = {
        "name": "walk_retargeted.npz",
        "fps": 50.0,
        "root_pos": torch.tensor(
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            dtype=torch.float32,
        ),
        "root_rot": _identity_quat(frames),
        "dof_pos": torch.tensor(
            [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
            dtype=torch.float32,
        ),
        "dof_names": ("left_hip", "right_hip"),
        "root_lin_vel": torch.tensor(
            [[3.0, 3.1, 3.2], [4.0, 4.1, 4.2], [5.0, 5.1, 5.2]],
            dtype=torch.float32,
        ),
        "root_ang_vel": torch.tensor(
            [[6.0, 6.1, 6.2], [7.0, 7.1, 7.2], [8.0, 8.1, 8.2]],
            dtype=torch.float32,
        ),
        "dof_vel": torch.tensor(
            [[9.0, 9.5], [10.0, 10.5], [11.0, 11.5]],
            dtype=torch.float32,
        ),
        "body_pos": torch.tensor(
            [
                [[12.0, 12.1, 12.2], [13.0, 13.1, 13.2]],
                [[14.0, 14.1, 14.2], [15.0, 15.1, 15.2]],
                [[16.0, 16.1, 16.2], [17.0, 17.1, 17.2]],
            ],
            dtype=torch.float32,
        ),
        "body_names": ("pelvis", "torso"),
        "body_rot": _identity_quat(frames, bodies=2),
        "body_lin_vel": torch.tensor(
            [
                [[18.0, 18.1, 18.2], [19.0, 19.1, 19.2]],
                [[20.0, 20.1, 20.2], [21.0, 21.1, 21.2]],
                [[22.0, 22.1, 22.2], [23.0, 23.1, 23.2]],
            ],
            dtype=torch.float32,
        ),
        "body_ang_vel": torch.tensor(
            [
                [[24.0, 24.1, 24.2], [25.0, 25.1, 25.2]],
                [[26.0, 26.1, 26.2], [27.0, 27.1, 27.2]],
                [[28.0, 28.1, 28.2], [29.0, 29.1, 29.2]],
            ],
            dtype=torch.float32,
        ),
    }
    fields.update(overrides)
    return ReferenceMotion(**fields)


def _rich_reference_motion_with_explicit_metadata(**overrides):
    encoded_name = b"walk.npz"
    fields = {
        "clip_starts": torch.tensor([0]),
        "clip_lengths": torch.tensor([3]),
        "clip_fps": torch.tensor([50.0]),
        "clip_name_bytes": torch.tensor(list(encoded_name), dtype=torch.uint8),
        "clip_name_offsets": torch.tensor([0, len(encoded_name)]),
    }
    fields.update(overrides)
    return dataclasses.replace(_rich_reference_motion(), **fields)


def test_reference_motion_npz_writer_writes_train_ready_tensor_payload(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = _rich_reference_motion()
    output_path = tmp_path / "new-dataset" / "walk.npz"

    ReferenceMotionNpzWriter().write(motion, output_path)

    with np.load(output_path) as exported:
        assert set(exported.files) == {
            "schema_version",
            "fps",
            "root_pos",
            "root_rot",
            "dof_pos",
            "root_lin_vel",
            "root_ang_vel",
            "dof_vel",
            "body_pos",
            "body_rot",
            "body_lin_vel",
            "body_ang_vel",
            "clip_starts",
            "clip_lengths",
            "clip_fps",
            "clip_name_bytes",
            "clip_name_offsets",
            "dof_names",
            "body_names",
        }
        assert exported["schema_version"].item() == 1
        assert exported["fps"].item() == 50.0
        np.testing.assert_array_equal(exported["clip_starts"], [0])
        np.testing.assert_array_equal(exported["clip_lengths"], [3])
        np.testing.assert_array_equal(exported["clip_fps"], [50.0])
        assert bytes(exported["clip_name_bytes"]).decode("utf-8") == motion.name
        np.testing.assert_array_equal(
            exported["clip_name_offsets"], [0, len(motion.name.encode("utf-8"))]
        )
        np.testing.assert_array_equal(exported["dof_names"], motion.dof_names)
        np.testing.assert_array_equal(exported["body_names"], motion.body_names)
        np.testing.assert_allclose(exported["root_pos"], motion.root_pos.numpy())
        np.testing.assert_allclose(exported["root_rot"], motion.root_rot.numpy())
        np.testing.assert_allclose(exported["dof_pos"], motion.dof_pos.numpy())
        np.testing.assert_allclose(
            exported["root_lin_vel"], motion.root_lin_vel.numpy()
        )
        np.testing.assert_allclose(
            exported["root_ang_vel"], motion.root_ang_vel.numpy()
        )
        np.testing.assert_allclose(exported["dof_vel"], motion.dof_vel.numpy())
        np.testing.assert_allclose(exported["body_pos"], motion.body_pos.numpy())
        np.testing.assert_allclose(exported["body_rot"], motion.body_rot.numpy())
        np.testing.assert_allclose(
            exported["body_lin_vel"], motion.body_lin_vel.numpy()
        )
        np.testing.assert_allclose(
            exported["body_ang_vel"], motion.body_ang_vel.numpy()
        )


def test_reference_motion_npz_writer_does_not_scan_passive_tensor_representation(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    root_lin_vel = torch.zeros(3, 3, dtype=torch.float64)
    root_lin_vel[1, 0] = float("nan")
    motion = _rich_reference_motion(
        root_pos=torch.zeros(3, 3, dtype=torch.float64),
        root_rot=torch.full((3, 4), 2.0),
        root_lin_vel=root_lin_vel,
        root_ang_vel=torch.zeros(3, 4),
        body_rot=torch.full((3, 2, 4), 3.0),
    )
    output_path = tmp_path / "passive-values.npz"

    ReferenceMotionNpzWriter().write(motion, output_path)

    with np.load(output_path, allow_pickle=False) as exported:
        assert exported["root_pos"].dtype == np.float64
        assert np.isnan(exported["root_lin_vel"][1, 0])
        assert exported["root_ang_vel"].shape == (3, 4)
        np.testing.assert_array_equal(exported["root_rot"], np.full((3, 4), 2.0))
        np.testing.assert_array_equal(exported["body_rot"], np.full((3, 2, 4), 3.0))


def test_versioned_single_clip_round_trips_through_public_writer_and_loader(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    body_contacts = torch.tensor(
        [[True, False], [False, True], [True, True]],
        dtype=torch.bool,
    )
    motion = _rich_reference_motion(
        name="walking-测试.npz",
        body_contacts=body_contacts,
    )
    output_path = tmp_path / "walk.npz"

    ReferenceMotionNpzWriter().write(motion, output_path)
    loaded_collection = MotionLoader.load(
        output_path,
        motion_format="mjlab",
        fps=30.0,
    )

    assert len(loaded_collection) == 1
    loaded = loaded_collection[0]
    assert loaded.name == output_path.name
    assert loaded.fps == 50.0
    torch.testing.assert_close(loaded.clip_starts, torch.tensor([0]))
    torch.testing.assert_close(loaded.clip_lengths, torch.tensor([3]))
    torch.testing.assert_close(loaded.clip_fps, torch.tensor([50.0]))
    assert loaded.clip_name(0) == "walking-测试.npz"
    (span,) = loaded.iter_clip_spans()
    assert span.parent is loaded
    assert span.clip_id == 0
    assert span.start_frame == 0
    assert span.frame_count == 3
    assert span.fps == 50.0
    assert span.name == "walking-测试.npz"
    assert span.to_packed_frame(2) == 2
    assert loaded.dof_names == ("left_hip", "right_hip")
    assert loaded.body_names == ("pelvis", "torso")
    assert loaded.dof_pos.shape[1] == len(loaded.dof_names)
    assert loaded.body_pos.shape[1] == len(loaded.body_names)
    assert loaded.root_pos.device.type == "cpu"
    assert loaded.clip_name_bytes.device.type == "cpu"
    for field_name in (
        "root_pos",
        "root_rot",
        "dof_pos",
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
    ):
        torch.testing.assert_close(
            getattr(loaded, field_name),
            getattr(motion, field_name),
        )
    torch.testing.assert_close(loaded.body_contacts, body_contacts)
    assert not hasattr(loaded, "schema_version")


def test_versioned_multi_clip_package_round_trips_through_writer_and_loader(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib import ReferenceMotion
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    first = _rich_reference_motion(
        name="walk.npz",
        body_contacts=torch.tensor(
            [[True, False], [False, True], [True, True]], dtype=torch.bool
        ),
    )
    second = _rich_reference_motion(
        name="turn-测试.npz",
        body_contacts=torch.tensor(
            [[False, False], [True, False], [False, True]], dtype=torch.bool
        ),
    )
    second = dataclasses.replace(
        second,
        **{
            field_name: torch.cat(
                [getattr(second, field_name), getattr(second, field_name)[-1:]],
                dim=0,
            )
            for field_name in (
                "root_pos",
                "root_rot",
                "dof_pos",
                "root_lin_vel",
                "root_ang_vel",
                "dof_vel",
                "body_pos",
                "body_rot",
                "body_lin_vel",
                "body_ang_vel",
                "body_contacts",
            )
        },
    )
    package = ReferenceMotion.from_clips([first, second])
    output_path = tmp_path / "package.npz"

    ReferenceMotionNpzWriter().write(package, output_path)
    loaded_collection = MotionLoader.load(output_path, motion_format="mjlab")

    assert len(loaded_collection) == 1
    loaded = loaded_collection[0]
    assert loaded.name == output_path.name
    assert loaded.fps == 50.0
    assert loaded.root_pos.shape[0] == 7
    torch.testing.assert_close(loaded.clip_starts, torch.tensor([0, 3]))
    torch.testing.assert_close(loaded.clip_lengths, torch.tensor([3, 4]))
    torch.testing.assert_close(loaded.clip_fps, torch.tensor([50.0, 50.0]))
    assert [loaded.clip_name(clip_id) for clip_id in range(2)] == [
        "walk.npz",
        "turn-测试.npz",
    ]
    assert loaded.dof_names == package.dof_names
    assert loaded.body_names == package.body_names
    for field_name in (
        "root_pos",
        "root_rot",
        "dof_pos",
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
        "body_contacts",
    ):
        torch.testing.assert_close(
            getattr(loaded, field_name), getattr(package, field_name)
        )


def test_runnable_writer_packages_one_versioned_artifact_then_loads_one_clip(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
        main,
    )

    body_contacts = torch.tensor(
        [[True, False], [False, True], [True, True]], dtype=torch.bool
    )
    motion = _rich_reference_motion(
        name="walk-测试.npz",
        body_contacts=body_contacts,
    )
    input_path = tmp_path / "individual.npz"
    output_path = tmp_path / "package.npz"
    ReferenceMotionNpzWriter().write(motion, input_path)

    main(["--input", str(input_path), "--output", str(output_path)])
    loaded_collection = MotionLoader.load(output_path, motion_format="mjlab")

    assert len(loaded_collection) == 1
    loaded = loaded_collection[0]
    torch.testing.assert_close(loaded.clip_starts, torch.tensor([0]))
    torch.testing.assert_close(loaded.clip_lengths, torch.tensor([3]))
    torch.testing.assert_close(loaded.clip_fps, torch.tensor([50.0]))
    assert loaded.clip_name(0) == "walk-测试.npz"
    assert loaded.dof_names == motion.dof_names
    assert loaded.body_names == motion.body_names
    for field_name in (
        "root_pos",
        "root_rot",
        "dof_pos",
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
        "body_contacts",
    ):
        torch.testing.assert_close(
            getattr(loaded, field_name), getattr(motion, field_name)
        )


def test_runnable_writer_packages_versioned_individual_artifacts_in_loader_order(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
        main,
    )

    input_dir = tmp_path / "individual"
    input_dir.mkdir()
    writer = ReferenceMotionNpzWriter()
    walk = _rich_reference_motion(
        name="walk.npz",
        body_contacts=torch.tensor(
            [[True, False], [False, True], [True, True]], dtype=torch.bool
        ),
    )
    turn = _rich_reference_motion(
        name="turn-测试.npz",
        body_contacts=torch.tensor(
            [[False, False], [True, False], [False, True]], dtype=torch.bool
        ),
    )
    turn = dataclasses.replace(
        turn,
        **{
            field_name: torch.cat(
                [getattr(turn, field_name), getattr(turn, field_name)[-1:]],
                dim=0,
            )
            for field_name in (
                "root_pos",
                "root_rot",
                "dof_pos",
                "root_lin_vel",
                "root_ang_vel",
                "dof_vel",
                "body_pos",
                "body_rot",
                "body_lin_vel",
                "body_ang_vel",
                "body_contacts",
            )
        },
    )
    writer.write(turn, input_dir / "b_turn.npz")
    writer.write(walk, input_dir / "a_walk.npz")
    output_path = tmp_path / "package.npz"

    main(["--input", str(input_dir), "--output", str(output_path)])
    loaded_collection = MotionLoader.load(output_path, motion_format="mjlab")

    assert len(loaded_collection) == 1
    loaded = loaded_collection[0]
    assert loaded.root_pos.device.type == "cpu"
    assert loaded.clip_name_bytes.device.type == "cpu"
    torch.testing.assert_close(loaded.clip_starts, torch.tensor([0, 3]))
    torch.testing.assert_close(loaded.clip_lengths, torch.tensor([3, 4]))
    torch.testing.assert_close(loaded.clip_fps, torch.tensor([50.0, 50.0]))
    assert [loaded.clip_name(clip_id) for clip_id in range(2)] == [
        "walk.npz",
        "turn-测试.npz",
    ]
    assert loaded.dof_names == ("left_hip", "right_hip")
    assert loaded.body_names == ("pelvis", "torso")
    for field_name in (
        "root_pos",
        "root_rot",
        "dof_pos",
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
        "body_contacts",
    ):
        torch.testing.assert_close(
            getattr(loaded, field_name),
            torch.cat([getattr(walk, field_name), getattr(turn, field_name)], dim=0),
        )


def test_runnable_writer_names_incompatible_input_and_leaves_no_artifact(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
        package_reference_motions,
    )

    input_dir = tmp_path / "individual"
    input_dir.mkdir()
    writer = ReferenceMotionNpzWriter()
    writer.write(_rich_reference_motion(name="good.npz"), input_dir / "a_good.npz")
    writer.write(
        _rich_reference_motion(
            name="bad.npz",
            dof_names=("right_hip", "left_hip"),
        ),
        input_dir / "b_bad.npz",
    )
    output_path = tmp_path / "package.npz"

    with pytest.raises(
        ValueError,
        match=r"clip 1=.*b_bad\.npz.*clip 1 dof_names",
    ):
        package_reference_motions(input_dir, output_path)

    assert not output_path.exists()


def test_runnable_writer_rejects_legacy_artifact_and_leaves_no_output(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        package_reference_motions,
    )

    motion = _rich_reference_motion()
    legacy_path = tmp_path / "legacy.npz"
    np.savez(
        legacy_path,
        **{
            field_name: getattr(motion, field_name).numpy()
            for field_name in (
                "root_pos",
                "root_rot",
                "dof_pos",
                "root_lin_vel",
                "root_ang_vel",
                "dof_vel",
                "body_pos",
                "body_rot",
                "body_lin_vel",
                "body_ang_vel",
            )
        },
    )
    output_path = tmp_path / "package.npz"

    with pytest.raises(
        ValueError,
        match=r"legacy\.npz.*schema version 1.*legacy rich reference artifacts",
    ):
        package_reference_motions(legacy_path, output_path)

    assert not output_path.exists()


def test_runnable_writer_rejects_already_multi_clip_artifact(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib import ReferenceMotion
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
        package_reference_motions,
    )

    package_input = tmp_path / "already-packaged.npz"
    ReferenceMotionNpzWriter().write(
        ReferenceMotion.from_clips(
            [
                _rich_reference_motion(name="walk.npz"),
                _rich_reference_motion(name="turn.npz"),
            ]
        ),
        package_input,
    )
    output_path = tmp_path / "repackaged.npz"

    with pytest.raises(
        ValueError,
        match=r"already-packaged\.npz.*contains multiple clips",
    ):
        package_reference_motions(package_input, output_path)

    assert not output_path.exists()


def test_runnable_writer_protects_output_until_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
        main,
    )

    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "package.npz"
    writer = ReferenceMotionNpzWriter()
    writer.write(_rich_reference_motion(), input_path)
    main(["--input", str(input_path), "--output", str(output_path)])
    original_contents = output_path.read_bytes()
    replacement_root_pos = torch.full((3, 3), 42.0)
    writer.write(
        _rich_reference_motion(root_pos=replacement_root_pos),
        input_path,
        overwrite=True,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        main(["--input", str(input_path), "--output", str(output_path)])
    assert output_path.read_bytes() == original_contents

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    )
    loaded = MotionLoader.load(output_path, motion_format="mjlab")[0]
    torch.testing.assert_close(loaded.root_pos, replacement_root_pos)


def test_runnable_writer_cli_defaults_to_cpu_and_parses_overwrite() -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import parse_args

    defaults = parse_args(["--input", "/tmp/input", "--output", "/tmp/output.npz"])
    overwrite = parse_args(
        [
            "--input",
            "/tmp/input",
            "--output",
            "/tmp/output.npz",
            "--device",
            "cuda:1",
            "--overwrite",
        ]
    )

    assert defaults.input == Path("/tmp/input")
    assert defaults.output == Path("/tmp/output.npz")
    assert defaults.device == "cpu"
    assert defaults.overwrite is False
    assert overwrite.device == "cuda:1"
    assert overwrite.overwrite is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_runnable_writer_honors_cuda_processing_and_writes_device_neutral_artifact(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
        package_reference_motions,
    )

    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "package.npz"
    ReferenceMotionNpzWriter().write(_rich_reference_motion(), input_path)

    package_reference_motions(input_path, output_path, device="cuda")
    loaded = MotionLoader.load(output_path, motion_format="mjlab", device="cpu")[0]

    assert loaded.root_pos.device.type == "cpu"
    assert loaded.clip_name_bytes.device.type == "cpu"


def test_reference_motion_npz_writer_module_is_directly_runnable() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(root / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mjlab_playground.motion_lib.reference_motion_npz_writer",
            "--help",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--device" in result.stdout
    assert "--overwrite" in result.stdout
    assert "reference motion assembly" in result.stdout
    assert "versioned reference motion artifact" in result.stdout
    assert "Versioned individual .npz" not in result.stdout
    assert "multi-clip .npz artifact" not in result.stdout


def test_reference_motion_npz_writer_rejects_non_npz_output_path(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.txt"

    with pytest.raises(ValueError, match=r"\.npz"):
        ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)

    assert not output_path.exists()


@pytest.mark.parametrize(
    "field_name",
    [
        "root_pos",
        "root_rot",
        "dof_pos",
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
    ],
)
def test_reference_motion_npz_writer_rejects_missing_train_ready_fields(
    tmp_path: Path,
    field_name: str,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = _rich_reference_motion()
    object.__setattr__(motion, field_name, None)

    with pytest.raises(ValueError, match=field_name):
        ReferenceMotionNpzWriter().write(motion, tmp_path / "walk.npz")


def test_reference_motion_npz_writer_writes_optional_body_contacts(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    body_contacts = torch.tensor(
        [[True, False], [False, True], [True, True]],
        dtype=torch.bool,
    )
    motion = dataclasses.replace(_rich_reference_motion(), body_contacts=body_contacts)
    output_path = tmp_path / "walk.npz"

    ReferenceMotionNpzWriter().write(motion, output_path)

    with np.load(output_path) as exported:
        assert "body_contacts" in exported.files
        np.testing.assert_array_equal(exported["body_contacts"], body_contacts.numpy())


def test_reference_motion_npz_writer_rejects_source_foot_contacts(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = dataclasses.replace(
        _rich_reference_motion(),
        foot_contacts=torch.zeros(3, 2),
    )

    with pytest.raises(ValueError, match="foot_contacts"):
        ReferenceMotionNpzWriter().write(motion, tmp_path / "walk.npz")


@pytest.mark.parametrize("field_name", ["dof_names", "body_names"])
def test_reference_motion_npz_writer_requires_axis_names(
    tmp_path: Path,
    field_name: str,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = dataclasses.replace(_rich_reference_motion(), **{field_name: None})

    with pytest.raises(ValueError, match=field_name):
        ReferenceMotionNpzWriter().write(motion, tmp_path / "walk.npz")


def test_reference_motion_npz_writer_rejects_partial_operational_clip_metadata(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = dataclasses.replace(
        _rich_reference_motion(),
        clip_starts=torch.tensor([0]),
    )
    output_path = tmp_path / "walk.npz"

    with pytest.raises(
        ValueError,
        match="operational clip metadata must be provided together",
    ):
        ReferenceMotionNpzWriter().write(motion, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("metadata", "expected_error"),
    [
        pytest.param(
            {"clip_starts": torch.tensor([1])},
            "clip spans must start at frame 0",
            id="first-start",
        ),
        pytest.param(
            {"clip_lengths": torch.tensor([4])},
            "clip spans must cover packed frame count 3, got 4",
            id="packed-frame-coverage",
        ),
        pytest.param(
            {"clip_fps": torch.tensor([60.0])},
            "schema version 1 requires one common FPS",
            id="common-fps",
        ),
        pytest.param(
            {"clip_starts": torch.tensor([0, 3])},
            "clip metadata tensors must have matching lengths",
            id="metadata-count",
        ),
    ],
)
def test_reference_motion_npz_writer_rejects_incoherent_clip_spans_and_fps(
    tmp_path: Path,
    metadata: dict[str, torch.Tensor],
    expected_error: str,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = _rich_reference_motion_with_explicit_metadata(**metadata)
    output_path = tmp_path / "walk.npz"

    with pytest.raises(ValueError, match=expected_error):
        ReferenceMotionNpzWriter().write(motion, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("fps", "expected_error"),
    [
        pytest.param(0.0, "fps must be positive and finite", id="zero"),
        pytest.param(float("inf"), "fps must be positive and finite", id="infinite"),
        pytest.param(29.97, "fps must be integer-valued", id="fractional"),
    ],
)
def test_reference_motion_npz_writer_rejects_invalid_common_fps(
    tmp_path: Path,
    fps: float,
    expected_error: str,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = dataclasses.replace(_rich_reference_motion(), fps=fps)
    output_path = tmp_path / "walk.npz"

    with pytest.raises(ValueError, match=expected_error):
        ReferenceMotionNpzWriter().write(motion, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("metadata", "expected_error"),
    [
        pytest.param(
            {
                "clip_starts": torch.tensor([[0]]),
                "clip_lengths": torch.tensor([[3]]),
                "clip_fps": torch.tensor([[50.0]]),
            },
            "clip metadata tensors must be 1D",
            id="rank",
        ),
        pytest.param(
            {"clip_lengths": torch.tensor([3.5])},
            "clip_starts and clip_lengths must be integer tensors",
            id="integer-spans",
        ),
    ],
)
def test_reference_motion_npz_writer_rejects_noncanonical_clip_span_metadata(
    tmp_path: Path,
    metadata: dict[str, torch.Tensor],
    expected_error: str,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = _rich_reference_motion_with_explicit_metadata(**metadata)
    output_path = tmp_path / "walk.npz"

    with pytest.raises(ValueError, match=expected_error):
        ReferenceMotionNpzWriter().write(motion, output_path)

    assert not output_path.exists()


def test_reference_motion_npz_writer_rejects_noncontiguous_multi_clip_spans(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib import ReferenceMotion
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    package = ReferenceMotion.from_clips(
        [
            _rich_reference_motion(name="walk.npz"),
            _rich_reference_motion(name="turn.npz"),
        ]
    )
    package = dataclasses.replace(package, clip_starts=torch.tensor([0, 2]))
    output_path = tmp_path / "package.npz"

    with pytest.raises(ValueError, match="clip spans must be contiguous"):
        ReferenceMotionNpzWriter().write(package, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("metadata", "expected_error"),
    [
        pytest.param(
            {"clip_name_offsets": torch.tensor([0])},
            "clip_name_offsets must contain one more entry than the clip count",
            id="offset-count",
        ),
        pytest.param(
            {"clip_name_offsets": torch.tensor([0, 7])},
            "clip_name_offsets must span clip_name_bytes length 8, got end 7",
            id="byte-span",
        ),
        pytest.param(
            {
                "clip_name_bytes": torch.tensor([0xFF], dtype=torch.uint8),
                "clip_name_offsets": torch.tensor([0, 1]),
            },
            "clip 0 name is not valid UTF-8",
            id="utf8",
        ),
        pytest.param(
            {"clip_name_bytes": torch.tensor(list(b"walk.npz"))},
            "clip_name_bytes must be a 1D uint8 CPU tensor",
            id="compact-bytes",
        ),
    ],
)
def test_reference_motion_npz_writer_rejects_incoherent_clip_identity_metadata(
    tmp_path: Path,
    metadata: dict[str, torch.Tensor],
    expected_error: str,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = _rich_reference_motion_with_explicit_metadata(**metadata)
    output_path = tmp_path / "walk.npz"

    with pytest.raises(ValueError, match=expected_error):
        ReferenceMotionNpzWriter().write(motion, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("names", "expected_error"),
    [
        pytest.param(
            {"dof_names": ("left_hip",)},
            "dof_names count must match dof_pos axis 2, got 1",
            id="dof-count",
        ),
        pytest.param(
            {"body_names": ("pelvis",)},
            "body_names count must match body_pos axis 2, got 1",
            id="body-count",
        ),
        pytest.param(
            {"dof_names": ("left_hip", "")},
            "dof_names must contain non-empty strings",
            id="non-empty",
        ),
        pytest.param(
            {"body_names": ("pelvis", chr(0xD800))},
            "body_names must contain valid UTF-8 strings",
            id="utf8",
        ),
    ],
)
def test_reference_motion_npz_writer_rejects_incoherent_axis_names(
    tmp_path: Path,
    names: dict[str, tuple[str, ...]],
    expected_error: str,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = dataclasses.replace(_rich_reference_motion(), **names)
    output_path = tmp_path / "walk.npz"

    with pytest.raises(ValueError, match=expected_error):
        ReferenceMotionNpzWriter().write(motion, output_path)

    assert not output_path.exists()


def test_reference_motion_npz_writer_rejects_rich_clip_with_fewer_than_three_frames(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    motion = _rich_reference_motion()
    short_motion = dataclasses.replace(
        motion,
        **{
            field_name: getattr(motion, field_name)[:2]
            for field_name in (
                "root_pos",
                "root_rot",
                "dof_pos",
                "root_lin_vel",
                "root_ang_vel",
                "dof_vel",
                "body_pos",
                "body_rot",
                "body_lin_vel",
                "body_ang_vel",
            )
        },
    )

    with pytest.raises(ValueError, match="at least 3 frames"):
        ReferenceMotionNpzWriter().write(short_motion, tmp_path / "walk.npz")


def test_reference_motion_npz_writer_protects_existing_output_by_default(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    original_contents = b"existing artifact"
    output_path.write_bytes(original_contents)

    with pytest.raises(FileExistsError, match="already exists"):
        ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)

    assert output_path.read_bytes() == original_contents


def test_reference_motion_npz_writer_does_not_replace_output_created_during_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    competing_contents = b"artifact published by another writer"
    real_savez = np.savez

    def serialize_while_another_writer_publishes(target, *args, **kwargs) -> None:
        real_savez(target, *args, **kwargs)
        output_path.write_bytes(competing_contents)

    monkeypatch.setattr(
        "mjlab_playground.motion_lib.reference_motion_npz_writer.np.savez",
        serialize_while_another_writer_publishes,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)

    assert output_path.read_bytes() == competing_contents
    assert set(tmp_path.iterdir()) == {output_path}


def test_reference_motion_npz_writer_replaces_output_only_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    writer = ReferenceMotionNpzWriter()
    writer.write(_rich_reference_motion(), output_path)
    replacement_root_pos = torch.full((3, 3), 42.0)

    writer.write(
        _rich_reference_motion(root_pos=replacement_root_pos),
        output_path,
        overwrite=True,
    )

    with np.load(output_path, allow_pickle=False) as exported:
        np.testing.assert_array_equal(
            exported["root_pos"], replacement_root_pos.numpy()
        )


def test_reference_motion_npz_writer_failed_serialization_leaves_no_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "new-dataset" / "walk.npz"
    serialization_parent: Path | None = None

    def fail_after_partial_write(target, *args, **kwargs) -> None:
        nonlocal serialization_parent
        target_path = (
            Path(target) if isinstance(target, (str, Path)) else Path(target.name)
        )
        serialization_parent = target_path.parent
        if hasattr(target, "write"):
            target.write(b"partial artifact")
        else:
            target_path.write_bytes(b"partial artifact")
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(
        "mjlab_playground.motion_lib.reference_motion_npz_writer.np.savez",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated serialization failure"):
        ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)

    assert serialization_parent == output_path.parent
    assert not output_path.exists()
    assert list(output_path.parent.iterdir()) == []


def test_reference_motion_npz_writer_failed_overwrite_preserves_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    writer = ReferenceMotionNpzWriter()
    writer.write(_rich_reference_motion(), output_path)
    original_contents = output_path.read_bytes()

    def fail_after_partial_write(target, *args, **kwargs) -> None:
        target.write(b"partial replacement")
        raise OSError("simulated overwrite failure")

    monkeypatch.setattr(
        "mjlab_playground.motion_lib.reference_motion_npz_writer.np.savez",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated overwrite failure"):
        writer.write(
            _rich_reference_motion(root_pos=torch.full((3, 3), 42.0)),
            output_path,
            overwrite=True,
        )

    assert output_path.read_bytes() == original_contents
    assert set(tmp_path.iterdir()) == {output_path}


def test_motion_loader_rejects_unsupported_reference_motion_schema(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)
    with np.load(output_path, allow_pickle=False) as exported:
        payload = {key: exported[key] for key in exported.files}
    payload["schema_version"] = np.asarray(2, dtype=np.int64)
    np.savez(output_path, **payload)

    with pytest.raises(
        ValueError, match="Unsupported reference motion schema version 2"
    ):
        MotionLoader.load(output_path, motion_format="mjlab")


@pytest.mark.parametrize(
    "missing_key",
    [
        "root_pos",
        "root_rot",
        "dof_pos",
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_lin_vel",
        "body_ang_vel",
    ],
)
def test_motion_loader_rejects_versioned_artifact_missing_required_tensor_key(
    tmp_path: Path,
    missing_key: str,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)
    with np.load(output_path, allow_pickle=False) as exported:
        payload = {key: exported[key] for key in exported.files if key != missing_key}
    np.savez(output_path, **payload)

    with pytest.raises(ValueError, match=missing_key):
        MotionLoader.load(output_path, motion_format="mjlab")


@pytest.mark.parametrize(
    "missing_key",
    [
        "fps",
        "clip_starts",
        "clip_lengths",
        "clip_fps",
        "clip_name_bytes",
        "clip_name_offsets",
        "dof_names",
        "body_names",
    ],
)
def test_motion_loader_rejects_versioned_artifact_missing_required_metadata_key(
    tmp_path: Path,
    missing_key: str,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)
    with np.load(output_path, allow_pickle=False) as exported:
        payload = {key: exported[key] for key in exported.files if key != missing_key}
    np.savez(output_path, **payload)

    with pytest.raises(ValueError, match=missing_key):
        MotionLoader.load(output_path, motion_format="mjlab")


@pytest.mark.parametrize(
    ("metadata_key", "malformed_value", "expected_error"),
    [
        pytest.param(
            "clip_starts",
            np.asarray([[0]], dtype=np.int64),
            "clip_starts.*1D integer array",
            id="two-dimensional-starts",
        ),
        pytest.param(
            "clip_lengths",
            np.asarray([3, 3], dtype=np.int64),
            "metadata tensors must have matching lengths",
            id="length-count",
        ),
        pytest.param(
            "clip_fps",
            np.asarray([50.0, 50.0], dtype=np.float32),
            "metadata tensors must have matching lengths",
            id="fps-count",
        ),
        pytest.param(
            "clip_name_offsets",
            np.asarray([0], dtype=np.int64),
            "one more entry than the clip count",
            id="name-offset-count",
        ),
    ],
)
def test_motion_loader_rejects_malformed_versioned_clip_metadata(
    tmp_path: Path,
    metadata_key: str,
    malformed_value: np.ndarray,
    expected_error: str,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)
    with np.load(output_path, allow_pickle=False) as exported:
        payload = {key: exported[key] for key in exported.files}
    payload[metadata_key] = malformed_value
    np.savez(output_path, **payload)

    with pytest.raises(ValueError, match=expected_error):
        MotionLoader.load(output_path, motion_format="mjlab")


def test_motion_loader_rejects_malformed_versioned_axis_metadata(
    tmp_path: Path,
) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)
    with np.load(output_path, allow_pickle=False) as exported:
        payload = {key: exported[key] for key in exported.files}
    payload["dof_names"] = np.asarray(["left_hip"])
    np.savez(output_path, **payload)

    with pytest.raises(ValueError, match="dof_names count must match"):
        MotionLoader.load(output_path, motion_format="mjlab")


def test_motion_loader_rejects_invalid_utf8_clip_identity(tmp_path: Path) -> None:
    from mjlab_playground.motion_lib.motion_loader import MotionLoader
    from mjlab_playground.motion_lib.reference_motion_npz_writer import (
        ReferenceMotionNpzWriter,
    )

    output_path = tmp_path / "walk.npz"
    ReferenceMotionNpzWriter().write(_rich_reference_motion(), output_path)
    with np.load(output_path, allow_pickle=False) as exported:
        payload = {key: exported[key] for key in exported.files}
    payload["clip_name_bytes"] = np.asarray([0xFF], dtype=np.uint8)
    payload["clip_name_offsets"] = np.asarray([0, 1], dtype=np.int64)
    np.savez(output_path, **payload)

    with pytest.raises(ValueError, match="clip 0 name is not valid UTF-8"):
        MotionLoader.load(output_path, motion_format="mjlab")
