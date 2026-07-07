from __future__ import annotations

import dataclasses
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
    from mjlab_playground.motion_lib import ReferenceMotionState

    frames = 3
    fields = {
        "name": "walk_retargeted.npz",
        "display_name": "walk_retargeted",
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
    return ReferenceMotionState(**fields)


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
        }
        np.testing.assert_allclose(exported["root_pos"], motion.root_pos.numpy())
        np.testing.assert_allclose(exported["root_rot"], motion.root_rot.numpy())
        np.testing.assert_allclose(exported["dof_pos"], motion.dof_pos.numpy())
        np.testing.assert_allclose(exported["root_lin_vel"], motion.root_lin_vel.numpy())
        np.testing.assert_allclose(exported["root_ang_vel"], motion.root_ang_vel.numpy())
        np.testing.assert_allclose(exported["dof_vel"], motion.dof_vel.numpy())
        np.testing.assert_allclose(exported["body_pos"], motion.body_pos.numpy())
        np.testing.assert_allclose(exported["body_rot"], motion.body_rot.numpy())
        np.testing.assert_allclose(exported["body_lin_vel"], motion.body_lin_vel.numpy())
        np.testing.assert_allclose(exported["body_ang_vel"], motion.body_ang_vel.numpy())


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
