from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .motion_loader import ReferenceMotionState


class ReferenceMotionNpzWriter:
    """Write one rich reference clip to one explicit `.npz` path."""

    _REQUIRED_FIELDS = (
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

    def write(self, motion: ReferenceMotionState, output_path: str | Path) -> None:
        path = Path(output_path)
        if path.suffix != ".npz":
            raise ValueError(
                "ReferenceMotionNpzWriter output path must end with "
                f".npz: {path}"
            )
        self._validate_train_ready(motion)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            field_name: self._to_numpy(getattr(motion, field_name))
            for field_name in self._REQUIRED_FIELDS
        }
        if motion.body_contacts is not None:
            payload["body_contacts"] = self._to_numpy(motion.body_contacts)
        np.savez(path, **payload)

    def _validate_train_ready(self, motion: ReferenceMotionState) -> None:
        if motion.foot_contacts is not None:
            raise ValueError(
                "ReferenceMotionNpzWriter v1 does not serialize source foot_contacts; "
                "use body_contacts for train-ready robot contact labels"
            )
        for field_name in self._REQUIRED_FIELDS:
            if getattr(motion, field_name) is None:
                raise ValueError(
                    "ReferenceMotionNpzWriter requires train-ready rich reference "
                    f"field {field_name}"
                )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()
