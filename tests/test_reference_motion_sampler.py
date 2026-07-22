from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MOTION_LIB_DOC = ROOT / "docs" / "motion_lib.md"
MOTION_LIB_README = SRC / "mjlab_playground" / "motion_lib" / "README.md"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _identity_root_rot(num_frames: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_frames, 1)


def _package_reference_motion(*, clip_lengths: list[int] | None = None):
    from mjlab_playground.motion_lib import ReferenceMotion

    lengths = torch.tensor(clip_lengths or [6, 11, 16], dtype=torch.long)
    starts = torch.empty_like(lengths)
    starts[0] = 0
    starts[1:] = torch.cumsum(lengths[:-1], dim=0)
    frame_count = int(lengths.sum().item())
    frame_values = torch.arange(frame_count, dtype=torch.float32)
    return ReferenceMotion(
        name="package",
        fps=10.0,
        root_pos=frame_values.unsqueeze(1).repeat(1, 3),
        root_rot=_identity_root_rot(frame_count),
        dof_pos=frame_values.unsqueeze(1).repeat(1, 2),
        clip_starts=starts,
        clip_lengths=lengths,
        clip_fps=torch.full((lengths.numel(),), 10.0),
    )


def test_motion_lib_public_exports_sampler_surface() -> None:
    import mjlab_playground.motion_lib as motion_lib_package
    from mjlab_playground.motion_lib import (
        ClipWeighting,
        MimicMotionManager,
        MotionManager,
        MotionManagerCfg,
        ReferenceMotionSample,
        TimeSampling,
    )

    expected_exports = {
        "ClipWeighting",
        "MimicMotionManager",
        "ReferenceMotionSample",
        "MotionManager",
        "MotionManagerCfg",
        "TimeSampling",
    }

    assert expected_exports <= set(motion_lib_package.__all__)
    assert ClipWeighting is not None
    assert TimeSampling is not None
    assert dataclasses.is_dataclass(ReferenceMotionSample)
    assert dataclasses.is_dataclass(MotionManagerCfg)
    assert issubclass(MimicMotionManager, MotionManager)


def test_motion_manager_module_is_the_public_source_module() -> None:
    from mjlab_playground.motion_lib.motion_manager import (
        MimicMotionManager,
        MotionManager,
        MotionManagerCfg,
        ReferenceMotionSample,
    )

    assert dataclasses.is_dataclass(ReferenceMotionSample)
    assert dataclasses.is_dataclass(MotionManagerCfg)
    assert issubclass(MimicMotionManager, MotionManager)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mjlab_playground.motion_lib.motion_sampler")


def test_motion_lib_does_not_export_old_sampler_class_names() -> None:
    import mjlab_playground.motion_lib as motion_lib_package

    assert "ReferenceMotionSampler" not in motion_lib_package.__all__
    assert "MimicReferenceMotionSampler" not in motion_lib_package.__all__
    with pytest.raises(ImportError):
        from mjlab_playground.motion_lib import ReferenceMotionSampler  # noqa: F401
    with pytest.raises(ImportError):
        from mjlab_playground.motion_lib import (
            MimicReferenceMotionSampler,  # noqa: F401
        )


def test_motion_lib_docs_show_motion_manager_usage_and_v1_boundaries() -> None:
    assert MOTION_LIB_DOC.exists()
    assert MOTION_LIB_README.exists()
    assert not (ROOT / "docs" / "motion_manager.md").exists()
    assert not (ROOT / "docs" / "motion_sampler.md").exists()
    text = MOTION_LIB_DOC.read_text(encoding="utf-8")
    required = [
        "ReferenceMotion",
        "clip_starts",
        "clip_lengths",
        "clip_fps",
        "MotionManager",
        "sample_batch",
        "MimicMotionManager",
        "sample_envs",
        "advance_envs",
        "done_envs",
        "MotionLib",
        "query",
        "motion_ids",
        "motion_times",
        "sampler output is motion IDs and motion times only",
        "MotionLib owns query and interpolation",
        "rewind sampling",
        "contact-label resampling",
        "viewer integration",
        "packaged artifact metadata",
    ]
    for needle in required:
        assert needle in text

    readme = MOTION_LIB_README.read_text(encoding="utf-8")
    for needle in ("sample_batch", "sample_envs", "advance_envs", "done_envs"):
        assert needle in readme


def test_motion_manager_returns_valid_start_times_from_windowed_package() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    sampler = MotionManager(
        _package_reference_motion(),
        MotionManagerCfg(
            clip_weighting="uniform",
            time_sampling="start",
            history_seconds=0.2,
            future_seconds=0.3,
            seed=7,
        ),
    )

    sample = sampler.sample_batch(8)

    assert dataclasses.is_dataclass(sample)
    assert set(sample.__dataclass_fields__) == {"motion_ids", "motion_times"}
    assert sample.motion_ids.shape == (8,)
    assert sample.motion_times.shape == (8,)
    torch.testing.assert_close(
        sample.motion_times,
        torch.full((8,), 0.2),
    )


def test_motion_manager_treats_missing_package_metadata_as_single_clip() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    motion = dataclasses.replace(
        _package_reference_motion(clip_lengths=[6]),
        clip_starts=None,
        clip_lengths=None,
        clip_fps=None,
    )
    sampler = MotionManager(
        motion,
        MotionManagerCfg(time_sampling="start", history_seconds=0.2),
    )

    sample = sampler.sample_batch(4)

    torch.testing.assert_close(sample.motion_ids, torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(sample.motion_times, torch.full((4,), 0.2))


def test_motion_manager_duration_weighting_favors_longer_valid_windows() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    sampler = MotionManager(
        _package_reference_motion(),
        MotionManagerCfg(
            clip_weighting="duration",
            time_sampling="start",
            seed=11,
        ),
    )

    sample = sampler.sample_batch(6000)
    counts = torch.bincount(sample.motion_ids, minlength=3)

    assert counts[2] > counts[1] > counts[0]


def test_motion_manager_uniform_weighting_samples_valid_clips_equally() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    sampler = MotionManager(
        _package_reference_motion(),
        MotionManagerCfg(
            clip_weighting="uniform",
            time_sampling="start",
            seed=19,
        ),
    )

    sample = sampler.sample_batch(6000)
    counts = torch.bincount(sample.motion_ids, minlength=3)

    assert int(counts.max() - counts.min()) < 400


def test_motion_manager_explicit_weighting_masks_invalid_clips() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    sampler = MotionManager(
        _package_reference_motion(clip_lengths=[3, 11, 16]),
        MotionManagerCfg(
            clip_weighting="explicit",
            time_sampling="start",
            history_seconds=0.2,
            future_seconds=0.1,
            clip_weights=torch.tensor([1000.0, 1.0, 0.0]),
            seed=13,
        ),
    )

    sample = sampler.sample_batch(32)

    torch.testing.assert_close(sample.motion_ids, torch.ones(32, dtype=torch.long))


def test_motion_manager_uniform_time_sampling_stays_inside_valid_window() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    sampler = MotionManager(
        _package_reference_motion(),
        MotionManagerCfg(
            clip_weighting="explicit",
            clip_weights=torch.tensor([0.0, 1.0, 0.0]),
            time_sampling="uniform",
            history_seconds=0.2,
            future_seconds=0.3,
            seed=17,
        ),
    )

    sample = sampler.sample_batch(128)

    torch.testing.assert_close(sample.motion_ids, torch.ones(128, dtype=torch.long))
    assert torch.all(sample.motion_times >= 0.2).item()
    assert torch.all(sample.motion_times <= 0.7).item()
    assert not torch.allclose(sample.motion_times, torch.full((128,), 0.2))


def test_motion_manager_rejects_packages_without_valid_sampling_windows() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    with pytest.raises(ValueError, match="No clips have a valid sampling window"):
        MotionManager(
            _package_reference_motion(clip_lengths=[3]),
            MotionManagerCfg(
                clip_weighting="uniform",
                history_seconds=0.2,
                future_seconds=0.1,
            ),
        )


def test_motion_manager_sample_can_be_handed_to_motion_lib_query_seam() -> None:
    from mjlab_playground.motion_lib import (
        MotionManager,
        MotionManagerCfg,
    )

    package = _package_reference_motion()
    sampler = MotionManager(
        package,
        MotionManagerCfg(
            clip_weighting="explicit",
            clip_weights=torch.tensor([0.0, 1.0, 0.0]),
            time_sampling="start",
            seed=23,
        ),
    )

    class FakeMotionLib:
        def query(
            self,
            motion: object,
            *,
            motion_ids: torch.Tensor,
            motion_times: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            assert motion is package
            return {"motion_ids": motion_ids, "motion_times": motion_times}

    sample = sampler.sample_batch(4)
    query_result = FakeMotionLib().query(
        package,
        motion_ids=sample.motion_ids,
        motion_times=sample.motion_times,
    )

    torch.testing.assert_close(
        query_result["motion_ids"], torch.ones(4, dtype=torch.long)
    )
    torch.testing.assert_close(query_result["motion_times"], torch.zeros(4))


def test_mimic_motion_manager_resamples_only_requested_env_tracks() -> None:
    from mjlab_playground.motion_lib import (
        MimicMotionManager,
        MotionManagerCfg,
    )

    sampler = MimicMotionManager(
        _package_reference_motion(),
        num_envs=4,
        cfg=MotionManagerCfg(
            clip_weighting="explicit",
            clip_weights=torch.tensor([0.0, 1.0, 0.0]),
            time_sampling="start",
            history_seconds=0.2,
            seed=29,
        ),
    )

    torch.testing.assert_close(
        sampler.motion_ids,
        torch.full((4,), -1, dtype=torch.long),
    )
    torch.testing.assert_close(sampler.motion_times, torch.zeros(4))

    sample = sampler.sample_envs(torch.tensor([1, 3]))

    torch.testing.assert_close(sample.motion_ids, torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(sample.motion_times, torch.full((2,), 0.2))
    torch.testing.assert_close(
        sampler.motion_ids,
        torch.tensor([-1, 1, -1, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        sampler.motion_times,
        torch.tensor([0.0, 0.2, 0.0, 0.2]),
    )


@pytest.mark.parametrize("env_ids", [torch.tensor([0.9]), torch.tensor([False])])
def test_mimic_motion_manager_rejects_non_integer_env_ids(
    env_ids: torch.Tensor,
) -> None:
    from mjlab_playground.motion_lib import MimicMotionManager

    sampler = MimicMotionManager(_package_reference_motion(), num_envs=4)

    with pytest.raises(TypeError, match="env_ids must be an integer tensor"):
        sampler.sample_envs(env_ids)


def test_mimic_motion_manager_advances_active_env_tracks() -> None:
    from mjlab_playground.motion_lib import (
        MimicMotionManager,
        MotionManagerCfg,
    )

    sampler = MimicMotionManager(
        _package_reference_motion(),
        num_envs=3,
        cfg=MotionManagerCfg(
            clip_weighting="explicit",
            clip_weights=torch.tensor([0.0, 1.0, 0.0]),
            time_sampling="start",
            history_seconds=0.2,
            seed=31,
        ),
    )
    sampler.sample_envs(torch.tensor([0, 2]))

    sampler.advance_envs(0.15)

    torch.testing.assert_close(
        sampler.motion_ids,
        torch.tensor([1, -1, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        sampler.motion_times,
        torch.tensor([0.35, 0.0, 0.35]),
    )


def test_mimic_motion_manager_done_envs_respects_current_clip_windows() -> None:
    from mjlab_playground.motion_lib import (
        MimicMotionManager,
        MotionManagerCfg,
    )

    sampler = MimicMotionManager(
        _package_reference_motion(),
        num_envs=4,
        cfg=MotionManagerCfg(
            clip_weighting="uniform",
            time_sampling="start",
            future_seconds=0.1,
        ),
    )
    sampler.motion_ids[:] = torch.tensor([0, 1, 2, -1], dtype=torch.long)
    sampler.motion_times[:] = torch.tensor([0.36, 0.84, 1.36, 99.0])

    done = sampler.done_envs(lookahead=0.05)

    torch.testing.assert_close(
        done,
        torch.tensor([True, False, True, False]),
    )


def test_mimic_motion_manager_adaptive_sampling_prefers_failed_time_bin() -> None:
    from mjlab_playground.motion_lib import (
        MimicMotionManager,
        MotionManagerCfg,
    )

    sampler = MimicMotionManager(
        _package_reference_motion(),
        num_envs=1200,
        cfg=MotionManagerCfg(
            clip_weighting="explicit",
            clip_weights=torch.tensor([0.0, 1.0, 0.0]),
            time_sampling="adaptive",
            adaptive_num_bins=4,
            seed=37,
        ),
    )
    sampler.motion_ids[:4] = 1
    sampler.motion_times[:4] = torch.tensor([0.62, 0.62, 0.62, 0.62])

    sampler.report_env_outcomes(
        torch.tensor([0, 1, 2, 3]),
        failed=torch.tensor([True, True, True, True]),
    )

    torch.testing.assert_close(
        sampler.adaptive_failure_pressure[1],
        torch.tensor([0.0, 0.0, 4.0, 0.0]),
    )

    sample = sampler.sample_envs(torch.arange(1200))
    sampled_bins = torch.bucketize(
        sample.motion_times,
        torch.tensor([0.25, 0.5, 0.75]),
        right=False,
    )
    counts = torch.bincount(sampled_bins, minlength=4)

    torch.testing.assert_close(sample.motion_ids, torch.ones(1200, dtype=torch.long))
    assert torch.all(sample.motion_times >= 0.0).item()
    assert torch.all(sample.motion_times <= 1.0).item()
    assert torch.all(counts > 0).item()
    assert counts[2] > counts[0]
    assert counts[2] > counts[1]
    assert counts[2] > counts[3]


def test_mimic_motion_manager_adaptive_pressure_is_clip_specific() -> None:
    from mjlab_playground.motion_lib import (
        MimicMotionManager,
        MotionManagerCfg,
    )

    sampler = MimicMotionManager(
        _package_reference_motion(),
        num_envs=5,
        cfg=MotionManagerCfg(
            clip_weighting="uniform",
            time_sampling="adaptive",
            adaptive_num_bins=4,
        ),
    )

    torch.testing.assert_close(
        sampler.adaptive_failure_pressure,
        torch.zeros(3, 4),
    )

    sampler.motion_ids[:] = torch.tensor([0, 1, 1, 2, -1], dtype=torch.long)
    sampler.motion_times[:] = torch.tensor([0.12, 0.62, 0.62, 1.2, 0.0])

    sampler.report_env_outcomes(
        torch.tensor([0, 1, 2, 3, 4]),
        failed=torch.tensor([True, False, True, True, True]),
    )

    torch.testing.assert_close(
        sampler.adaptive_failure_pressure,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_mimic_motion_manager_adaptive_sampling_respects_valid_window() -> None:
    from mjlab_playground.motion_lib import (
        MimicMotionManager,
        MotionManagerCfg,
    )

    sampler = MimicMotionManager(
        _package_reference_motion(),
        num_envs=1000,
        cfg=MotionManagerCfg(
            clip_weighting="explicit",
            clip_weights=torch.tensor([0.0, 1.0, 0.0]),
            time_sampling="adaptive",
            history_seconds=0.2,
            future_seconds=0.3,
            adaptive_num_bins=5,
            seed=41,
        ),
    )
    sampler.motion_ids[:4] = 1
    sampler.motion_times[:4] = torch.tensor([0.69, 0.69, 0.69, 0.69])
    sampler.report_env_outcomes(
        torch.tensor([0, 1, 2, 3]),
        failed=torch.tensor([True, True, True, True]),
    )

    sample = sampler.sample_envs(torch.arange(1000))
    sampled_bins = torch.bucketize(
        sample.motion_times,
        torch.tensor([0.3, 0.4, 0.5, 0.6]),
        right=False,
    )
    counts = torch.bincount(sampled_bins, minlength=5)

    assert torch.all(sample.motion_times >= 0.2).item()
    assert torch.all(sample.motion_times <= 0.7).item()
    assert counts[4] > counts[:4].max()


def test_motion_manager_rejects_rewind_time_sampling_mode() -> None:
    from mjlab_playground.motion_lib import MotionManagerCfg

    with pytest.raises(
        ValueError,
        match="time_sampling must be 'start', 'uniform', or 'adaptive'",
    ):
        MotionManagerCfg(time_sampling="rewind")  # type: ignore[arg-type]
