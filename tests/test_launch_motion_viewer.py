from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_MOTION_VIEWER_PATH = (
    ROOT
    / "src"
    / "mjlab_playground"
    / "motion_lib"
    / "scripts"
    / "launch_motion_viewer.py"
)
LAUNCH_MOTION_VIEWER_MODULE = "mjlab_playground.motion_lib.scripts.launch_motion_viewer"
LEGACY_MOTION_VIEWER_PATH = (
    ROOT
    / "src"
    / "mjlab_playground"
    / "motion_lib"
    / "tools"
    / "motion_viewer.py"
)


def _load_motion_viewer_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_launch_motion_viewer = _load_motion_viewer_module(
    LAUNCH_MOTION_VIEWER_PATH,
    "launch_motion_viewer",
)
parse_args = _launch_motion_viewer.parse_args


class FakeReferenceMotion:
    def __init__(self, frame_count: int, marker: float) -> None:
        self.root_pos = torch.full((frame_count, 3), marker)
        self.root_rot = torch.zeros(frame_count, 4)
        self.dof_pos = torch.zeros(frame_count, 29)


class FakeSceneAdapter:
    def __init__(self) -> None:
        self.applied: list[tuple[object, int]] = []
        self.mj_model = object()
        self.mj_data = object()

    def apply(self, motion: object, frame_index: int) -> None:
        self.applied.append((motion, frame_index))

    def sync_display_data(self) -> None:
        pass


def _write_pyroki_npz(path: Path, *, dof_count: int = 29) -> None:
    np.savez(
        path,
        base_frame_pos=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        base_frame_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        joint_angles=np.zeros((1, dof_count), dtype=np.float64),
    )


def test_motion_viewer_cli_defaults_to_pyroki_format_and_astro_30fps() -> None:
    args = parse_args(["--motion-files", "/tmp/motion.npz"])

    assert args.motion_files == Path("/tmp/motion.npz")
    assert args.motion_format == "pyroki"
    assert args.fps == 30.0
    assert args.robot == "astro"
    assert args.record_video is False
    assert args.output_dir is None


def test_motion_viewer_cli_accepts_interactive_recording_options() -> None:
    args = parse_args(
        [
            "--motion-files",
            "/tmp/motion.npz",
            "--record-video",
            "--output-dir",
            "/tmp/videos",
        ]
    )

    assert args.record_video is True
    assert args.output_dir == Path("/tmp/videos")


def test_motion_viewer_cli_accepts_headless_batch_recording_options() -> None:
    args = parse_args(
        [
            "--motion-files",
            "/tmp/motions",
            "--headless",
            "--record-video",
        ]
    )

    assert args.headless is True
    assert args.record_video is True


def test_motion_viewer_cli_rejects_video_fps_override() -> None:
    try:
        parse_args(
            [
                "--motion-files",
                "/tmp/motions",
                "--headless",
                "--record-video",
                "--video-fps",
                "60",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected --video-fps override to be rejected")


def test_motion_viewer_main_rejects_headless_without_recording(tmp_path: Path) -> None:
    motion_viewer_main = _launch_motion_viewer.main

    try:
        motion_viewer_main(["--motion-files", str(tmp_path / "motion.npz"), "--headless"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected headless mode without recording to fail")


def test_offscreen_frame_renderer_uses_mjlab_renderer_and_viewer_config() -> None:
    calls = []

    class FakeViewerConfig:
        class OriginType:
            ASSET_ROOT = "asset-root"

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeOffscreenRenderer:
        def __init__(self, *, model, cfg, scene, sim_model=None) -> None:
            calls.append(("init", model, cfg, scene, sim_model))

        def initialize(self) -> None:
            calls.append(("initialize",))

        def update(self, data) -> None:
            calls.append(("update", data))

        def render(self) -> str:
            calls.append(("render",))
            return "rendered-frame"

        def close(self) -> None:
            calls.append(("close",))

    scene_adapter = SimpleNamespace(
        mj_model="mj-model",
        scene="scene",
        sim=SimpleNamespace(model="sim-model", data="sim-data"),
    )

    renderer = _launch_motion_viewer.OffscreenFrameRenderer(
        scene_adapter,
        offscreen_renderer_cls=FakeOffscreenRenderer,
        viewer_config_cls=FakeViewerConfig,
    )
    frame = renderer.render_frame()
    renderer.close()

    cfg = calls[0][2]
    assert cfg.kwargs == {
        "height": 480,
        "width": 640,
        "origin_type": "asset-root",
        "entity_name": "robot",
        "distance": _launch_motion_viewer.DEFAULT_CAMERA_CONFIG.distance,
        "elevation": _launch_motion_viewer.DEFAULT_CAMERA_CONFIG.elevation,
        "azimuth": _launch_motion_viewer.DEFAULT_CAMERA_CONFIG.azimuth,
    }
    assert calls == [
        ("init", "mj-model", cfg, "scene", "sim-model"),
        ("initialize",),
        ("update", "sim-data"),
        ("render",),
        ("close",),
    ]
    assert frame == "rendered-frame"


def test_create_motion_viewer_loads_motions_and_builds_selected_robot_adapter(
    tmp_path: Path,
) -> None:
    motions = [FakeReferenceMotion(frame_count=1, marker=1.0)]
    scene_adapter = FakeSceneAdapter()
    calls = {}

    def load_motions(*args, **kwargs):
        calls["load_motions"] = (args, kwargs)
        return motions

    def build_adapter(*args, **kwargs):
        calls["scene_adapter_builder"] = (args, kwargs)
        return scene_adapter

    viewer = _launch_motion_viewer.create_motion_viewer(
        tmp_path / "motion.npz",
        motion_format="pyroki",
        fps=50.0,
        robot="astro",
        device="cpu",
        load_motions=load_motions,
        scene_adapter_builder=build_adapter,
    )

    assert viewer.motions == motions
    assert viewer.scene_adapter is scene_adapter
    assert calls["load_motions"] == (
        (tmp_path / "motion.npz",),
        {"motion_format": "pyroki", "fps": 50.0, "device": "cpu"},
    )
    assert calls["scene_adapter_builder"] == (
        ("astro",),
        {"device": "cpu"},
    )


def test_motion_viewer_main_launches_interactive_viewer(tmp_path: Path) -> None:
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    viewer = _launch_motion_viewer.MotionViewer([motion], FakeSceneAdapter())
    calls = {}

    def create_viewer(*args, **kwargs):
        calls["create_viewer"] = (args, kwargs)
        return viewer

    _launch_motion_viewer.main(
        ["--motion-files", str(tmp_path / "motion.npz"), "--fps", "50"],
        create_viewer=create_viewer,
        run_viewer=lambda built_viewer: calls.setdefault("run_viewer", built_viewer),
    )

    assert calls["create_viewer"] == (
        (tmp_path / "motion.npz",),
        {
            "motion_format": "pyroki",
            "fps": 50.0,
            "robot": "astro",
            "device": "cpu",
        },
    )
    assert calls["run_viewer"] is viewer


def test_motion_viewer_main_reports_terminal_status_by_default(
    tmp_path: Path,
    capsys,
) -> None:
    class FakeInteractiveViewer:
        def run_interactive(self, *, status_reporter, **kwargs):
            status_reporter(
                "Motion 1/1 | walking | [--------------------] "
                "0/3 (0.0%) | speed 1x | playing"
            )

    _launch_motion_viewer.main(
        ["--motion-files", str(tmp_path / "motion.npz")],
        create_viewer=lambda *_, **__: FakeInteractiveViewer(),
    )

    assert (
        "\rMotion 1/1 | walking | [--------------------] "
        "0/3 (0.0%) | speed 1x | playing"
        in capsys.readouterr().out
    )


def test_motion_viewer_main_wires_interactive_root_tracking_camera(
    tmp_path: Path,
) -> None:
    calls = {}
    configured: list[tuple[object, object]] = []

    class FakeTrackingSceneAdapter:
        def configure_tracking_camera(self, viewer_handle, camera_config) -> None:
            configured.append((viewer_handle, camera_config))

    class FakeInteractiveViewer:
        scene_adapter = FakeTrackingSceneAdapter()

        def run_interactive(self, **kwargs):
            calls["run_interactive"] = kwargs

    _launch_motion_viewer.main(
        ["--motion-files", str(tmp_path / "motion.npz")],
        create_viewer=lambda *_, **__: FakeInteractiveViewer(),
    )

    run_kwargs = calls["run_interactive"]
    assert run_kwargs["status_reporter"] is not None
    assert run_kwargs["viewer_handle_configurator"] is not None

    handle = object()
    run_kwargs["viewer_handle_configurator"](handle)

    assert configured == [(handle, _launch_motion_viewer.DEFAULT_CAMERA_CONFIG)]


def test_viewer_handle_camera_status_reports_live_adjustable_camera_values() -> None:
    handle = SimpleNamespace(
        cam=SimpleNamespace(
            distance=2.3456,
            elevation=-5.6789,
            azimuth=20.1234,
        )
    )

    assert (
        _launch_motion_viewer._viewer_handle_camera_status(handle)
        == "camera distance=2.35 elevation=-5.68 azimuth=20.12"
    )


def test_motion_viewer_main_wires_interactive_camera_status_reporting(
    tmp_path: Path,
) -> None:
    calls = {}

    class FakeInteractiveViewer:
        scene_adapter = FakeSceneAdapter()

        def run_interactive(self, **kwargs):
            calls["run_interactive"] = kwargs

    _launch_motion_viewer.main(
        ["--motion-files", str(tmp_path / "motion.npz")],
        create_viewer=lambda *_, **__: FakeInteractiveViewer(),
    )

    run_kwargs = calls["run_interactive"]
    assert run_kwargs["viewer_handle_status_provider"] is not None
    handle = SimpleNamespace(
        cam=SimpleNamespace(distance=2.0, elevation=-5.0, azimuth=20.0)
    )
    assert (
        run_kwargs["viewer_handle_status_provider"](handle)
        == "camera distance=2 elevation=-5 azimuth=20"
    )


def test_motion_viewer_main_wires_interactive_recording_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motion_path = tmp_path / "walk.npz"
    output_dir = tmp_path / "custom-videos"
    motion_path.write_bytes(b"motion")
    calls = {}
    subprocess_calls = []

    def fake_run(command, check):
        subprocess_calls.append((command, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_launch_motion_viewer.subprocess, "run", fake_run)

    class FakeInteractiveViewer:
        motions = [SimpleNamespace(name="walk.npz")]

        def run_interactive(self, **kwargs):
            calls["run_interactive"] = kwargs

    _launch_motion_viewer.main(
        [
            "--motion-files",
            str(motion_path),
            "--record-video",
            "--output-dir",
            str(output_dir),
        ],
        create_viewer=lambda *_, **__: FakeInteractiveViewer(),
    )

    run_kwargs = calls["run_interactive"]
    assert run_kwargs["recording_output_paths"] == [output_dir / "walk.mp4"]
    assert run_kwargs["recording_executor"] is not None
    assert run_kwargs["status_reporter"] is not None

    run_kwargs["recording_executor"](0, output_dir / "walk.mp4")

    assert subprocess_calls == [
        (
            [
                sys.executable,
                "-m",
                "mjlab_playground.motion_lib.scripts.launch_motion_viewer",
                "--motion-files",
                str(motion_path),
                "--format",
                "pyroki",
                "--fps",
                "30.0",
                "--robot",
                "astro",
                "--device",
                "cpu",
                "--headless",
                "--record-video",
                "--output-dir",
                str(output_dir),
            ],
            False,
        )
    ]


def test_motion_viewer_main_records_headless_batch_and_does_not_run_interactive(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    (motion_dir / "jump.npz").write_bytes(b"motion")
    (motion_dir / "walk.npz").write_bytes(b"motion")
    (motion_dir / "README.txt").write_text("sidecar")
    calls = {}

    class FakeHeadlessViewer:
        motions = [
            SimpleNamespace(name="jump.npz"),
            SimpleNamespace(name="walk.npz"),
        ]

        def record_headless(self, **kwargs):
            calls["record_headless"] = kwargs
            return []

        def run_interactive(self, **kwargs):
            raise AssertionError("headless recording must not launch interactive playback")

    _launch_motion_viewer.main(
        [
            "--motion-files",
            str(motion_dir),
            "--headless",
            "--record-video",
        ],
        create_viewer=lambda *_, **__: FakeHeadlessViewer(),
    )

    record_kwargs = calls["record_headless"]
    output_paths = record_kwargs["recording_output_paths"]
    output_dir = output_paths[0].parent
    assert record_kwargs["recording_output_paths"] == [
        output_dir / "jump.mp4",
        output_dir / "walk.mp4",
    ]
    assert output_dir.parent == tmp_path / "renderings"
    assert record_kwargs["recording_attachment_factory"] is not None
    assert callable(record_kwargs["frame_renderer_factory"])
    assert record_kwargs["status_reporter"] is not None


def test_motion_viewer_main_smoke_test_verifies_one_frame_without_interactive_viewer(
    tmp_path: Path,
) -> None:
    calls = {}

    def verify_viewer_path(*args, **kwargs):
        calls["verify_viewer_path"] = (args, kwargs)

    def create_viewer(*args, **kwargs):
        raise AssertionError("smoke test must not create an interactive viewer")

    _launch_motion_viewer.main(
        [
            "--motion-files",
            str(tmp_path / "motion.npz"),
            "--fps",
            "50",
            "--smoke-test",
        ],
        create_viewer=create_viewer,
        run_viewer=lambda _: calls.setdefault("run_viewer", True),
        verify_viewer_path=verify_viewer_path,
    )

    assert calls == {
        "verify_viewer_path": (
            (tmp_path / "motion.npz",),
            {
                "motion_format": "pyroki",
                "fps": 50.0,
                "robot": "astro",
                "device": "cpu",
            },
        )
    }


def test_verify_motion_viewer_path_loads_pyroki_builds_astro_and_applies_one_frame(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)
    scene_adapter = FakeSceneAdapter()
    calls = {}

    def build_adapter(*args, **kwargs):
        calls["build_adapter"] = (args, kwargs)
        return scene_adapter

    viewer = _launch_motion_viewer.verify_motion_viewer_path(
        motion_path,
        motion_format="pyroki",
        fps=30.0,
        robot="astro",
        device="cpu",
        scene_adapter_builder=build_adapter,
    )

    assert calls["build_adapter"] == (("astro",), {"device": "cpu"})
    assert len(viewer.motions) == 1
    assert scene_adapter.applied == [(viewer.motions[0], 0)]


def test_verify_motion_viewer_path_labels_bad_pyroki_source_data(tmp_path: Path) -> None:
    motion_path = tmp_path / "bad.npz"
    np.savez(motion_path, base_frame_pos=np.zeros((1, 3)))

    with pytest.raises(
        _launch_motion_viewer.MotionViewerVerificationError,
        match="Failed to load pyroki motion source data",
    ):
        _launch_motion_viewer.verify_motion_viewer_path(
            motion_path,
            scene_adapter_builder=lambda *_, **__: FakeSceneAdapter(),
        )


def test_verify_motion_viewer_path_labels_unsupported_format_selection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        _launch_motion_viewer.MotionViewerVerificationError,
        match="Unsupported motion format 'proto'",
    ):
        _launch_motion_viewer.verify_motion_viewer_path(
            tmp_path / "motion.motion",
            motion_format="proto",
            load_motions=lambda *_, **__: (_ for _ in ()).throw(
                NotImplementedError("proto motion loading is not implemented yet")
            ),
            scene_adapter_builder=lambda *_, **__: FakeSceneAdapter(),
        )


def test_verify_motion_viewer_path_labels_astro_dof_mismatch(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path, dof_count=2)

    def build_adapter(*_, **__):
        class FailingAstroAdapter(FakeSceneAdapter):
            def apply(self, motion: FakeReferenceMotion, frame_index: int) -> None:
                raise ValueError("Astro reference motion DOF count must be 3, got 2")

        return FailingAstroAdapter()

    with pytest.raises(
        _launch_motion_viewer.MotionViewerVerificationError,
        match="Incompatible Astro reference motion DOF count",
    ):
        _launch_motion_viewer.verify_motion_viewer_path(
            motion_path,
            scene_adapter_builder=build_adapter,
        )


def test_verify_motion_viewer_path_labels_scene_construction_errors(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_pyroki_npz(motion_path)

    with pytest.raises(
        _launch_motion_viewer.MotionViewerVerificationError,
        match="Failed to construct astro scene adapter",
    ):
        _launch_motion_viewer.verify_motion_viewer_path(
            motion_path,
            scene_adapter_builder=lambda *_, **__: (_ for _ in ()).throw(
                RuntimeError("scene setup failed")
            ),
        )


def test_launch_motion_viewer_runner_help_avoids_task_registration_imports() -> None:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            LAUNCH_MOTION_VIEWER_MODULE,
            "--help",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    help_text = " ".join(result.stdout.split())
    assert "--motion-files" in result.stdout
    assert "--record-video" in result.stdout
    assert "then resumes the same viewer" in help_text
    assert "renderings/<timestamp>" in result.stdout
    assert "--video-fps" not in result.stdout
    assert "--camera-distance" not in result.stdout
    assert "--camera-elevation" not in result.stdout
    assert "--camera-azimuth" not in result.stdout
    assert "--smoke-test" in result.stdout
    assert "rsl_rl" not in result.stderr


def test_legacy_motion_viewer_script_is_retired() -> None:
    assert not LEGACY_MOTION_VIEWER_PATH.exists()


def test_motion_viewer_cli_accepts_proto_format() -> None:
    args = parse_args(
        [
            "--motion-files",
            "/tmp/motion.motion",
            "--format",
            "proto",
            "--fps",
            "50",
        ]
    )

    assert args.motion_format == "proto"
    assert args.fps == 50.0
    assert args.robot == "astro"


def test_motion_viewer_cli_rejects_unsupported_robot() -> None:
    try:
        parse_args(["--motion-files", "/tmp/motion.npz", "--robot", "g1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected unsupported robot to fail during parsing")
