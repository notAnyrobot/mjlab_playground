from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab_playground.motion_lib.motion_viewer import (
    PlaybackAction,
    RecordingStatus,
    TerminalStatusReporter,
    ViewerTick,
)
from mjlab_playground.motion_lib.viewers import resolve_viewer_mode

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
    ROOT / "src" / "mjlab_playground" / "motion_lib" / "tools" / "motion_viewer.py"
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

    def apply_reference_frame(self, motion: object, frame_index: int) -> None:
        self.applied.append((motion, frame_index))

    def sync_display_data(self) -> None:
        pass


def test_terminal_status_reporter_keeps_status_on_one_terminal_line() -> None:
    stream = io.StringIO()
    reporter = TerminalStatusReporter(stream=stream, max_width=80)
    long_status = (
        "Motion 1/32 | a_very_long_reference_motion_name | "
        "[===-----------------] 313/2526 (12.4%) | speed 1x | playing | "
        "camera distance=2 elevation=-19.44 azimuth=11.94"
    )

    reporter(long_status)

    rendered = stream.getvalue()
    assert rendered.startswith("\rMotion 1/32 | ")
    assert rendered.endswith("distance=2 elevation=-19.44 azimuth=11.94")
    assert "\n" not in rendered
    assert len(rendered.removeprefix("\r")) == 80


def _write_pyroki_npz(path: Path, *, dof_count: int = 29) -> None:
    np.savez(
        path,
        base_frame_pos=np.array(
            [[1.0, 2.0, 3.0], [1.1, 2.0, 3.0], [1.2, 2.0, 3.0]],
            dtype=np.float64,
        ),
        base_frame_wxyz=np.array(
            [[1.0, 0.0, 0.0, 0.0]] * 3,
            dtype=np.float64,
        ),
        joint_angles=np.zeros((3, dof_count), dtype=np.float64),
    )


def test_motion_viewer_cli_defaults_to_pyroki_format_and_astro_30fps() -> None:
    args = parse_args(["--motion-files", "/tmp/motion.npz"])

    assert args.motion_files == Path("/tmp/motion.npz")
    assert args.motion_format == "pyroki"
    assert args.fps == 30.0
    assert args.robot == "astro"
    assert args.record_video is False
    assert args.output_dir is None
    assert args.viewer == "auto"


def test_auto_viewer_mode_selects_viser_on_macos() -> None:
    assert (
        resolve_viewer_mode("auto", platform_name="darwin", display_available=True)
        == "viser"
    )


def test_auto_viewer_mode_selects_native_on_linux_with_display() -> None:
    assert (
        resolve_viewer_mode("auto", platform_name="linux", display_available=True)
        == "native"
    )


def test_auto_viewer_mode_selects_viser_on_linux_without_display() -> None:
    assert (
        resolve_viewer_mode("auto", platform_name="linux", display_available=False)
        == "viser"
    )


@pytest.mark.parametrize("mode", ["native", "viser"])
def test_explicit_viewer_mode_is_honored(mode: str) -> None:
    assert (
        resolve_viewer_mode(
            mode,
            platform_name="unsupported",
            display_available=False,
        )
        == mode
    )


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


def test_motion_viewer_cli_rejects_output_directory_without_recording() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(
            [
                "--motion-files",
                "/tmp/motion.npz",
                "--output-dir",
                "/tmp/videos",
            ]
        )

    assert exc_info.value.code == 2


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


def test_motion_viewer_cli_rejects_smoke_test_with_recording() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(
            [
                "--motion-files",
                "/tmp/motion.npz",
                "--smoke-test",
                "--record-video",
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize("viewer_mode", ["native", "viser"])
@pytest.mark.parametrize(
    "presentation_free_args",
    [["--headless", "--record-video"], ["--smoke-test"]],
)
def test_motion_viewer_cli_rejects_explicit_viewer_without_interactive_presentation(
    viewer_mode: str,
    presentation_free_args: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(
            [
                "--motion-files",
                "/tmp/motion.npz",
                "--viewer",
                viewer_mode,
                *presentation_free_args,
            ]
        )

    assert exc_info.value.code == 2


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
        motion_viewer_main(
            ["--motion-files", str(tmp_path / "motion.npz"), "--headless"],
            load_motions=lambda *_, **__: (_ for _ in ()).throw(
                AssertionError("invalid options must fail before viewer construction")
            ),
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected headless mode without recording to fail")


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--output-dir", "/tmp/videos"],
        ["--smoke-test", "--record-video"],
        ["--headless", "--record-video", "--viewer", "native"],
        ["--smoke-test", "--viewer", "viser"],
    ],
)
def test_motion_viewer_main_rejects_invalid_modes_before_heavy_construction(
    tmp_path: Path,
    invalid_args: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _launch_motion_viewer.main(
            ["--motion-files", str(tmp_path / "motion.npz"), *invalid_args],
            load_motions=lambda *_, **__: (_ for _ in ()).throw(
                AssertionError("invalid options must fail before viewer construction")
            ),
            verify_viewer_path=lambda *_, **__: (_ for _ in ()).throw(
                AssertionError("invalid options must fail before smoke construction")
            ),
        )

    assert exc_info.value.code == 2


def test_build_scene_adapter_resolves_astro_config_for_mujoco_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    class FakeMujocoSceneAdapter:
        def __init__(self, cfg) -> None:
            calls["adapter_init"] = {
                "robot_cfg": cfg.robot_cfg,
                "output_fps": cfg.output_fps,
                "device": cfg.device,
            }

    monkeypatch.setattr(
        _launch_motion_viewer,
        "_load_astro_constants_module",
        lambda: SimpleNamespace(get_astro_robot_cfg=lambda: "astro-robot-cfg"),
    )
    monkeypatch.setattr(
        _launch_motion_viewer,
        "MujocoSceneAdapter",
        FakeMujocoSceneAdapter,
    )

    adapter = _launch_motion_viewer.build_scene_adapter(
        "astro",
        output_fps=50.0,
        device="cuda:0",
    )

    assert isinstance(adapter, FakeMujocoSceneAdapter)
    assert calls["adapter_init"] == {
        "robot_cfg": "astro-robot-cfg",
        "output_fps": 50.0,
        "device": "cuda:0",
    }


def test_motion_viewer_main_routes_native_playback_through_adapter_factory(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    scene = FakeSceneAdapter()
    calls = []

    class FakeNativeAdapter:
        def run(self, update) -> None:
            calls.append(("run", update))

        def close(self) -> None:
            calls.append(("close",))

    def viewer_adapter_factory(mode, *, scene, **kwargs):
        calls.append(("factory", mode, scene, kwargs))
        return FakeNativeAdapter()

    _launch_motion_viewer.main(
        [
            "--motion-files",
            str(tmp_path / "motion.npz"),
            "--viewer",
            "native",
        ],
        load_motions=lambda *_, **__: [motion],
        scene_adapter_builder=lambda *_, **__: scene,
        viewer_adapter_factory=viewer_adapter_factory,
    )

    assert calls[0][0:3] == ("factory", "native", scene)
    assert calls[0][3]["status_reporter"] is not None
    assert calls[0][3]["recording_enabled"] is False
    assert calls[1][0] == "run"
    assert calls[2] == ("close",)


def test_motion_viewer_main_resolves_and_reports_auto_mode_before_construction(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    scene = FakeSceneAdapter()
    calls = []

    class FakeViserAdapter:
        def run(self, update) -> None:
            calls.append(("run", update))

        def close(self) -> None:
            calls.append(("close",))

    def load_motions(*args, **kwargs):
        calls.append(("load_motions", args, kwargs))
        return [motion]

    def viewer_adapter_factory(mode, *, scene, **kwargs):
        calls.append(("factory", mode, scene, kwargs))
        return FakeViserAdapter()

    _launch_motion_viewer.main(
        ["--motion-files", str(tmp_path / "motion.npz")],
        load_motions=load_motions,
        scene_adapter_builder=lambda *_, **__: scene,
        viewer_adapter_factory=viewer_adapter_factory,
        platform_name="linux",
        display_available=False,
        report_viewer_selection=lambda mode: calls.append(("report", mode)),
    )

    assert calls[0] == ("report", "viser")
    assert calls[1][0] == "load_motions"
    assert calls[2][0:3] == ("factory", "viser", scene)
    assert calls[3][0] == "run"
    assert calls[4] == ("close",)


def test_motion_viewer_main_does_not_fallback_after_explicit_adapter_failure(
    tmp_path: Path,
) -> None:
    motion = FakeReferenceMotion(frame_count=1, marker=1.0)
    scene = FakeSceneAdapter()
    requested_modes = []

    def failing_factory(mode, *, scene, **kwargs):
        requested_modes.append(mode)
        raise RuntimeError("viser unavailable")

    with pytest.raises(RuntimeError, match="viser unavailable"):
        _launch_motion_viewer.main(
            [
                "--motion-files",
                str(tmp_path / "motion.npz"),
                "--viewer",
                "viser",
            ],
            load_motions=lambda *_, **__: [motion],
            scene_adapter_builder=lambda *_, **__: scene,
            viewer_adapter_factory=failing_factory,
        )

    assert requested_modes == ["viser"]


def test_motion_viewer_main_injects_background_recorder_into_core(
    tmp_path: Path,
) -> None:
    motion_path = tmp_path / "walk.npz"
    output_dir = tmp_path / "custom-videos"
    motion_path.write_bytes(b"motion")
    motion = SimpleNamespace(name="walk.npz", root_pos=np.zeros((2, 3)), fps=30.0)
    scene = FakeSceneAdapter()
    calls: dict[str, object] = {}

    class FakeBackgroundRecorder:
        status = RecordingStatus.IDLE
        output_path = None
        error = None

        def poll(self) -> None:
            pass

        def wait(self) -> None:
            self.status = RecordingStatus.SUCCEEDED

        def cancel(self) -> None:
            self.status = RecordingStatus.FAILED

        def close(self) -> None:
            pass

        def record_selected_clip(self, motion_index, selected_motion) -> None:
            calls["record"] = (motion_index, selected_motion)

    recorder = FakeBackgroundRecorder()

    def make_background_recorder(**kwargs):
        calls["background_factory"] = kwargs
        return recorder

    class FakeAdapter:
        def run(self, update) -> None:
            update(ViewerTick(0.0, (PlaybackAction.RECORD_SELECTED_CLIP,)))
            update(ViewerTick(0.0, (PlaybackAction.STOP,)))

        def close(self) -> None:
            pass

    def make_viewer_adapter(*args, **kwargs):
        calls["adapter_factory"] = (args, kwargs)
        return FakeAdapter()

    _launch_motion_viewer.main(
        [
            "--motion-files",
            str(motion_path),
            "--format",
            "mjlab",
            "--fps",
            "50",
            "--record-video",
            "--output-dir",
            str(output_dir),
        ],
        load_motions=lambda *_, **__: [motion],
        scene_adapter_builder=lambda *_, **__: scene,
        viewer_adapter_factory=make_viewer_adapter,
        background_recorder_factory=make_background_recorder,
    )

    factory_kwargs = calls["background_factory"]
    assert isinstance(factory_kwargs, dict)
    assert factory_kwargs["targets"][0].source_path == motion_path
    assert factory_kwargs["targets"][0].output_path == output_dir / "walk.mp4"
    assert factory_kwargs["motion_format"] == "mjlab"
    assert factory_kwargs["fps"] == 50.0
    assert factory_kwargs["robot"] == "astro"
    assert factory_kwargs["device"] == "cpu"
    _, adapter_kwargs = calls["adapter_factory"]
    assert adapter_kwargs["recording_enabled"] is True
    assert calls["record"] == (0, motion)


def test_motion_viewer_main_records_headless_batch_without_a_viewer_adapter(
    tmp_path: Path,
) -> None:
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    (motion_dir / "jump.npz").write_bytes(b"motion")
    (motion_dir / "walk.npz").write_bytes(b"motion")
    (motion_dir / "README.txt").write_text("sidecar")
    calls = {}
    scene = FakeSceneAdapter()

    motions = [
        SimpleNamespace(name="jump.npz"),
        SimpleNamespace(name="walk.npz"),
    ]

    class FakeDeterministicRecorder:
        def record(self, motions, request):
            calls["record"] = (motions, request)
            return ()

    def make_recorder(built_scene, **kwargs):
        calls["recorder_factory"] = (built_scene, kwargs)
        return FakeDeterministicRecorder()

    _launch_motion_viewer.main(
        [
            "--motion-files",
            str(motion_dir),
            "--headless",
            "--record-video",
        ],
        load_motions=lambda *_, **__: motions,
        scene_adapter_builder=lambda *_, **__: scene,
        viewer_adapter_factory=lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("headless mode must not construct a viewer adapter")
        ),
        deterministic_recorder_factory=make_recorder,
    )

    recorded_motions, request = calls["record"]
    output_dir = request.targets[0].output_path.parent
    assert recorded_motions == motions
    assert [target.output_path for target in request.targets] == [
        output_dir / "jump.mp4",
        output_dir / "walk.mp4",
    ]
    assert output_dir.parent == tmp_path / "renderings"
    built_scene, recorder_kwargs = calls["recorder_factory"]
    assert built_scene is scene
    assert recorder_kwargs["status_reporter"] is not None


def test_motion_viewer_main_smoke_test_verifies_one_frame_without_interactive_viewer(
    tmp_path: Path,
) -> None:
    calls = {}

    def verify_viewer_path(*args, **kwargs):
        calls["verify_viewer_path"] = (args, kwargs)

    _launch_motion_viewer.main(
        [
            "--motion-files",
            str(tmp_path / "motion.npz"),
            "--fps",
            "50",
            "--smoke-test",
        ],
        load_motions=lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("smoke test must not load through interactive path")
        ),
        verify_viewer_path=verify_viewer_path,
        viewer_adapter_factory=lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("smoke test must not construct a viewer adapter")
        ),
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

    result = _launch_motion_viewer.verify_motion_viewer_path(
        motion_path,
        motion_format="pyroki",
        fps=30.0,
        robot="astro",
        device="cpu",
        scene_adapter_builder=build_adapter,
    )

    assert calls["build_adapter"] == (
        ("astro",),
        {"output_fps": 30.0, "device": "cpu"},
    )
    assert result is None
    assert len(scene_adapter.applied) == 1
    assert scene_adapter.applied[0][1] == 0


def test_verify_motion_viewer_path_labels_bad_pyroki_source_data(
    tmp_path: Path,
) -> None:
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
            def apply_reference_frame(
                self,
                motion: FakeReferenceMotion,
                frame_index: int,
            ) -> None:
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
    assert "background child process while interactive playback continues" in help_text
    assert "renderings/<timestamp>" in result.stdout
    assert "--video-fps" not in result.stdout
    assert "--camera-distance" not in result.stdout
    assert "--camera-elevation" not in result.stdout
    assert "--camera-azimuth" not in result.stdout
    assert "--smoke-test" in result.stdout
    assert "--viewer" in result.stdout
    assert "rsl_rl" not in result.stderr


def test_launch_motion_viewer_import_does_not_import_legacy_astro_adapter() -> None:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = (
        "import sys; "
        f"import {LAUNCH_MOTION_VIEWER_MODULE}; "
        "forbidden = ["
        "'mjlab_playground.motion_lib.astro_scene_adapter'"
        "]; "
        "print([name for name in forbidden if name in sys.modules])"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


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


def test_motion_viewer_cli_accepts_mjlab_train_ready_format() -> None:
    args = parse_args(
        [
            "--motion-files",
            "/tmp/mjlab-astro/walk.npz",
            "--format",
            "mjlab",
            "--fps",
            "50",
        ]
    )

    assert args.motion_format == "mjlab"
    assert args.fps == 50.0
    assert args.robot == "astro"


def test_motion_viewer_cli_rejects_unsupported_robot() -> None:
    try:
        parse_args(["--motion-files", "/tmp/motion.npz", "--robot", "g1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected unsupported robot to fail during parsing")
