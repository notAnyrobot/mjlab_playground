__all__ = [
    "MotionLib",
    "MotionLibCfg",
    "MotionFormat",
    "MotionLoader",
    "MotionResamplingCfg",
    "PyrokiMotionLoader",
    "ReferenceMotionNpzLoader",
    "ReferenceMotionNpzWriter",
    "ReferenceMotionResampler",
    "ReferenceMotionState",
]


def __getattr__(name: str):
    if name in {"MotionLib", "MotionLibCfg"}:
        from mjlab_playground.motion_lib.motion_lib import MotionLib, MotionLibCfg

        exports = {
            "MotionLib": MotionLib,
            "MotionLibCfg": MotionLibCfg,
        }
    elif name in {
        "MotionFormat",
        "MotionLoader",
        "PyrokiMotionLoader",
        "ReferenceMotionNpzLoader",
        "ReferenceMotionState",
    }:
        from mjlab_playground.motion_lib.motion_loader import (
            MotionFormat,
            MotionLoader,
            PyrokiMotionLoader,
            ReferenceMotionNpzLoader,
            ReferenceMotionState,
        )

        exports = {
            "MotionFormat": MotionFormat,
            "MotionLoader": MotionLoader,
            "PyrokiMotionLoader": PyrokiMotionLoader,
            "ReferenceMotionNpzLoader": ReferenceMotionNpzLoader,
            "ReferenceMotionState": ReferenceMotionState,
        }
    elif name in {"MotionResamplingCfg", "ReferenceMotionResampler"}:
        from mjlab_playground.motion_lib.motion_resampler import (
            MotionResamplingCfg,
            ReferenceMotionResampler,
        )

        exports = {
            "MotionResamplingCfg": MotionResamplingCfg,
            "ReferenceMotionResampler": ReferenceMotionResampler,
        }
    elif name == "ReferenceMotionNpzWriter":
        from mjlab_playground.motion_lib.reference_motion_npz_writer import (
            ReferenceMotionNpzWriter,
        )

        exports = {
            "ReferenceMotionNpzWriter": ReferenceMotionNpzWriter,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = exports[name]
    globals()[name] = value
    return value
