"""Compatibility shims for running ORTrack with modern PyTorch."""

import sys
import types


def ensure_torch_six():
    if "torch._six" in sys.modules:
        return
    module = types.ModuleType("torch._six")
    module.string_classes = (str, bytes)
    module.int_classes = (int,)
    sys.modules["torch._six"] = module
