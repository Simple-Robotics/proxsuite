"""
Copyright (c) 2022-2025, INRIA
"""

import numpy  # noqa F401 for OpenMP proper linkage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .proxsuite_pywrap import *  # noqa F403


def _load_main_module():
    import platform
    import importlib

    machine = platform.machine()
    has_vectorization_instructions = not machine.startswith(
        ("arm", "aarch64", "power", "ppc64", "s390x", "sparc")
    )

    def load_module(main_module_name):
        try:
            return importlib.import_module("." + main_module_name, __name__)
        except ModuleNotFoundError:
            return False

    if has_vectorization_instructions:  # noqa
        from . import instructionset

        all_modules = [
            ("proxsuite_pywrap_avx512", instructionset.has_AVX512F),
            ("proxsuite_pywrap_avx2", instructionset.has_AVX2),
        ]

        for module_name, checker in all_modules:
            if checker() and (mod := load_module(module_name)):
                return mod

    return load_module("proxsuite_pywrap")


_submodule = _load_main_module()


def __getattr__(name: str):
    # reroute proxsuite module's attributes to be that of the loaded submodule.
    return getattr(_submodule, name)


def __dir__():
    # returns iterable of all accessible attributes in the module.
    # implement this for instropection, for e.g. autocomplete in interpreters (IPython).
    # We return the attributes from the submodule.
    return dir(_submodule)
