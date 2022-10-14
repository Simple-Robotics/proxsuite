import platform

if (
    "i386" in platform.processor()
    or "x86_64" in platform.processor()
    or "Intel64" in platform.processor()
):
    from . import instructionset


def load_main_module(globals):
    def load_module(main_module_name):
        import importlib

        try:
            main_module = importlib.import_module("." + main_module_name, __name__)
            globals.update(main_module.__dict__)
            del globals[main_module_name]
            return True
        except ModuleNotFoundError:
            return False

    if "arm" not in platform.processor():
        all_modules = [
            ("proxsuite_pywrap_avx512", instructionset.has_AVX512F),
            ("proxsuite_pywrap_avx2", instructionset.has_AVX2),
        ]

        for module_name, checker in all_modules:
            if checker() and load_module(module_name):
                return

    assert load_module("proxsuite_pywrap") == True


load_main_module(globals=globals())
del load_main_module
del platform
