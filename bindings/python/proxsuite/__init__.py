from . import instructionset as instructionset

if instructionset.has_AVX512F():
    from .proxsuite_pywrap_avx512 import *

    main_module = proxsuite_pywrap_avx512
    del proxsuite_pywrap_avx512
elif instructionset.has_AVX2():
    from .proxsuite_pywrap_avx2 import *

    main_module = proxsuite_pywrap_avx2
    del proxsuite_pywrap_avx2
else:
    from .proxsuite_pywrap import *

    main_module = proxsuite_pywrap
    del proxsuite_pywrap

__verion__ = main_module.__version__

del main_module
