try:
    import torch  # noqa F401
except ImportError:
    import warnings

    warnings.warn(
        "PyTorch not found. Please install via pip or conda for example to use QPFunction."
    )
    del warnings
