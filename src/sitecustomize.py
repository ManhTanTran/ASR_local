"""Runtime compatibility patches for hosted notebook environments."""

try:
    import numpy as _np
except Exception:
    _np = None

if _np is not None and not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid

