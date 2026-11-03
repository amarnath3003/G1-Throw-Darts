"""Avoid an unused, faulty optional native dependency in this Windows setup.

The shared Python installation's pyarrow/arrow.dll raises 0xc0000005 during
Pandas import/shutdown (Windows Application Error records on 2026-10-30).
This experiment uses NumPy archives and CSV/JSON, never Arrow or Parquet.
Mark Arrow unavailable in this process before Pandas/SB3 imports; Pandas has
an ordinary optional-dependency fallback. No global packages are modified.
"""
import sys

if sys.platform == 'win32' and 'pyarrow' not in sys.modules:
    sys.modules['pyarrow'] = None
