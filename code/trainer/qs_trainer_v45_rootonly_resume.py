"""Legacy trainer wrapper for root-only qscale0375 band1of8 continuation runs.

This module intentionally loads the archived trainer snapshot that matches the
original `Qwen2.5-Coder-3B-QGSG-SFTinit-RL-rootonly-qscale0375-band1of8-`
`strictpool-nofreeze-v1` experiment, so a resume run can preserve the old
selection/reward semantics without changing the current mainline
`qsv.qs_trainer`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LEGACY_FILENAME = (
    "qs_trainer-Qwen2.5-Coder-3B-QGSG-SFTinit-RL-rootonly-"
    "qscale0375-band1of8-strictpool-nofreeze-v1.py"
)
_LEGACY_PATH = Path(__file__).with_name(_LEGACY_FILENAME)

if not _LEGACY_PATH.exists():
    raise FileNotFoundError(f"Legacy trainer snapshot not found: {_LEGACY_PATH}")

_SPEC = importlib.util.spec_from_file_location(
    "qsv._legacy_rootonly_qscale0375_band1of8_v1",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Failed to load spec for legacy trainer: {_LEGACY_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

QuestionSolverTrainer = _MODULE.QuestionSolverTrainer

