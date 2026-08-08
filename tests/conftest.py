from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_NAME = "aiwonderland"
_SENTINELS = ("pyproject.toml", "setup.py", "setup.cfg")  # TODO: Add 1 in 3 files


def _find_project_root(start, package_name):
    for candidate in (start, *start.parents):
        if (candidate / package_name).is_dir() or any(
            (candidate / sentinel).is_file() for sentinel in _SENTINELS
        ):
            return candidate.resolve()
    raise RuntimeError(
        f"Could not locate the '{package_name}' project root from {start!r}."
    )


_ROOT: Path = _find_project_root(Path(__file__).resolve().parent, _PACKAGE_NAME)
_ROOT_STR: str = str(_ROOT)

if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)


def pytest_configure(config):  # noqa: ARG001 - pytest hook signature
    config._aiwonderland_root = _ROOT  # type: ignore[attr-defined]