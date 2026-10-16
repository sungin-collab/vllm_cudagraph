# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the pure CPU modules without vllm.__init__'s PyTorch initialization."""

import importlib
import importlib.util
import sys
from pathlib import Path

_PACKAGE = "_vllm_cudagraph_cpu"
_SOURCE = Path(__file__).resolve().parents[2] / "vllm" / "compilation"
if _PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PACKAGE, _SOURCE / "__init__.py", submodule_search_locations=[str(_SOURCE)]
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = package
    spec.loader.exec_module(package)

memory = importlib.import_module(f"{_PACKAGE}.cudagraph_memory")
planner = importlib.import_module(f"{_PACKAGE}.cudagraph_memory_planner")
