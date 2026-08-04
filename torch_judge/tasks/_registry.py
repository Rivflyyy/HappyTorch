"""Auto-discovery registry for task definitions."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

CATEGORY_ORDER = [
    "基础层",
    "注意力机制",
    "完整架构",
    "现代激活函数",
    "参数高效微调",
    "条件调制 — Diffusion",
    "LLM 推理组件",
    "扩散模型训练",
    "ML 基础与解码策略",
    "NumPy 手写神经网络",
    "RLHF",
]
_CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORY_ORDER)}

TASKS: dict[str, dict[str, Any]] = {}

_pkg_dir = str(Path(__file__).parent)
for _info in pkgutil.iter_modules([_pkg_dir]):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__package__}.{_info.name}")
    if hasattr(_mod, "TASK"):
        TASKS[_info.name] = _mod.TASK


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id)


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        TASKS.items(),
        key=lambda t: (
            _CATEGORY_INDEX.get(t[1].get("category", ""), 99),
            DIFFICULTY_ORDER.get(t[1]["difficulty"], 9),
        ),
    )
