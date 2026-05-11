"""Track submission history in a local JSON file."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

SUBMISSIONS_PATH = os.environ.get("SUBMISSIONS_PATH", "data/submissions.json")
MAX_PER_TASK = 20


def _load() -> dict[str, list[dict[str, Any]]]:
    path = Path(SUBMISSIONS_PATH)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data: dict[str, list[dict[str, Any]]]) -> None:
    path = Path(SUBMISSIONS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_submission(
    task_id: str,
    code: str,
    passed: int,
    total: int,
    total_time: float,
    results: list[dict[str, Any]],
    output: str,
) -> None:
    """Record a submission. Auto-evicts oldest when exceeding MAX_PER_TASK."""
    data = _load()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "passed": passed,
        "total": total,
        "success": passed == total,
        "total_time": total_time,
        "results": results,
        "output": output,
    }
    task_history = data.get(task_id, [])
    task_history.append(entry)
    if len(task_history) > MAX_PER_TASK:
        task_history = task_history[-MAX_PER_TASK:]
    data[task_id] = task_history
    _save(data)


def get_submissions(task_id: str) -> list[dict[str, Any]]:
    """Return submission history for a task, newest first."""
    data = _load()
    return list(reversed(data.get(task_id, [])))


def clear_submissions(task_id: str | None = None) -> None:
    """Clear submissions for a specific task, or all if task_id is None."""
    if task_id is None:
        _save({})
    else:
        data = _load()
        data.pop(task_id, None)
        _save(data)
