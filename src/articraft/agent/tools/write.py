from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from articraft.agent.tools._core import Tool, ToolContext, display_path, schema, workspace_path

MIN_REWRITE_SHARE = 0.4
"""Smallest share of changed lines a whole-file rewrite of an existing file may have."""


def changed_line_share(old: str, new: str) -> tuple[int, int, float]:
    """Return changed lines, total lines, and the changed share of a rewrite."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    total = max(len(old_lines), len(new_lines))
    if total == 0:
        return 0, 0, 0.0
    changed = 0
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            changed += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            changed += i2 - i1
        elif tag == "insert":
            changed += j2 - j1
    return changed, total, changed / total


def _current_text(path: Path) -> str | None:
    """The file's current text, or None when there is no diff to measure."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = workspace_path(context.workspace, str(args["path"]))
    content = str(args["content"])
    resolved = path.resolve()
    current = _current_text(path) if resolved in context.written_paths and path.is_file() else None
    if current is not None:
        changed, total, share = changed_line_share(current, content)
        if share < MIN_REWRITE_SHARE:
            raise ValueError(
                f"{display_path(context.workspace, path)} was already written and this write changes "
                f"{share:.0%} of its lines ({changed} of {total}). Use edit for a change this small: "
                "it takes exact old_text/new_text replacements against the current file, and several "
                "disjoint replacements can go in one call. write is for a new file or for replacing at "
                f"least {MIN_REWRITE_SHARE:.0%} of an existing file's lines."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    context.written_paths.add(resolved)
    return {
        "path": display_path(context.workspace, path),
        "bytes": len(content.encode("utf-8")),
    }


TOOL = Tool(
    "write",
    schema(
        "write",
        "Create or overwrite a file in the run workspace. Parent directories are created automatically. "
        "Rewriting a file you already wrote is for an intentional whole-file replacement: a write that "
        "changes less than 40% of its lines is refused, and edit is the tool for a change that size.",
        {
            "path": {
                "type": "string",
                "description": "Path to create or overwrite inside the run workspace. Relative paths are resolved against the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Complete UTF-8 file contents to write.",
            },
        },
        ["path", "content"],
    ),
    run,
)
