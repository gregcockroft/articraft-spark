"""The turn-budget reminder, and the salvage of a clean compile when a run ends on an error.

A run that edits on past its last compile reaches the turn limit with nothing recorded, and a model
error late in a run used to discard a revision that had compiled clean. The reminder fires when 20
and 10 turns remain, only while the latest workspace is not compiled clean, once each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import (
    GOOD_MAIN_PY,
    ScriptedModel,
    WarmEnvironment,
    calls,
    run_scenario,
    text,
    tool_call,
)

TAG = "<turn_budget>"


def write_main(content: str = GOOD_MAIN_PY):
    return calls(tool_call("write", {"path": "main.py", "content": content}))


def read_main():
    return calls(tool_call("read", {"path": "main.py"}))


def compile_workspace():
    return calls(tool_call("compile"))


def reminders(conversation: list[dict[str, Any]]) -> list[str]:
    return [
        str(event.get("content"))
        for event in conversation
        if event.get("role") == "user" and TAG in str(event.get("content"))
    ]


def test_an_uncompiled_run_is_told_its_budget_at_20_and_10_turns_left(tmp_path: Path) -> None:
    script = [write_main(), *[read_main() for _ in range(11)], compile_workspace(), text("done")]
    artifacts = run_scenario(
        "a box", script, env=WarmEnvironment(output_dir=tmp_path), max_turns=22
    )

    sent = reminders(artifacts.conversation)
    assert len(sent) == 2
    assert "20 turns remain" in sent[0]
    assert "10 turns remain" in sent[1]
    assert "No successful compile has completed yet." in sent[0]
    assert "the run records nothing" in sent[0]
    model = artifacts.model
    assert isinstance(model, ScriptedModel)
    queries = model.queries
    assert queries[2].contains(TAG, "20 turns remain")
    assert not queries[1].contains(TAG)
    assert artifacts.record.terminate_reason == "final_response"


def test_after_a_clean_compile_the_reminder_says_only_that_compile_is_kept(
    tmp_path: Path,
) -> None:
    marker = "object_model = build_object_model()"
    assert marker in GOOD_MAIN_PY
    edit = tool_call(
        "edit",
        {"path": "main.py", "edits": [{"old_text": marker, "new_text": marker + "  # v2"}]},
    )
    script = [
        write_main(),
        compile_workspace(),
        calls(edit),
        compile_workspace(),
        text("done"),
    ]
    artifacts = run_scenario(
        "a box", script, env=WarmEnvironment(output_dir=tmp_path), max_turns=23
    )

    sent = reminders(artifacts.conversation)
    assert len(sent) == 1
    assert "20 turns remain" in sent[0]
    assert "The workspace has changed since the last successful compile." in sent[0]
    assert "keeps only that last clean compile" in sent[0]


def test_a_compiled_workspace_gets_no_budget_reminder(tmp_path: Path) -> None:
    script = [write_main(), compile_workspace(), *[read_main() for _ in range(10)], text("done")]
    artifacts = run_scenario(
        "a box", script, env=WarmEnvironment(output_dir=tmp_path), max_turns=22
    )

    assert reminders(artifacts.conversation) == []
    assert artifacts.record.terminate_reason == "final_response"


def test_a_limit_as_long_as_the_run_never_fires(tmp_path: Path) -> None:
    # max_turns=11: 10 turns remain after turn 1, so only the 10-turn reminder exists.
    script = [write_main(), *[read_main() for _ in range(8)], compile_workspace(), text("done")]
    artifacts = run_scenario(
        "a box", script, env=WarmEnvironment(output_dir=tmp_path), max_turns=11
    )
    sent = reminders(artifacts.conversation)
    assert len(sent) == 1
    assert "10 turns remain" in sent[0]

    # max_turns=10: no limit is shorter than the run, so nothing fires.
    artifacts = run_scenario(
        "a box",
        [write_main(), *[read_main() for _ in range(7)], compile_workspace(), text("done")],
        env=WarmEnvironment(output_dir=tmp_path),
        run_id="short",
        max_turns=10,
    )
    assert reminders(artifacts.conversation) == []


def test_a_model_error_after_a_clean_compile_keeps_that_compile(tmp_path: Path) -> None:
    """A recorded run compiled clean at turn 75 and lost it to a context overflow at turn 76."""
    artifacts = run_scenario(
        "a box",
        [write_main(), compile_workspace()],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=5,
    )

    assert artifacts.record.status == "success"
    assert artifacts.record.terminate_reason == "error_last_clean"
    assert artifacts.record.result.endswith(".usdz")
    assert (artifacts.run_dir / artifacts.record.result).is_file()
    assert "ScriptExhaustedError" in artifacts.record.error


def test_a_model_error_with_no_clean_compile_is_still_an_error(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [write_main()],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=5,
    )

    assert artifacts.record.status == "error"
    assert artifacts.record.terminate_reason == "error"
    assert artifacts.record.result == ""


BROKEN_MAIN_PY = """
from articraft.sdk import RigidBodyAssembly


def build_object_model() -> RigidBodyAssembly:
    raise RuntimeError("a revision that does not build")


object_model = build_object_model()
"""


def test_a_salvaged_run_reports_the_clean_compile_it_kept(tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a box",
        [write_main(), compile_workspace(), write_main(BROKEN_MAIN_PY), compile_workspace()],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=6,
    )

    assert artifacts.record.terminate_reason == "error_last_clean"
    outputs = artifacts.tool_outputs()
    assert outputs[-1]["result"]["status"] != "success"
    assert artifacts.result["compile_report"]["status"] == "success"
