"""The replay runner: a recorded draw re-run through the real loop, no model.

These tests are self-contained -- they synthesize a recorded conversation in
``tmp_path`` rather than reading a trial, so the suite never depends on a
recording living outside the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from harness import GOOD_MAIN_PY, WarmEnvironment
from replay import load_replay, replay


def _row(role: str, **fields: Any) -> dict[str, Any]:
    return {"role": role, **fields}


def _assistant(*tool_calls: dict[str, Any], content: str = "\n") -> dict[str, Any]:
    return _row(
        "assistant",
        content=content,
        tool_calls=list(tool_calls),
        token_usage={"output_tokens": 10},
    )


def _call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": json.dumps(args)}


def _write_conversation(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def recorded(tmp_path: Path) -> Path:
    """A two-turn draw that writes a good model, compiles it, then answers."""
    return _write_conversation(
        tmp_path / "conversation.jsonl",
        [
            _row("system", content="system prompt"),
            _row("user", content="a chest of drawers"),
            _assistant(_call("write", {"path": "main.py", "content": GOOD_MAIN_PY}, "c1")),
            {"type": "function_call_output", "call_id": "c1", "output": '{"result": "ok"}'},
            _assistant(_call("compile", {}, "c2")),
            {"type": "function_call_output", "call_id": "c2", "output": '{"result": "ok"}'},
            _assistant(content="done"),
        ],
    )


def test_load_replay_takes_the_actions_and_drops_the_reasoning(recorded: Path) -> None:
    script = load_replay(recorded)

    assert script.prompt == "a chest of drawers"
    assert script.recorded_turns == 3
    assert script.recorded_tool_counts == {"write": 1, "compile": 1}
    # The recorded ids and argument bytes are replayed as recorded.
    assert script.steps[0]["tool_calls"][0]["id"] == "c1"
    assert json.loads(script.steps[0]["tool_calls"][0]["arguments"])["path"] == "main.py"
    # The last turn made no call, so it replays as its visible text.
    assert script.steps[2]["text"] == "done"
    assert script.steps[2]["tool_calls"] == []
    # Nothing carries the recorded reasoning into the replay.
    assert all("provider_content" not in step for step in script.steps)


def test_replay_runs_the_recorded_actions_through_the_real_loop(
    recorded: Path, tmp_path: Path
) -> None:
    outcome = replay(recorded, tmp_path=tmp_path, env=WarmEnvironment(output_dir=tmp_path))

    assert outcome.terminate_reason == "final_response"
    assert outcome.consumed_turns == 3
    assert outcome.tool_counts == {"write": 1, "compile": 1}
    # The replay's counts are reported beside the record's, never instead.
    counts = outcome.counts()
    assert counts["tool_calls_replayed"] == counts["tool_calls_recorded"]


def test_replay_stops_where_the_loop_stops(recorded: Path, tmp_path: Path) -> None:
    """A loop that ends early leaves steps unconsumed, and the gap is the count."""
    outcome = replay(
        recorded, tmp_path=tmp_path, max_turns=1, env=WarmEnvironment(output_dir=tmp_path)
    )

    assert outcome.terminate_reason in {"max_turns", "max_turns_last_clean"}
    assert outcome.consumed_turns == 1
    assert outcome.script.recorded_turns == 3
    assert outcome.tool_counts == {"write": 1}


def test_replay_waits_for_a_backgrounded_session_and_says_so(tmp_path: Path) -> None:
    """Model latency let a background process finish; a replay must not outrun it.

    Without the wait the harness refuses the next tool call while the session
    is live (``harness.py:446-451``) and a compile the record shows succeeding
    is refused -- an artefact of the instrument, not of the code under test.
    """
    recorded = _write_conversation(
        tmp_path / "conversation.jsonl",
        [
            _row("system", content="system prompt"),
            _row("user", content="a chest of drawers"),
            _assistant(_call("write", {"path": "main.py", "content": GOOD_MAIN_PY}, "c1")),
            {"type": "function_call_output", "call_id": "c1", "output": '{"result": "ok"}'},
            # A command that outlives its own yield window, exactly as medium_1
            # d1's `previews.py` turn did -- session 17 came back running: true
            # at 10.0069 s, the default yield, with the process still alive.
            _assistant(_call("exec_command", {"command": "sleep 2", "yield_time_ms": 200}, "c2")),
            {"type": "function_call_output", "call_id": "c2", "output": '{"result": "ok"}'},
            _assistant(_call("compile", {}, "c3")),
            {"type": "function_call_output", "call_id": "c3", "output": '{"result": "ok"}'},
            _assistant(content="done"),
        ],
    )

    outcome = replay(recorded, tmp_path=tmp_path, env=WarmEnvironment(output_dir=tmp_path))

    assert outcome.settle.waits, "a live session should have been waited for"
    wait = outcome.settle.waits[0]
    assert wait["still_live"] == []
    assert wait["seconds"] > 0
    # And the compile is reached rather than refused for a session still open.
    assert outcome.tool_counts.get("compile") == 1
    assert outcome.terminate_reason == "final_response"
