"""The compile gate: ask for a compile when a run edits for N turns without one.

Twice a local-model run has spent a whole hundred-turn budget building a
workspace and recorded nothing, because it never called ``compile``. The gate
counts consecutive turns that change the workspace without compiling it and
appends one ``<compile_required>`` note when that count reaches its threshold.

**The gate is off by default** (``DEFAULT_COMPILE_GATE_TURNS = 0``), so every
test that exercises it passes ``compile_gate_turns=`` explicitly rather than
relying on a default. ``RESTORE_N = 8`` is the value that restores the gate as
it shipped, and the restore path itself is tested from the environment variable.

Every test here is the scripted-agent lane (``ScriptedModel`` + ``run_scenario``,
no paid call, no real compile), and every guard in the gate has a case where its
condition is false: the count below N, a compile in the turn, a read-only turn,
a failed compile, and a live exec session.
"""

from __future__ import annotations

from typing import Any

import pytest
from harness import (
    ModelQuery,
    Response,
    RunArtifacts,
    calls,
    compile_success_tool,
    fake_compile_payload,
    run_scenario,
    stub_schema,
    text,
    tool_call,
)
from pydantic import ValidationError

import articraft.agent.tools as agent_tools
from articraft.agent.harness import AgentConfig
from articraft.agent.tools import Tool, ToolContext
from articraft.settings import DEFAULT_COMPILE_GATE_TURNS, Settings

# The value that restores the gate as it shipped before it was turned off by default.
RESTORE_N = 8
N = RESTORE_N
GATE_SENTENCE = f"The last {N} turns changed the workspace"


def _write(index: int) -> Response:
    """One workspace-changing turn, on its own path so no rewrite guard fires."""
    return calls(tool_call("write", {"path": f"part{index}.py", "content": f"VALUE = {index}\n"}))


def _writes(start: int, count: int) -> list[Response]:
    return [_write(index) for index in range(start, start + count)]


def _gate_notes(artifacts: RunArtifacts) -> list[str]:
    """Every gate note the agent actually appended to the conversation."""
    return [
        str(event.get("content"))
        for event in artifacts.conversation
        if event.get("role") == "user" and GATE_SENTENCE in str(event.get("content"))
    ]


def compile_failure_tool() -> Tool:
    """A fake ``compile`` that always fails: a compile the gate must still count."""

    async def run_compile(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        result = fake_compile_payload(status="error", error="boom")
        context.compile_result = result
        return dict(result)

    return Tool("compile", stub_schema("compile"), run_compile)


def _tools(**extra: Tool) -> dict[str, Tool]:
    base = {"write": agent_tools.get("write"), "compile": compile_success_tool()}
    return base | extra


def test_gate_fires_once_after_n_editing_turns(monkeypatch, tmp_path) -> None:
    """P0.1: N editing turns produce exactly one note, in the NEXT request."""
    monkeypatch.setattr(agent_tools, "TOOLS", _tools())
    artifacts = run_scenario(
        "a box",
        [*_writes(1, N), calls(tool_call("compile")), text("done")],
        tmp_path=tmp_path,
        max_turns=N + 2,
        compile_gate_turns=N,
    )

    assert artifacts.record.status == "success"
    assert len(_gate_notes(artifacts)) == 1
    queries = artifacts.model.queries
    assert not queries[N - 1].contains(GATE_SENTENCE), "fired before the Nth editing turn ran"
    assert queries[N].contains("<compile_required>", GATE_SENTENCE)


def test_gate_never_fires_when_the_model_keeps_compiling(monkeypatch, tmp_path) -> None:
    """P0.2: the condition's false case -- a compile every few turns resets the count."""
    monkeypatch.setattr(agent_tools, "TOOLS", _tools())
    script: list[Response] = []
    for block in range(3):
        script += _writes(block * 3 + 1, 3)
        script.append(calls(tool_call("compile")))
    script.append(text("done"))

    artifacts = run_scenario(
        "a box", script, tmp_path=tmp_path, max_turns=len(script), compile_gate_turns=N
    )

    assert artifacts.record.status == "success"
    assert _gate_notes(artifacts) == []


def test_gate_fires_at_most_once_per_window(monkeypatch, tmp_path) -> None:
    """P0.3: 2N editing turns give 2 notes, not one on every turn past N."""
    monkeypatch.setattr(agent_tools, "TOOLS", _tools())
    artifacts = run_scenario(
        "a box",
        [*_writes(1, 2 * N), calls(tool_call("compile")), text("done")],
        tmp_path=tmp_path,
        max_turns=2 * N + 2,
        compile_gate_turns=N,
    )

    assert artifacts.record.status == "success"
    assert len(_gate_notes(artifacts)) == 2
    queries = artifacts.model.queries
    assert queries[N].contains(GATE_SENTENCE)
    assert queries[2 * N].contains(GATE_SENTENCE)


def test_reading_turns_neither_advance_nor_reset_the_count(monkeypatch, tmp_path) -> None:
    """P0.7: a read-only turn is not an edit, and is not a compile either."""
    monkeypatch.setattr(agent_tools, "TOOLS", _tools(read=agent_tools.get("read")))
    reads = [calls(tool_call("read", {"path": "part1.py"})) for _ in range(3)]
    artifacts = run_scenario(
        "a box",
        [
            *_writes(1, N - 1),
            *reads,
            _write(N),
            calls(tool_call("compile")),
            text("done"),
        ],
        tmp_path=tmp_path,
        max_turns=N + 5,
        compile_gate_turns=N,
    )

    assert artifacts.record.status == "success"
    assert len(_gate_notes(artifacts)) == 1
    queries = artifacts.model.queries
    # The three reads sit at turns N..N+2 and must not carry the count over N.
    assert not queries[N + 1].contains(GATE_SENTENCE), "a read advanced the count"
    assert not queries[N + 2].contains(GATE_SENTENCE), "a read advanced the count"
    # The Nth *editing* turn is turn N+3; its note is in the request after it.
    assert queries[N + 3].contains(GATE_SENTENCE), "the reads reset the count"


def test_a_failed_compile_still_resets_the_count(monkeypatch, tmp_path) -> None:
    """P0.8: the gate asks for a compile, not for a successful one."""
    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        {"write": agent_tools.get("write"), "compile": compile_failure_tool()},
    )
    artifacts = run_scenario(
        "a box",
        [
            *_writes(1, N - 1),
            calls(tool_call("compile")),
            *_writes(N, N - 1),
            text("giving up"),
        ],
        tmp_path=tmp_path,
        max_turns=2 * N,
        compile_gate_turns=N,
        assert_exhausted=False,
    )

    # Two runs of N-1 editing turns either side of one failed compile: without
    # the reset the second run would cross N and fire.
    assert _gate_notes(artifacts) == []


def test_a_live_exec_session_defers_the_note(monkeypatch, tmp_path) -> None:
    """The gate never asks for a compile the agent would refuse to run."""
    monkeypatch.setattr(
        agent_tools,
        "TOOLS",
        _tools(
            exec_command=agent_tools.get("exec_command"),
            write_stdin=agent_tools.get("write_stdin"),
        ),
    )

    def send_stdin(query: ModelQuery) -> Response:
        session_id = next(
            output["result"]["session_id"]
            for output in query.tool_outputs()
            if output.get("result", {}).get("session_id")
        )
        return calls(
            tool_call(
                "write_stdin", {"session_id": session_id, "chars": "go\n", "yield_time_ms": 2000}
            )
        )

    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("exec_command", {"command": "read value", "yield_time_ms": 10})),
            *_writes(1, N),
            send_stdin,
            _write(N + 1),
            calls(tool_call("compile")),
            text("done"),
        ],
        tmp_path=tmp_path,
        max_turns=N + 5,
        compile_gate_turns=N,
    )

    assert artifacts.record.status == "success"
    assert len(_gate_notes(artifacts)) == 1
    queries = artifacts.model.queries
    # The Nth editing turn is turn N+1, and the session is still live there.
    assert not queries[N + 1].contains(GATE_SENTENCE), "asked for a compile mid exec_command"
    # Turn N+2 ends the session; turn N+3 edits again and the held note lands.
    assert queries[N + 3].contains(GATE_SENTENCE)


def test_gate_turns_zero_turns_the_gate_off(monkeypatch, tmp_path) -> None:
    """0: twenty editing turns with no compile append no note at all."""
    monkeypatch.setattr(agent_tools, "TOOLS", _tools())
    artifacts = run_scenario(
        "a box",
        [*_writes(1, 20), calls(tool_call("compile")), text("done")],
        tmp_path=tmp_path,
        max_turns=22,
        compile_gate_turns=0,
    )

    assert artifacts.record.status == "success"
    assert _gate_notes(artifacts) == []
    assert not any(
        "<compile_required>" in str(event.get("content"))
        for event in artifacts.conversation
        if event.get("role") == "user"
    )


def test_the_gate_is_off_by_default(monkeypatch, tmp_path) -> None:
    """The default: a run that never compiles is never nudged (Greg's order, 2026-09-19)."""
    monkeypatch.setattr(agent_tools, "TOOLS", _tools())
    artifacts = run_scenario(
        "a box",
        [*_writes(1, 20), calls(tool_call("compile")), text("done")],
        tmp_path=tmp_path,
        max_turns=22,
    )

    assert AgentConfig().compile_gate_turns == 0
    assert artifacts.record.status == "success"
    assert not any(
        "<compile_required>" in str(event.get("content"))
        for event in artifacts.conversation
        if event.get("role") == "user"
    )


def test_setting_eight_restores_the_gate_end_to_end(monkeypatch, tmp_path) -> None:
    """The restore path the order promises: ARTICRAFT_COMPILE_GATE_TURNS=8 brings the gate back.

    The threshold comes from ``Settings`` rather than a literal, so this fails if the
    environment variable ever stops reaching the agent.
    """
    monkeypatch.setattr(agent_tools, "TOOLS", _tools())
    monkeypatch.setenv("ARTICRAFT_COMPILE_GATE_TURNS", str(RESTORE_N))
    resolved = Settings().compile_gate_turns  # pyright: ignore[reportCallIssue]
    assert resolved == RESTORE_N

    artifacts = run_scenario(
        "a box",
        [*_writes(1, resolved), calls(tool_call("compile")), text("done")],
        tmp_path=tmp_path,
        max_turns=resolved + 2,
        compile_gate_turns=resolved,
    )

    assert artifacts.record.status == "success"
    assert len(_gate_notes(artifacts)) == 1
    queries = artifacts.model.queries
    assert not queries[resolved - 1].contains(GATE_SENTENCE), "fired before the Nth editing turn"
    assert queries[resolved].contains("<compile_required>", GATE_SENTENCE)


def test_gate_turns_setting_moves_the_threshold(monkeypatch, tmp_path) -> None:
    """The setting is the threshold: 3 fires after three editing turns, and says 3."""
    monkeypatch.setattr(agent_tools, "TOOLS", _tools())
    artifacts = run_scenario(
        "a box",
        [*_writes(1, 3), calls(tool_call("compile")), text("done")],
        tmp_path=tmp_path,
        max_turns=5,
        compile_gate_turns=3,
    )

    queries = artifacts.model.queries
    assert not queries[2].contains("<compile_required>")
    assert queries[3].contains("The last 3 turns changed the workspace")


def test_settings_carry_the_gate_to_the_agent(monkeypatch) -> None:
    """ARTICRAFT_COMPILE_GATE_TURNS reaches the agent; unset, it is the module default of 0."""
    monkeypatch.delenv("ARTICRAFT_COMPILE_GATE_TURNS", raising=False)
    assert DEFAULT_COMPILE_GATE_TURNS == 0
    assert Settings().compile_gate_turns == 0  # pyright: ignore[reportCallIssue]
    assert AgentConfig().compile_gate_turns == 0

    monkeypatch.setenv("ARTICRAFT_COMPILE_GATE_TURNS", str(RESTORE_N))
    assert Settings().compile_gate_turns == RESTORE_N  # pyright: ignore[reportCallIssue]

    monkeypatch.setenv("ARTICRAFT_COMPILE_GATE_TURNS", "-1")
    with pytest.raises(ValidationError):
        Settings()  # pyright: ignore[reportCallIssue]
