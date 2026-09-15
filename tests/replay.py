"""Replay a recorded draw's model turns through the real agent loop.

A recorded run leaves ``conversation.jsonl``: system, user, then alternating
assistant turns and ``function_call_output`` results. This module turns that
file's **assistant turns** into a :class:`~harness.ScriptedModel` script and
runs the real :class:`~articraft.agent.harness.Agent` against a fresh
workspace, so a draw that cost hours on a GPU can be re-run through changed
harness code in seconds with no model.

Only the model's *actions* are replayed -- the tool calls, with their recorded
names, ids and argument bytes, and the visible text of a turn that made no
call. The reasoning is dropped: it belongs to the model, and replaying it
would measure the record rather than the harness.

The tool *results* are whatever the changed code produces now, which is the
point. A replay is therefore a counterfactual ("what would the loop do if the
model acted this way again"), not a reconstruction of the original run. Where
the recorded results mattered -- the model reacting to what a tool said -- the
replay cannot follow, and the divergence report says where the two parted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness import Response, RunArtifacts, ScriptedModel, calls, run_scenario, text

import articraft.agent.harness as agent_harness
from articraft.agent.workspace.local import LocalWorkspace

__all__ = [
    "ReplayOutcome",
    "ReplayScript",
    "load_replay",
    "replay",
]

# A replay collapses the recorded run's wall clock: turns that were minutes
# apart while the model generated tokens arrive here in milliseconds. That is
# not neutral. `exec_command` can leave a process running in the background
# (`running: true`), and the harness refuses every later tool call while one
# is live (`harness.py:446-451`). The recorded run's model latency let those
# processes exit; a replay would still be holding them open, so a compile the
# record shows succeeding is refused -- an artefact of the instrument, not a
# behaviour of the code under test.
#
# So the replay waits for a live session before each turn, which restores the
# one thing model latency provided. It is bounded, and every wait is reported,
# because a silent wait would hide a genuine hang.
_SETTLE_TIMEOUT = 120.0
_SETTLE_POLL = 0.1


@dataclass(frozen=True)
class ReplayScript:
    """The parts of a recorded conversation a replay needs."""

    prompt: str
    steps: list[dict[str, Any]]
    source: Path
    recorded_turns: int
    recorded_tool_counts: dict[str, int]
    images: list[str] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return len(self.steps)


@contextlib.contextmanager
def _captured_context() -> Iterator[list[Any]]:
    """Collect the ``ToolContext`` the agent builds, without touching ``src/``.

    The replay needs one thing from the running agent -- whether an exec
    session is still live -- and the harness keeps that on a context it
    creates itself. Wrapping the constructor for the duration of one run is
    the test-lane way to reach it; nothing in the package changes.
    """
    seen: list[Any] = []
    original = agent_harness.ToolContext

    def capture(*args: Any, **kwargs: Any) -> Any:
        context = original(*args, **kwargs)
        seen.append(context)
        return context

    agent_harness.ToolContext = capture  # type: ignore[assignment]
    try:
        yield seen
    finally:
        agent_harness.ToolContext = original  # type: ignore[assignment]


@dataclass
class _SettleLog:
    """Every wait the replay spent letting a background session finish."""

    waits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return round(sum(wait["seconds"] for wait in self.waits), 2)


class ReplayModel(ScriptedModel):
    """A ``ScriptedModel`` that waits for background work before each turn.

    See ``_SETTLE_TIMEOUT``: the wait restores the wall clock the recorded
    run's model latency supplied, and is recorded so it is never silent.
    """

    def __init__(self, steps: Iterable[Response], contexts: list[Any], log: _SettleLog):
        super().__init__(steps)
        self._contexts = contexts
        self._log = log

    async def _settle(self, turn: int) -> None:
        context = self._contexts[-1] if self._contexts else None
        sessions = getattr(context, "exec_sessions", None)
        if sessions is None or not sessions.live_ids():
            return
        started = time.monotonic()
        live = sessions.live_ids()
        while sessions.live_ids() and time.monotonic() - started < _SETTLE_TIMEOUT:
            await asyncio.sleep(_SETTLE_POLL)
        self._log.waits.append(
            {
                "turn": turn,
                "session_ids": list(live),
                "seconds": round(time.monotonic() - started, 2),
                "still_live": list(sessions.live_ids()),
            }
        )

    async def query(self, messages: Any, *, tools: Any = None) -> Response:
        await self._settle(len(self.queries) + 1)
        return await super().query(messages, tools=tools)


@dataclass(frozen=True)
class ReplayOutcome:
    """One replay's counts, beside the recorded draw's own."""

    script: ReplayScript
    artifacts: RunArtifacts
    consumed_turns: int
    tool_counts: dict[str, int]
    settle: _SettleLog = field(default_factory=_SettleLog)

    @property
    def terminate_reason(self) -> str | None:
        return self.artifacts.record.terminate_reason

    @property
    def status(self) -> str | None:
        return self.artifacts.record.status

    def counts(self) -> dict[str, Any]:
        """The row a trial reports: what the replay did, beside the record."""
        return {
            "source": str(self.script.source),
            "terminate_reason": self.terminate_reason,
            "status": self.status,
            "turns_replayed": self.consumed_turns,
            "turns_recorded": self.script.recorded_turns,
            "tool_calls_replayed": dict(sorted(self.tool_counts.items())),
            "tool_calls_recorded": dict(sorted(self.script.recorded_tool_counts.items())),
            "settled_seconds": self.settle.seconds,
            "settled_waits": self.settle.waits,
        }


def _prompt_of(rows: Iterable[dict[str, Any]]) -> tuple[str, list[str]]:
    """The first user message: the task the draw was given, and its images."""
    for row in rows:
        if row.get("role") == "user":
            content = row.get("content")
            images = [str(image) for image in (row.get("images") or [])]
            return str(content if content is not None else ""), images
    raise ValueError("no user message: not a recorded conversation")


def _tool_calls_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The recorded tool calls of one assistant turn, in the wire shape.

    Recorded calls carry ``id``/``name``/``arguments`` already, which is the
    shape ``ScriptedModel`` replays. A call missing either header is kept as
    recorded rather than repaired: how the loop treats a header-less call is
    exactly what item 5 changes, so the instrument must not fix it first.
    """
    out: list[dict[str, Any]] = []
    for call in row.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        out.append({"id": call.get("id"), "name": call.get("name"), "arguments": arguments})
    return out


def load_replay(path: str | Path) -> ReplayScript:
    """Parse a recorded ``conversation.jsonl`` into a replayable script."""
    source = Path(path)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    prompt, images = _prompt_of(rows)

    steps: list[dict[str, Any]] = []
    recorded_tool_counts: dict[str, int] = {}
    for row in rows:
        if row.get("role") != "assistant":
            continue
        tool_calls = _tool_calls_of(row)
        for call in tool_calls:
            name = str(call.get("name"))
            recorded_tool_counts[name] = recorded_tool_counts.get(name, 0) + 1
        if tool_calls:
            steps.append(calls(*tool_calls))
        else:
            steps.append(text(str(row.get("content") or "")))

    if not steps:
        raise ValueError(f"no assistant turns in {source}")
    return ReplayScript(
        prompt=prompt,
        steps=steps,
        source=source,
        recorded_turns=len(steps),
        recorded_tool_counts=recorded_tool_counts,
        images=images,
    )


def replay(
    path: str | Path,
    *,
    tmp_path: Path,
    max_turns: int | None = None,
    run_id: str = "replay",
    env: LocalWorkspace | None = None,
) -> ReplayOutcome:
    """Replay a recorded draw through the current agent loop.

    ``max_turns`` defaults to the recorded turn count, which is what makes a
    draw that ended at its own ceiling end at the ceiling here too. The script
    is not asserted exhausted: a loop that ends the run early -- which is what
    a repeat-failure guard is for -- leaves steps unconsumed by design, and
    that gap is the measurement.
    """
    script = load_replay(path)
    settle = _SettleLog()
    with _captured_context() as contexts:
        model = ReplayModel(script.steps, contexts, settle)
        artifacts = run_scenario(
            script.prompt,
            model=model,
            env=env,
            tmp_path=None if env is not None else tmp_path,
            run_id=run_id,
            max_turns=max_turns if max_turns is not None else script.turns,
            assert_exhausted=False,
        )
    consumed = script.turns - model.remaining
    tool_counts: dict[str, int] = {}
    for query_index in range(consumed):
        for call in script.steps[query_index].get("tool_calls") or []:
            name = str(call.get("name"))
            tool_counts[name] = tool_counts.get(name, 0) + 1
    return ReplayOutcome(
        script=script,
        artifacts=artifacts,
        consumed_turns=consumed,
        tool_counts=tool_counts,
        settle=settle,
    )
