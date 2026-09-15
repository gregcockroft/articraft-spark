"""Re-derive a recorded run's headline numbers from its own turn-by-turn log.

`stats.json` says a run took N turns and spent M output tokens. That is a summary written by the
tooling. `conversation.jsonl` is the log the harness wrote as the run happened, one JSON object per
message. This recomputes the summary from the log and reports any disagreement, so the numbers in a
sample's README can be checked rather than taken on trust.

    python spark/verify_run.py spark/results/<sample>

Needs nothing but the standard library. Exit 0 when every number agrees, 1 when one does not, and 2
when the sample carries no conversation.jsonl (not every recorded run has one).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def tool_name(call: dict) -> str:
    return call.get("name") or (call.get("function") or {}).get("name") or "?"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    sample = Path(sys.argv[1])
    log = sample / "conversation.jsonl"
    stats_path = sample / "stats.json"
    if not log.exists():
        print(f"{sample}: no conversation.jsonl, nothing to re-derive")
        return 2
    if not stats_path.exists():
        print(f"{sample}: no stats.json to check against")
        return 2

    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    turns = [r for r in rows if r.get("role") == "assistant"]
    usage = [r["token_usage"] for r in turns if r.get("token_usage")]
    calls = [(i, tool_name(c)) for i, r in enumerate(turns, 1) for c in (r.get("tool_calls") or [])]
    compiles = [i for i, name in calls if name == "compile"]

    stats = json.loads(stats_path.read_text())
    conv = stats.get("conversation", {})

    derived = {
        "turns": len(turns),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "peak_turn_input_tokens": max((u["input_tokens"] for u in usage), default=0),
        "compile_calls": len(compiles),
        "first_clean_compile_turn": compiles[-1] if compiles else None,
    }
    claimed = {
        "turns": conv.get("turns"),
        "output_tokens": stats.get("output_tokens"),
        "input_tokens": stats.get("input_tokens"),
        "peak_turn_input_tokens": conv.get("peak_turn_input_tokens"),
        "compile_calls": conv.get("compile_calls"),
        "first_clean_compile_turn": conv.get("first_clean_compile_turn"),
    }

    bad = 0
    print(f"{sample}  ({len(rows)} messages, {len(turns)} assistant turns)")
    for key, got in derived.items():
        want = claimed[key]
        if want is None:
            print(f"   {key:26s} {got!s:>10}   (stats.json does not claim this)")
        elif got == want:
            print(f"   {key:26s} {got!s:>10}   matches stats.json  OK")
        else:
            print(f"   {key:26s} {got!s:>10}   stats.json says {want}  MISMATCH")
            bad += 1

    names: dict[str, int] = {}
    for _, name in calls:
        names[name] = names.get(name, 0) + 1
    print("   tool calls                " + ", ".join(f"{n}x{c}" for n, c in sorted(names.items())))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
