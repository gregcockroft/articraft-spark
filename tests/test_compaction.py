from __future__ import annotations

import json

from articraft.agent.compaction import (
    KEEP_RECENT_TOKENS,
    RESERVE_TOKENS,
    prepare_compaction,
)


def _history(total_tokens: int) -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "quickstart"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "build the reference"},
                {"type": "input_image", "image_url": "data:image/png;base64,REFERENCE"},
            ],
        },
        {
            "role": "assistant",
            "content": "old work",
            "tool_calls": [
                {
                    "name": "write",
                    "arguments": json.dumps({"path": "main.py", "content": "file body"}),
                },
                {
                    "name": "view_image",
                    "arguments": json.dumps({"path": "preview.png"}),
                },
            ],
        },
        {"type": "function_call_output", "call_id": "write", "output": "written"},
        {
            "role": "assistant",
            "content": "r" * (KEEP_RECENT_TOKENS * 4),
            "tool_calls": [],
            "token_usage": {"total_tokens": total_tokens},
        },
    ]


def test_compaction_threshold_and_message_replacement() -> None:
    threshold = 272_000 - RESERVE_TOKENS
    assert prepare_compaction(_history(threshold), 272_000) is None

    messages = _history(threshold + 1)
    plan = prepare_compaction(messages, 272_000)
    assert plan is not None

    compacted = plan.apply(messages, "checkpoint")
    assert plan.tokens_before == threshold + 1
    assert compacted[:3] == messages[:3]
    assert compacted[2]["content"][1]["image_url"].endswith("REFERENCE")
    assert compacted[3]["compaction"]["summary"] == "checkpoint"
    assert compacted[4:] == plan.recent_messages


def test_summary_input_omits_large_payloads() -> None:
    messages = _history(260_000)
    messages[3]["tool_calls"][0]["arguments"] = json.dumps(
        {"path": "main.py", "content": "SECRET_FILE_BODY" * 200}
    )
    messages[3]["tool_calls"].append(
        {
            "name": "edit",
            "arguments": json.dumps(
                {
                    "path": "main.py",
                    "edits": [
                        {
                            "old_text": "SECRET_OLD_TEXT" * 200,
                            "new_text": "SECRET_NEW_TEXT" * 200,
                        }
                    ],
                }
            ),
        }
    )
    messages[4]["output"] = [
        {"type": "input_text", "text": "x" * 3_000},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,SECRET_IMAGE_DATA",
        },
    ]

    plan = prepare_compaction(messages, 272_000)
    assert plan is not None
    summary_input = plan.summary_messages[1]["content"]
    assert "SECRET_FILE_BODY" not in summary_input
    assert "SECRET_OLD_TEXT" not in summary_input
    assert "SECRET_NEW_TEXT" not in summary_input
    assert "SECRET_IMAGE_DATA" not in summary_input
    assert "[file content omitted]" in summary_input
    assert "[edit text omitted]" in summary_input
    assert "more characters omitted" in summary_input
    assert "image payload omitted" in summary_input


def test_repeated_compaction_updates_checkpoint_and_paths() -> None:
    messages = _history(260_000)
    first = prepare_compaction(messages, 272_000)
    assert first is not None

    compacted = first.apply(messages, "first checkpoint")
    compacted.extend(
        [
            {"type": "function_call_output", "call_id": "view", "output": "viewed"},
            {
                "role": "assistant",
                "content": "n" * (KEEP_RECENT_TOKENS * 4),
                "tool_calls": [
                    {
                        "name": "edit",
                        "arguments": json.dumps({"path": "previews.py"}),
                    }
                ],
                "token_usage": {"total_tokens": 260_000},
            },
        ]
    )

    second = prepare_compaction(compacted, 272_000)
    assert second is not None
    summary_input = second.summary_messages[1]["content"]
    assert "<previous-checkpoint>\nfirst checkpoint" in summary_input
    assert "<compaction-checkpoint>" not in summary_input
    assert second.modified_files == ["main.py", "previews.py"]
    assert second.inspected_images == ["preview.png"]


def test_the_reply_the_server_may_write_is_reserved_from_the_window() -> None:
    """A 229k prompt with a 32k reply cap overflowed a 262k window before compaction fired."""
    window, reply = 262_144, 32_768
    threshold = window - RESERVE_TOKENS - reply
    assert prepare_compaction(_history(threshold), window, response_tokens=reply) is None
    assert prepare_compaction(_history(threshold + 1), window, response_tokens=reply) is not None
    # Without the reserve the same history would not compact, which is how the run died.
    assert prepare_compaction(_history(229_377), window) is None
    assert prepare_compaction(_history(229_377), window, response_tokens=reply) is not None


def test_reasoning_sent_back_with_assistant_turns_counts_toward_the_recent_messages_kept() -> None:
    """The kept tail is measured by estimate; reasoning the provider resends is part of it.

    A recorded run kept 80k estimated (about 110k real) tokens of "recent" messages because its
    reasoning was not counted, so a compaction at 118k left 116k.
    """
    messages = _history(250_000)
    for index in range(6):
        messages.append({"type": "function_call_output", "call_id": f"c{index}", "output": "ok"})
        messages.append(
            {
                "role": "assistant",
                "content": "short",
                "tool_calls": [],
                "provider_content": [
                    {"type": "openrouter_reasoning", "reasoning": "r" * (12_000 * 4)}
                ],
                "token_usage": {"total_tokens": 250_000},
            }
        )
    plan = prepare_compaction(messages, 262_144)
    assert plan is not None
    # Two reasoning turns already pass KEEP_RECENT_TOKENS; without counting them all six were kept.
    kept = [m for m in plan.recent_messages if m.get("role") == "assistant"]
    assert len(kept) == 2


def test_text_a_provider_keeps_beside_the_content_is_not_counted_twice() -> None:
    message = {
        "role": "assistant",
        "content": "x" * 4_000,
        "provider_content": [{"type": "text", "text": "x" * 4_000}],
    }
    from articraft.agent.compaction import _message_tokens

    assert _message_tokens(message) == 1_000


def test_a_turn_that_names_its_context_is_measured_by_it_not_by_its_billed_total() -> None:
    """A thinking-budget turn bills two requests; its usage nearly doubles the real window."""
    messages = _history(250_000)
    assert prepare_compaction(messages, 262_144) is not None

    messages[-1]["context_tokens"] = 125_000
    assert prepare_compaction(messages, 262_144) is None
