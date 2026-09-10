"""OpenRouter provider pointed at a local OpenAI-compatible server (vLLM)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest

from articraft.agent.provider.openrouter import OpenRouterModel
from articraft.errors import ModelError
from articraft.settings import Settings

LOCAL_BASE_URL = "http://localhost:8001/v1"
PNG = base64.b64encode(b"fake-png-bytes").decode("ascii")
DATA_URL = f"data:image/png;base64,{PNG}"


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


def text_response(text: str = "done") -> dict[str, Any]:
    return {
        "id": "gen-local",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
    }


def local_model(
    responses: list[dict[str, Any]],
    **settings: Any,
) -> tuple[OpenRouterModel, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=responses.pop(0))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    settings.setdefault("provider", "openrouter")
    settings.setdefault("openrouter_base_url", LOCAL_BASE_URL)
    settings.setdefault("openrouter_model", "qwen")
    return OpenRouterModel(Settings(**settings), client=client), requests


def request_json(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_local_base_url_is_used_and_needs_no_api_key() -> None:
    model, requests = local_model([text_response()])

    result = run(model.query([{"role": "user", "content": "build a hinge"}]))

    assert result["text"] == "done"
    assert requests[0].url == "http://localhost:8001/v1/chat/completions"
    assert "Authorization" not in requests[0].headers


def test_local_base_url_still_sends_a_key_when_one_is_set() -> None:
    model, requests = local_model([text_response()], openrouter_api_key="local")

    run(model.query([{"role": "user", "content": "build a hinge"}]))

    assert requests[0].headers["Authorization"] == "Bearer local"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8001/v1",
        "http://localhost:8001/v1/",
        "http://localhost:8001/v1/chat/completions",
    ],
)
def test_base_url_forms_reach_the_same_endpoint(base_url: str) -> None:
    model, requests = local_model([text_response()], openrouter_base_url=base_url)

    run(model.query([{"role": "user", "content": "hi"}]))

    assert requests[0].url == "http://localhost:8001/v1/chat/completions"


def test_openrouter_host_still_requires_a_key() -> None:
    with pytest.raises(ModelError, match="OpenRouter credentials are required"):
        OpenRouterModel(
            Settings(provider="openrouter", openrouter_model="qwen", openrouter_api_key=None)
        )


def test_images_are_off_by_default_and_rejected() -> None:
    model, _ = local_model([text_response()])

    assert model.supports_images is False
    with pytest.raises(ModelError, match="ARTICRAFT_OPENROUTER_IMAGES"):
        run(
            model.query(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "match this"},
                            {"type": "input_image", "image_url": DATA_URL, "detail": "high"},
                        ],
                    }
                ]
            )
        )


def test_images_are_sent_as_image_url_parts_when_enabled() -> None:
    model, requests = local_model([text_response()], openrouter_supports_images=True)

    assert model.supports_images is True
    run(
        model.query(
            [
                {"role": "system", "content": "write clean code"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "match this"},
                        {"type": "input_image", "image_url": DATA_URL, "detail": "high"},
                    ],
                },
            ]
        )
    )

    assert request_json(requests[0])["messages"] == [
        {"role": "system", "content": "write clean code"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "match this"},
                {"type": "image_url", "image_url": {"url": DATA_URL, "detail": "high"}},
            ],
        },
    ]


def test_original_detail_is_dropped_because_the_api_rejects_it() -> None:
    model, requests = local_model([text_response()], openrouter_supports_images=True)

    run(
        model.query(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": DATA_URL, "detail": "original"}
                    ],
                }
            ]
        )
    )

    content = request_json(requests[0])["messages"][0]["content"]
    assert content == [{"type": "image_url", "image_url": {"url": DATA_URL}}]


def test_tool_result_image_rides_in_a_following_user_message() -> None:
    """A chat-completions tool message carries text only, so view_image needs a carrier."""
    model, requests = local_model([text_response()], openrouter_supports_images=True)

    run(
        model.query(
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_view",
                    "output": [
                        {"type": "input_text", "text": '{"result": {"path": "shot.png"}}'},
                        {"type": "input_image", "image_url": DATA_URL, "detail": "high"},
                    ],
                }
            ]
        )
    )

    assert request_json(requests[0])["messages"] == [
        {
            "role": "tool",
            "tool_call_id": "call_view",
            "content": '{"result": {"path": "shot.png"}}',
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image returned by the preceding tool call."},
                {"type": "image_url", "image_url": {"url": DATA_URL, "detail": "high"}},
            ],
        },
    ]


def test_text_only_tool_result_adds_no_carrier_message() -> None:
    model, requests = local_model([text_response()])

    run(
        model.query(
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_read",
                    "output": '{"result": "ok"}',
                }
            ]
        )
    )

    assert request_json(requests[0])["messages"] == [
        {"role": "tool", "tool_call_id": "call_read", "content": '{"result": "ok"}'}
    ]


def test_reasoning_from_vllm_round_trips_into_the_next_request() -> None:
    """vLLM's qwen3 reasoning parser emits message.reasoning, which must be replayed."""
    model, requests = local_model(
        [
            {
                "id": "gen-local",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "thinking done",
                            "tool_calls": [],
                            "reasoning": "the drawer needs a slide joint",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            text_response("second"),
        ]
    )

    first = run(model.query([{"role": "user", "content": "build a dresser"}]))
    assert first["provider_content"] == [
        {"type": "openrouter_reasoning", "reasoning": "the drawer needs a slide joint"}
    ]

    run(
        model.query(
            [
                {"role": "user", "content": "build a dresser"},
                {
                    "role": "assistant",
                    "content": "thinking done",
                    "tool_calls": [],
                    "provider_content": first["provider_content"],
                },
            ]
        )
    )
    assistant = request_json(requests[1])["messages"][1]
    assert assistant["reasoning"] == "the drawer needs a slide joint"


def test_local_endpoint_passes_preflight_without_a_key() -> None:
    """The CLI preflight must not demand a credential a local server never checks."""
    from articraft.api import _missing_provider_settings

    settings = Settings(
        provider="openrouter",
        openrouter_model="qwen",
        openrouter_base_url=LOCAL_BASE_URL,
        openrouter_api_key=None,
    )
    assert _missing_provider_settings(settings) == []


def test_openrouter_host_preflight_still_demands_a_key() -> None:
    from articraft.api import _missing_provider_settings

    settings = Settings(provider="openrouter", openrouter_model="qwen", openrouter_api_key=None)
    assert _missing_provider_settings(settings) == ["OPENROUTER_API_KEY"]


def test_preflight_still_demands_a_model() -> None:
    from articraft.api import _missing_provider_settings

    settings = Settings(
        provider="openrouter", openrouter_base_url=LOCAL_BASE_URL, openrouter_model=" "
    )
    assert _missing_provider_settings(settings) == ["ARTICRAFT_OPENROUTER_MODEL or --model"]


def reasoning_only_response() -> dict[str, Any]:
    """What vLLM returns when a turn produced thinking and nothing else."""
    return {
        "id": "gen-local",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [],
                    "reasoning": "Let me think about the drawer slides before I write anything.",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_reasoning_only_turn_is_empty_not_fatal() -> None:
    """It must not raise: the agent loop has its own empty-response handling."""
    model, _ = local_model([reasoning_only_response()])

    result = run(model.query([{"role": "user", "content": "build a dresser"}]))

    assert result["text"] == ""
    assert result["tool_calls"] == []
    assert result["provider_content"] == [
        {
            "type": "openrouter_reasoning",
            "reasoning": "Let me think about the drawer slides before I write anything.",
        }
    ]


def test_reasoning_is_not_promoted_into_text() -> None:
    """Promoting it would let thinking end the run as if it were the final answer."""
    model, _ = local_model([reasoning_only_response()])

    result = run(model.query([{"role": "user", "content": "build a dresser"}]))

    assert "drawer slides" not in result["text"]


def test_a_truly_empty_response_still_raises() -> None:
    model, _ = local_model(
        [
            {
                "id": "gen-local",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
            }
        ]
    )

    with pytest.raises(ModelError, match="did not contain text or tool calls"):
        run(model.query([{"role": "user", "content": "build a dresser"}]))


def test_generation_sends_no_max_tokens_by_default() -> None:
    """Unset keeps the upstream behaviour: the hosted model bounds its own reply."""
    model, requests = local_model([text_response()])

    run(model.query([{"role": "user", "content": "build a dresser"}]))

    assert "max_tokens" not in request_json(requests[0])


def test_generation_caps_output_when_configured() -> None:
    """A self-hosted server bounds nothing, so one turn can run to the context ceiling."""
    model, requests = local_model([text_response()], openrouter_max_output_tokens=32_768)

    run(model.query([{"role": "user", "content": "build a dresser"}]))

    assert request_json(requests[0])["max_tokens"] == 32_768


def test_chat_template_kwargs_are_sent_when_configured() -> None:
    """vLLM applies the model's own chat template; Qwen3 thinking is switched there."""
    model, requests = local_model(
        [text_response()],
        openrouter_chat_template_kwargs='{"enable_thinking": false}',
    )

    run(model.query([{"role": "user", "content": "build a dresser"}]))

    assert request_json(requests[0])["chat_template_kwargs"] == {"enable_thinking": False}


def test_no_chat_template_kwargs_by_default() -> None:
    model, requests = local_model([text_response()])

    run(model.query([{"role": "user", "content": "build a dresser"}]))

    assert "chat_template_kwargs" not in request_json(requests[0])


def test_malformed_chat_template_kwargs_is_a_clear_error() -> None:
    model, _ = local_model([text_response()], openrouter_chat_template_kwargs="not json")

    with pytest.raises(ModelError, match="must be a JSON object"):
        run(model.query([{"role": "user", "content": "build a dresser"}]))


def _image_message(tag: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": tag},
            {"type": "input_image", "image_url": f"{DATA_URL}#{tag}", "detail": "high"},
        ],
    }


def test_images_are_not_pruned_by_default() -> None:
    model, requests = local_model([text_response()], openrouter_supports_images=True)

    run(model.query([_image_message(f"i{n}") for n in range(6)]))

    parts = [
        item
        for message in request_json(requests[0])["messages"]
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    assert len(parts) == 6


def test_pruning_keeps_the_reference_and_the_most_recent() -> None:
    """The oldest image is the reference being reconstructed - never drop it first."""
    model, requests = local_model(
        [text_response()],
        openrouter_supports_images=True,
        openrouter_max_images=3,
    )

    run(model.query([_image_message(f"i{n}") for n in range(6)]))

    urls = [
        item["image_url"]["url"]
        for message in request_json(requests[0])["messages"]
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    assert len(urls) == 3
    assert urls[0].endswith("#i0"), "the reference image must survive"
    assert urls[1].endswith("#i4")
    assert urls[2].endswith("#i5")


def test_pruned_images_leave_a_note_the_model_can_act_on() -> None:
    model, requests = local_model(
        [text_response()],
        openrouter_supports_images=True,
        openrouter_max_images=2,
    )

    run(model.query([_image_message(f"i{n}") for n in range(5)]))

    texts = [
        item["text"]
        for message in request_json(requests[0])["messages"]
        for item in message["content"]
        if item.get("type") == "text"
    ]
    assert any("view_image it again" in t for t in texts)


def test_pruning_is_a_no_op_below_the_limit() -> None:
    model, requests = local_model(
        [text_response()],
        openrouter_supports_images=True,
        openrouter_max_images=8,
    )

    run(model.query([_image_message(f"i{n}") for n in range(3)]))

    urls = [
        item["image_url"]["url"]
        for message in request_json(requests[0])["messages"]
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    assert len(urls) == 3
