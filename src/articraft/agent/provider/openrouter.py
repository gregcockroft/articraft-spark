from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from articraft.errors import ModelError
from articraft.settings import Settings, get_settings

_OPENROUTER_HOST = "openrouter.ai"
_TOOL_IMAGE_TEXT = "Image returned by the preceding tool call."
_DROPPED_IMAGE_TEXT = (
    "[an earlier image was dropped from this conversation to stay within the "
    "endpoint's per-prompt image limit; view_image it again if you still need it]"
)
_RETRY_BASE_SECONDS = 0.5
_RETRY_MAX_SECONDS = 20.0
# Qwen's thinking-budget recipe: append a closing sentence to the truncated reasoning,
# close the think block, and the model answers from the thinking it has. Qwen's own
# wording ends "give the solution based on the thinking directly now"; this one points
# at the tools, which is what an agent turn has to produce (6/6 tool calls against 5/6
# in a small live comparison on Qwen3.6).
_THINKING_CLOSE = (
    "Considering the limited time by the user, I have to stop thinking and act on this "
    "plan now, using the tools directly."
)
logger = logging.getLogger(__name__)


class OpenRouterModel:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = settings or get_settings()
        self._api_url = _chat_completions_url(self.config.openrouter_base_url)
        # A self-hosted OpenAI-compatible server (vLLM, llama.cpp, Ollama) needs no
        # credential; only openrouter.ai itself does.
        if requires_api_key(self.config) and not (self.config.openrouter_api_key or "").strip():
            raise ModelError("OpenRouter credentials are required. Set OPENROUTER_API_KEY.")
        if not self.config.openrouter_model.strip():
            raise ModelError(
                "OpenRouter model is required. Pass --model or set ARTICRAFT_OPENROUTER_MODEL."
            )
        # openrouter.ai routes to text-only models by default; a local vision server
        # opts in with ARTICRAFT_OPENROUTER_IMAGES=1.
        self.supports_images = bool(self.config.openrouter_supports_images)
        self._client = client

    @property
    def context_window_tokens(self) -> int:
        """Return the configured window because OpenRouter has no local model catalog."""
        return self.config.openrouter_context_window_tokens

    async def query(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Query OpenRouter and return the response shape used by the agent."""
        request: dict[str, Any] = {
            "model": self.config.openrouter_model,
            "messages": _messages(
                messages,
                images=self.supports_images,
                max_images=self.config.openrouter_max_images,
            ),
        }
        # A hosted model bounds its own reply. A self-hosted server does not, so an
        # uncapped turn can generate until the context ceiling; 0 keeps the old
        # unbounded behaviour.
        if self.config.openrouter_max_output_tokens:
            request["max_tokens"] = self.config.openrouter_max_output_tokens
        template_kwargs = _chat_template_kwargs(self.config)
        if template_kwargs:
            request["chat_template_kwargs"] = template_kwargs
        converted_tools = _tools(tools or [])
        if converted_tools:
            request["tools"] = converted_tools
        if self.config.openrouter_thinking_budget_tokens:
            return await self._query_with_thinking_budget(request)

        response = await self._send_with_retries(request)
        payload = _response_payload(response)
        _raise_for_provider_error(response.status_code, payload)
        text, tool_calls, provider_content = _assistant_output(payload)
        if not text and not tool_calls and not provider_content:
            raise ModelError("OpenRouter response did not contain text or tool calls")
        # A reasoning-only turn is a real thing for a model served with a reasoning
        # parser: the visible content is empty while `reasoning` holds the thinking.
        # That is an empty turn, not a transport error, so it is returned as one and
        # the agent loop's own empty-response handling gets to nudge the model. The
        # reasoning is NOT promoted into `text`: the loop treats a turn with text and
        # a compiled workspace as the final answer, and thinking is not an answer.

        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": _response_token_usage(payload),
            "cost": _response_cost(payload),
            "provider_content": provider_content,
            "response": payload,
        }

    async def _query_with_thinking_budget(self, request: dict[str, Any]) -> dict[str, Any]:
        """Bound one turn's reasoning, then make the model act on what it has.

        A served reasoning model can deliberate to the output cap with no tool call --
        Qwen3.6 did so three turns running at 32,768 tokens each. vLLM's own
        `thinking_token_budget` is accepted and ignored on that server, so the bound is
        enforced here, the way Qwen's thinking-budget recipe does it: stream the turn,
        and once reasoning alone has spent the budget, close the stream (which aborts
        generation) and resend the conversation with that reasoning as an assistant
        prefix ending in a closing sentence, so the answer comes from the thinking it
        has. A turn that starts acting within the budget streams to completion unchanged.
        """
        budget = self.config.openrouter_thinking_budget_tokens
        turn = await self._stream_until_budget(request, budget)
        if not turn.budget_hit:
            return turn.result()

        prefix = turn.reasoning.rstrip() + "\n\n" + _THINKING_CLOSE
        continuation = dict(request)
        continuation["messages"] = [
            *request["messages"],
            {"role": "assistant", "content": "", "reasoning": prefix},
        ]
        continuation["continue_final_message"] = True
        continuation["add_generation_prompt"] = False
        # The prefix already closed the think block. The reasoning parser labels output
        # without a closing tag as truncated reasoning unless thinking is declared off.
        continuation["chat_template_kwargs"] = {
            **request.get("chat_template_kwargs", {}),
            "enable_thinking": False,
        }
        response = await self._send_with_retries(continuation)
        payload = _response_payload(response)
        _raise_for_provider_error(response.status_code, payload)
        text, tool_calls, provider_content = _assistant_output(payload)
        reasoning_item = {
            "type": "openrouter_reasoning",
            "reasoning": prefix,
            "thinking_budget_tokens": budget,
        }
        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": _add_usage(turn.token_usage(), _response_token_usage(payload)),
            "cost": _response_cost(payload),
            "provider_content": [reasoning_item, *provider_content],
            "response": payload,
        }

    async def _stream_until_budget(self, request: dict[str, Any], budget: int) -> _StreamedTurn:
        streaming = {
            **request,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        turn = _StreamedTurn()
        for attempt in range(1, self.config.openrouter_max_attempts + 1):
            try:
                async with self._client_or_create().stream(
                    "POST",
                    self._api_url,
                    headers=self._headers(),
                    json=streaming,
                    timeout=self.config.openrouter_request_timeout_seconds,
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        if _retryable_status(response.status_code) and (
                            attempt < self.config.openrouter_max_attempts
                        ):
                            raise _Retry(f"HTTP {response.status_code}")
                        raise ModelError(
                            _provider_error(response.status_code, _error_payload(response))
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        turn.take(json.loads(data))
                        if not turn.acting and turn.completion_tokens >= budget:
                            turn.budget_hit = True
                            # Leaving the block closes the response, and the server
                            # aborts the generation on the disconnect.
                            break
                return turn
            except _Retry as exc:
                reason = str(exc)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.config.openrouter_max_attempts or turn.completion_tokens:
                    raise ModelError(
                        f"OpenRouter request failed: {_format_exception(exc)}"
                    ) from exc
                reason = _format_exception(exc)
            delay = _retry_delay(attempt)
            logger.warning(
                "OpenRouter stream failed (attempt %s/%s), retrying in %.2fs: %s",
                attempt,
                self.config.openrouter_max_attempts,
                delay,
                reason,
            )
            await asyncio.sleep(delay)
        raise AssertionError("retry loop did not return or raise")

    async def summarize_context(
        self,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        """Create a plain checkpoint with one completion call and no tools."""
        request: dict[str, Any] = {
            "model": self.config.openrouter_model,
            "messages": _messages(
                messages,
                images=self.supports_images,
                max_images=self.config.openrouter_max_images,
            ),
            "max_tokens": min(
                max_output_tokens,
                self.config.openrouter_summary_max_output_tokens,
            ),
        }
        # The same chat-template arguments the generating turns get. Without them this one call goes
        # out at the template's own defaults - for Qwen3.8 that is `reasoning_effort: xhigh` - so the
        # run sends one model two configurations, and the unconfigured one is the call most likely to
        # be spent entirely on reasoning and come back with no text. An empty summary is fatal
        # (agent/harness.py re-checks it and ends the run), so a 90-minute run can be lost to a
        # summary that thought instead of answering.
        template_kwargs = _chat_template_kwargs(self.config)
        if template_kwargs:
            request["chat_template_kwargs"] = template_kwargs
        response = await self._send_with_retries(request)
        payload = _response_payload(response)
        _raise_for_provider_error(response.status_code, payload)
        text, _, _ = _assistant_output(payload)
        if not text:
            raise ModelError("OpenRouter summary response did not contain text")
        return {
            "text": text,
            "token_usage": _response_token_usage(payload),
            "cost": _response_cost(payload),
        }

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def _send_with_retries(self, request: dict[str, Any]) -> httpx.Response:
        for attempt in range(1, self.config.openrouter_max_attempts + 1):
            response: httpx.Response | None = None
            try:
                response = await self._client_or_create().post(
                    self._api_url,
                    headers=self._headers(),
                    json=request,
                    timeout=self.config.openrouter_request_timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.config.openrouter_max_attempts:
                    raise ModelError(
                        f"OpenRouter request failed: {_format_exception(exc)}"
                    ) from exc
                delay = _retry_delay(attempt)
                logger.warning(
                    "OpenRouter request failed (attempt %s/%s), retrying in %.2fs: %s",
                    attempt,
                    self.config.openrouter_max_attempts,
                    delay,
                    _format_exception(exc),
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code < 400:
                payload = _response_payload(response)
                provider_status = _embedded_provider_status(payload)
                if (
                    provider_status is not None
                    and _retryable_status(provider_status)
                    and attempt < self.config.openrouter_max_attempts
                ):
                    delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                    logger.warning(
                        "OpenRouter provider failed (attempt %s/%s), retrying in %.2fs: "
                        "provider code %s",
                        attempt,
                        self.config.openrouter_max_attempts,
                        delay,
                        provider_status,
                    )
                    await asyncio.sleep(delay)
                    continue
                return response

            if _retryable_status(response.status_code) and (
                attempt < self.config.openrouter_max_attempts
            ):
                delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "OpenRouter request failed (attempt %s/%s), retrying in %.2fs: HTTP %s",
                    attempt,
                    self.config.openrouter_max_attempts,
                    delay,
                    response.status_code,
                )
                await asyncio.sleep(delay)
                continue

            payload = _error_payload(response)
            raise ModelError(_provider_error(response.status_code, payload))

        raise AssertionError("retry loop did not return or raise")

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = (self.config.openrouter_api_key or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        referer = (self.config.openrouter_http_referer or "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        title = (self.config.openrouter_app_title or "").strip()
        if title:
            headers["X-OpenRouter-Title"] = title
        return headers


class _Retry(Exception):
    """A retryable HTTP status on the streaming request."""


@dataclass
class _StreamedTurn:
    """One turn assembled from chat-completions stream chunks."""

    reasoning_parts: list[str] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    acting: bool = False
    budget_hit: bool = False

    def take(self, chunk: dict[str, Any]) -> None:
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            piece = delta.get("reasoning") or delta.get("reasoning_content")
            if isinstance(piece, str) and piece:
                self.reasoning_parts.append(piece)
            if isinstance(delta.get("content"), str) and delta["content"]:
                self.text_parts.append(delta["content"])
                self.acting = True
            for raw_call in delta.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                self.acting = True
                call = self.calls.setdefault(
                    _int(raw_call.get("index")), {"id": "", "name": "", "arguments": []}
                )
                if raw_call.get("id"):
                    call["id"] = str(raw_call["id"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    call["name"] = str(function["name"])
                if function.get("arguments"):
                    call["arguments"].append(str(function["arguments"]))
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)

    @property
    def completion_tokens(self) -> int:
        return _int(self.usage.get("completion_tokens"))

    def token_usage(self) -> dict[str, int]:
        return _response_token_usage({"usage": self.usage})

    def result(self) -> dict[str, Any]:
        text = "".join(self.text_parts)
        tool_calls = []
        for index in sorted(self.calls):
            call = self.calls[index]
            if not call["id"] or not call["name"]:
                raise ModelError("OpenRouter returned a function call without an id or name")
            tool_calls.append(
                {"id": call["id"], "name": call["name"], "arguments": "".join(call["arguments"])}
            )
        provider_content: list[dict[str, Any]] = []
        if self.reasoning:
            provider_content.append({"type": "openrouter_reasoning", "reasoning": self.reasoning})
        if not text and not tool_calls and not provider_content:
            raise ModelError("OpenRouter response did not contain text or tool calls")
        return {
            "text": text,
            "tool_calls": tool_calls,
            "token_usage": self.token_usage(),
            "cost": 0.0,
            "provider_content": provider_content,
            "response": {
                "choices": [{"message": {"content": text}, "finish_reason": self.finish_reason}],
                "usage": self.usage,
            },
        }


def _add_usage(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
    return {key: first.get(key, 0) + second.get(key, 0) for key in first.keys() | second.keys()}


def _messages(
    messages: list[dict[str, Any]],
    *,
    images: bool = False,
    max_images: int = 0,
) -> list[dict[str, Any]]:
    converted = _convert_messages(messages, images=images)
    if max_images:
        _prune_images(converted, max_images)
    return converted


def _prune_images(converted: list[dict[str, Any]], keep: int) -> None:
    """Drop the middle of the image history, keeping the first and the most recent.

    Images accumulate and are never removed, so any server-side per-prompt cap is
    eventually exhausted and the request fails outright, killing the run. The FIRST
    image is the reference the run is reconstructing and is the last thing worth
    dropping, so it is kept and the oldest *subsequent* images go first.
    """
    positions = [
        (index, item_index)
        for index, message in enumerate(converted)
        if isinstance(message.get("content"), list)
        for item_index, item in enumerate(message["content"])
        if isinstance(item, dict) and item.get("type") == "image_url"
    ]
    if len(positions) <= keep:
        return

    survivors = (
        positions[:1] + positions[len(positions) - (keep - 1) :] if keep > 1 else positions[:1]
    )
    for index, item_index in positions:
        if (index, item_index) in survivors:
            continue
        converted[index]["content"][item_index] = {
            "type": "text",
            "text": _DROPPED_IMAGE_TEXT,
        }


def _convert_messages(messages: list[dict[str, Any]], *, images: bool) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("type") == "function_call_output":
            text, image_parts = _split_content(message.get("output"), images=images)
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("call_id") or ""),
                    "content": text,
                }
            )
            # A chat-completions tool message carries text only, so an image a tool
            # returned rides in a user message immediately after it.
            if image_parts:
                converted.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": _TOOL_IMAGE_TEXT}, *image_parts],
                    }
                )
            continue

        role = message.get("role")
        if role in {"system", "user"}:
            text, image_parts = _split_content(message.get("content"), images=images)
            if image_parts:
                parts: list[dict[str, Any]] = []
                if text:
                    parts.append({"type": "text", "text": text})
                parts.extend(image_parts)
                converted.append({"role": role, "content": parts})
            else:
                converted.append({"role": role, "content": text})
        elif role == "assistant":
            converted.append(_assistant_message(message))
    return converted


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    text = _text_content(message.get("content"))
    converted: dict[str, Any] = {
        "role": "assistant",
        "content": text or None,
    }
    tool_calls = [
        {
            "id": str(call.get("id") or ""),
            "type": "function",
            "function": {
                "name": str(call.get("name") or ""),
                "arguments": _arguments_text(call.get("arguments")),
            },
        }
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    ]
    if tool_calls:
        converted["tool_calls"] = tool_calls
    for item in message.get("provider_content") or []:
        if not isinstance(item, dict) or item.get("type") != "openrouter_reasoning":
            continue
        reasoning = item.get("reasoning")
        if isinstance(reasoning, str):
            converted["reasoning"] = reasoning
        reasoning_details = item.get("reasoning_details")
        if isinstance(reasoning_details, list):
            converted["reasoning_details"] = reasoning_details
    return converted


def _text_content(content: Any) -> str:
    text, _ = _split_content(content, images=False)
    return text


def _split_content(content: Any, *, images: bool) -> tuple[str, list[dict[str, Any]]]:
    """Split articraft content into chat-completions text and image_url parts."""
    if isinstance(content, str):
        return content, []
    if content is None:
        return "", []
    if not isinstance(content, list):
        return json.dumps(content), []

    text: list[str] = []
    image_parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "input_image":
            if not images:
                raise ModelError(
                    "OpenRouterModel is configured for text input and function calling, not "
                    "images. Set ARTICRAFT_OPENROUTER_IMAGES=1 for a vision-capable endpoint."
                )
            image_parts.append(_image_part(item))
        elif kind == "input_text":
            text.append(str(item.get("text") or ""))
    return "\n".join(text), image_parts


def _image_part(item: dict[str, Any]) -> dict[str, Any]:
    url = str(item.get("image_url") or "")
    if not url:
        raise ModelError("OpenRouter image input requires an image_url")
    image_url: dict[str, Any] = {"url": url}
    # articraft's "original" is not a chat-completions detail; send only what the API takes.
    detail = item.get("detail")
    if detail in {"low", "high", "auto"}:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _arguments_text(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments)


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "parameters": tool.get("parameters") or {},
                "strict": bool(tool.get("strict", False)),
            },
        }
        for tool in tools
        if tool.get("type") == "function"
    ]


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelError(
            f"OpenRouter response was not valid JSON (HTTP {response.status_code})"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelError(f"OpenRouter response was not an object (HTTP {response.status_code})")
    return payload


def _error_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {"error": {"message": response.text.strip() or "request failed"}}
    if isinstance(payload, dict):
        return payload
    return {"error": {"message": response.text.strip() or "request failed"}}


def _raise_for_provider_error(status: int, payload: dict[str, Any]) -> None:
    error = payload.get("error")
    if isinstance(error, dict):
        raise ModelError(_provider_error(status, payload))

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelError(f"OpenRouter response did not contain choices (HTTP {status})")
    first = choices[0]
    if not isinstance(first, dict):
        raise ModelError(f"OpenRouter response contained an invalid choice (HTTP {status})")
    if isinstance(first.get("error"), dict) or first.get("finish_reason") == "error":
        raise ModelError(_provider_error(status, first))


def _embedded_provider_status(payload: dict[str, Any]) -> int | None:
    error = payload.get("error")
    if isinstance(error, dict):
        return _status_code(error.get("code"))

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    choice_error = choices[0].get("error")
    if isinstance(choice_error, dict):
        return _status_code(choice_error.get("code"))
    return None


def _status_code(value: Any) -> int | None:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if status > 0 else None


def _provider_error(status: int, payload: dict[str, Any]) -> str:
    error = payload.get("error")
    message = ""
    code: Any = None
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = error.get("code")
    elif isinstance(error, str):
        message = error.strip()

    detail = message or "request failed"
    code_detail = f", provider code {code}" if code not in {None, status} else ""
    return f"OpenRouter request failed (HTTP {status}{code_detail}): {detail}"


def _assistant_output(
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelError("OpenRouter response did not contain a valid choice")
    first = choices[0]
    message = first.get("message")
    if not isinstance(message, dict):
        raise ModelError("OpenRouter response choice did not contain an assistant message")

    content = message.get("content")
    text = content if isinstance(content, str) else ""
    calls: list[dict[str, Any]] = []
    for raw_call in message.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        call_id = str(raw_call.get("id") or "")
        name = str(function.get("name") or "")
        if not call_id or not name:
            raise ModelError("OpenRouter returned a function call without an id or name")
        calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": _arguments_text(function.get("arguments")),
            }
        )
    provider_content: list[dict[str, Any]] = []
    reasoning: dict[str, Any] = {"type": "openrouter_reasoning"}
    raw_reasoning = message.get("reasoning")
    if isinstance(raw_reasoning, str):
        reasoning["reasoning"] = raw_reasoning
    reasoning_details = message.get("reasoning_details")
    if isinstance(reasoning_details, list):
        reasoning["reasoning_details"] = reasoning_details
    if len(reasoning) > 1:
        provider_content.append(reasoning)
    return text, calls, provider_content


def _response_token_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}

    input_tokens = _int(usage.get("prompt_tokens"))
    output_tokens = _int(usage.get("completion_tokens"))
    total_tokens = _int(usage.get("total_tokens")) or input_tokens + output_tokens
    details = usage.get("prompt_tokens_details")
    cached_input_tokens = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _response_cost(payload: dict[str, Any]) -> float:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    try:
        return float(usage.get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _retryable_status(status: int) -> bool:
    return status in {408, 429} or status >= 500


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                now = datetime.now(retry_at.tzinfo or UTC)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                delay = (retry_at - now).total_seconds()
                return max(0.0, delay)
            except (OverflowError, TypeError, ValueError):
                pass
    return min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2 ** (attempt - 1)))


def _chat_completions_url(base_url: str) -> str:
    """Build the chat-completions endpoint from a base URL such as http://localhost:8001/v1."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ModelError(
            "OpenRouter base URL is required. Set ARTICRAFT_OPENROUTER_BASE_URL or leave it unset."
        )
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _chat_template_kwargs(settings: Settings) -> dict[str, Any]:
    """Parse the server-side chat-template arguments, if any were configured.

    A self-hosted server applies the model's own chat template, and some templates
    take arguments the OpenAI schema has no field for -- Qwen3's `enable_thinking`
    being the one that matters here. vLLM reads them from `chat_template_kwargs`.
    """
    raw = (settings.openrouter_chat_template_kwargs or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ModelError(
            "ARTICRAFT_OPENROUTER_CHAT_TEMPLATE_KWARGS must be a JSON object: "
            f"{_format_exception(exc)}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelError("ARTICRAFT_OPENROUTER_CHAT_TEMPLATE_KWARGS must be a JSON object")
    return parsed


def requires_api_key(settings: Settings) -> bool:
    """Report whether the configured endpoint needs a credential.

    openrouter.ai always does. A self-hosted OpenAI-compatible server (vLLM,
    llama.cpp, Ollama) does not, so a run must not be blocked for want of a key.
    """
    try:
        url = _chat_completions_url(settings.openrouter_base_url)
    except ModelError:
        return True
    return _is_openrouter_host(url)


def _is_openrouter_host(url: str) -> bool:
    return (urlparse(url).hostname or "").lower().endswith(_OPENROUTER_HOST)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message or repr(exc)}"


__all__ = ["OpenRouterModel", "requires_api_key"]
