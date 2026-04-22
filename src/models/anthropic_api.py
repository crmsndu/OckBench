"""Anthropic Messages API client with streaming and adaptive thinking."""
import json
import logging
from typing import Optional

import httpx

from ..core.schemas import ModelResponse, TokenUsage
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseModelClient):
    """Client for Anthropic /v1/messages with adaptive thinking and streaming."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(model, api_key, base_url, timeout, max_retries, **kwargs)

        self.endpoint = (base_url or "https://api.anthropic.com").rstrip("/")
        if self.endpoint.endswith("/v1"):
            self.messages_url = self.endpoint + "/messages"
        else:
            self.messages_url = self.endpoint + "/v1/messages"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'dummy-key'}",
            "x-api-key": api_key or "dummy-key",
            "anthropic-version": "2023-06-01",
        }

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0, read=timeout),
        )

    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        request_body = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "stream": True,
        }

        try:
            text = ""
            input_tokens = 0
            output_tokens = 0
            model_name = self.model
            stop_reason = "end_turn"
            buffer = ""

            async with self._http_client.stream(
                "POST", self.messages_url, headers=self.headers, json=request_body
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )

                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            d = json.loads(data_str)
                            event_type = d.get("type")

                            if event_type == "message_start":
                                msg = d.get("message", {})
                                usage = msg.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                model_name = msg.get("model", self.model)

                            elif event_type == "content_block_delta":
                                delta = d.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text += delta.get("text", "")

                            elif event_type == "message_delta":
                                usage = d.get("usage", {})
                                output_tokens = usage.get("output_tokens", output_tokens)
                                stop_reason = d.get("delta", {}).get("stop_reason", stop_reason)

                        except json.JSONDecodeError:
                            continue

            tokens = TokenUsage(
                prompt_tokens=input_tokens,
                answer_tokens=output_tokens,
                reasoning_tokens=0,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,
                model=model_name,
                finish_reason=stop_reason,
            )

        except httpx.HTTPStatusError as e:
            error_body = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"Anthropic API HTTP error: {e.response.status_code} - {error_body}")
            raise Exception(f"Error code: {e.response.status_code} - {error_body}")
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
