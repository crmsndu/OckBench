"""
OpenAI API client implementation.

Supports:
- Official OpenAI API
- OpenAI-compatible APIs (vLLM, SGLang, local serving)
- O1/O3 models with reasoning tokens
"""
import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI

from .base import BaseModelClient
from ..core.schemas import ModelResponse, TokenUsage


logger = logging.getLogger(__name__)


class OpenAIClient(BaseModelClient):
    """
    Client for OpenAI API and OpenAI-compatible endpoints.
    
    Supports standard models (GPT-4, GPT-3.5) and reasoning models (O1, O3).
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name (e.g., 'gpt-5', 'o3')
            api_key: OpenAI API key
            base_url: Base URL for API (for compatible endpoints)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters
        """
        super().__init__(model, api_key, base_url, timeout, max_retries, **kwargs)
        
        # Initialize OpenAI client.
        # - max_retries=0: BaseModelClient.generate owns retry; SDK's own loop
        #   would stack with ours (up to 9 effective attempts) and double the
        #   wall clock on transient errors.
        # - timeout is interpreted per-operation, not as a total deadline:
        #   `read` caps the gap between streamed chunks, so long reasoning
        #   runs can complete while a truly stalled stream still fails.
        client_kwargs = {
            'timeout': httpx.Timeout(connect=30.0, read=float(timeout), write=60.0, pool=30.0),
            'max_retries': 0,
        }
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
            # Local servers may not need a real key
            if 'api_key' not in client_kwargs:
                client_kwargs['api_key'] = 'dummy-key'

        self.client = AsyncOpenAI(**client_kwargs)
    
    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """Call OpenAI API."""
        # Build messages - prompt is already formatted by runner
        messages = [{"role": "user", "content": prompt}]
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Handle reasoning models (gpt-5, o1, o3) vs standard models
        if self._is_reasoning_model():
            # Reasoning models use max_completion_tokens and reasoning_effort
            request_params["max_completion_tokens"] = max_output_tokens
            
            if kwargs.get("reasoning_effort"):
                request_params["reasoning_effort"] = kwargs["reasoning_effort"]
            
            # Reasoning models (gpt-5, o1, o3) don't support temperature, top_p, top_k, 
            # repeat_penalty and other sampling parameters
            # Remove these parameters for all reasoning models
            request_params.pop("temperature", None)
            request_params.pop("top_p", None)
            request_params.pop("top_k", None)
            request_params.pop("repeat_penalty", None)
            # Remove any other sampling-related parameters that might be in kwargs
            for param in ["top_p", "top_k", "repeat_penalty", "frequency_penalty", "presence_penalty"]:
                kwargs.pop(param, None)
        else:
            # Standard models use max_tokens
            request_params["max_tokens"] = max_output_tokens

            # Add optional parameters for standard models
            if kwargs.get("top_p") is not None:
                request_params["top_p"] = kwargs["top_p"]
            if kwargs.get("top_k") is not None:
                request_params["top_k"] = kwargs["top_k"]
            if kwargs.get("frequency_penalty") is not None:
                request_params["frequency_penalty"] = kwargs["frequency_penalty"]
            if kwargs.get("presence_penalty") is not None:
                request_params["presence_penalty"] = kwargs["presence_penalty"]

        # Handle enable_thinking for models that support thinking mode (via extra_body)
        # Qwen3 uses {"enable_thinking": bool}, DeepSeek uses {"thinking": bool}
        if kwargs.get("enable_thinking") is not None:
            if "deepseek" in self.model.lower():
                thinking_key = "thinking"
            else:
                thinking_key = "enable_thinking"
            request_params["extra_body"] = {
                "chat_template_kwargs": {thinking_key: kwargs["enable_thinking"]}
            }

        # Handle OpenRouter reasoning parameter for models like DeepSeek
        # OpenRouter uses {"reasoning": {"enabled": true}} instead of extra_body
        if "openrouter" in (self.base_url or ""):
            if kwargs.get("enable_thinking"):
                request_params["extra_body"] = {"reasoning": {"enabled": True}}
            elif kwargs.get("reasoning_effort") and not self._is_reasoning_model():
                request_params["extra_body"] = {"reasoning": {"effort": kwargs["reasoning_effort"]}}

        # Make API call with streaming to avoid proxy gateway timeouts
        try:
            request_params["stream"] = True
            request_params["stream_options"] = {"include_usage": True}

            text = ""
            finish_reason = None
            model_name = self.model
            usage_data = None

            stream = await self.client.chat.completions.create(**request_params)
            async for chunk in stream:
                if chunk.model:
                    model_name = chunk.model

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        text += delta.content
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                # Usage comes in the final chunk when stream_options.include_usage is set
                if chunk.usage:
                    usage_data = chunk

            if usage_data:
                tokens = self._extract_tokens(usage_data)
            else:
                tokens = TokenUsage(
                    prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                    output_tokens=0, total_tokens=0,
                )

            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,  # Will be set by base class
                model=model_name,
                finish_reason=finish_reason or "stop",
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _extract_tokens(self, response) -> TokenUsage:
        """
        Extract token usage from OpenAI response.
        
        Args:
            response: OpenAI API response object
        
        Returns:
            TokenUsage: Token usage information
        """
        usage = response.usage
        # Standard token fields
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        
        # Reasoning tokens (for o1/o3/gpt-5 models)
        # These models may have completion_tokens_details with reasoning_tokens
        reasoning_tokens = 0
        if hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            if details and hasattr(details, 'reasoning_tokens'):
                reasoning_tokens = details.reasoning_tokens or 0
        
        # For reasoning models, completion_tokens is the TOTAL output (includes reasoning)
        # So answer_tokens = completion_tokens - reasoning_tokens
        # For non-reasoning models, reasoning_tokens is 0, so answer_tokens = completion_tokens
        answer_tokens = completion_tokens - reasoning_tokens
        
        # output_tokens = reasoning_tokens + answer_tokens = completion_tokens
        output_tokens = completion_tokens
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            total_tokens=getattr(usage, 'total_tokens', 0)
        )
    
    def _is_reasoning_model(self) -> bool:
        """
        Check if model is a reasoning model (o1, o3, gpt-5).
        
        Returns:
            bool: True if reasoning model
        """
        model_lower = self.model.lower()
        return any(prefix in model_lower for prefix in ['o1', 'o3', 'o4', 'gpt-5'])

