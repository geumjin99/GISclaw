# GISclaw — an LLM agent for geospatial analysis.
# Copyright (C) 2026 Han Jinzhen
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of GISclaw. GISclaw is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License in the LICENSE file, or
# <https://www.gnu.org/licenses/>, for more details.

"""
LLM engines.

Two backends, one interface: `OpenAIEngine` for OpenAI and every
OpenAI-compatible endpoint (DeepSeek, Gemini's compatibility layer, a model
served on your own machine by Ollama/LM Studio/vLLM, and any custom base_url),
and `ClaudeEngine` for Anthropic. Both expose the same
`load_model()` / `generate()` / `get_stats()` surface, so the agent does not
care which one it is holding.

`generate()` never raises on a failed call: it returns the error as the reply
text, prefixed with "Error during …", and the caller decides what to do with
it (the product turns the first one into an aborted run rather than letting
the loop retry blindly).
"""
import os
import re
import time
from typing import Optional, Dict, Any, List

# Open-weight reasoning models (Qwen3, the R1 distills, and most of what a local
# server hosts) put their scratch work in <think>…</think> ahead of the answer.
# The ReAct loop parses the reply line by line and would read that as the model
# failing to produce an Action, so it never gets that far.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Drop a leading reasoning block, keeping only what follows it."""
    if not text or "think>" not in text.lower():
        return text
    text = _THINK_BLOCK.sub("", text)
    low = text.lower()
    # Some servers emit the closing tag only, having eaten the opening one.
    if "</think>" in low:
        text = text[low.rindex("</think>") + len("</think>"):]
    # An unclosed block means the reply ran out of room mid-thought: there is no
    # answer in it, and passing the raw monologue on would only confuse the loop.
    elif "<think>" in low:
        text = text[:low.index("<think>")]
    return text.strip()


class OpenAIEngine:
    """OpenAI Chat Completions, and anything that speaks the same protocol.

    Point `base_url` at DeepSeek, Gemini's compatibility layer, a local server
    (Ollama, LM Studio, vLLM) or a private endpoint and the rest of the code is
    unchanged. A local server usually wants no credential at all; pass any
    placeholder, since the client refuses to start with an empty one.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = None,
        cost_per_m: tuple = (2.5, 10.0),  # (input_cost, output_cost) per 1M tokens
    ):
        self.model_name = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.cost_per_m = cost_per_m
        self.client = None

        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        # OpenAI and DeepSeek serve repeated prompt prefixes from a cache and
        # bill those tokens at a fraction of the input rate. They are counted
        # apart so the estimate does not overstate what a long run cost.
        self._cache_read_tokens = 0
        # Some models reject a temperature. Once one has, stop sending it.
        self._send_temperature = True

    def load_model(self) -> bool:
        """Create the client. Returns False instead of raising."""
        try:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)
            print(f"OpenAI API initialized. Model: {self.model_name}" + (f" (base_url: {self.base_url})" if self.base_url else ""))
            return True
        except ImportError:
            print("Error: openai not installed. Install with: pip install openai")
            return False
        except Exception as e:
            print(f"Error initializing OpenAI: {e}")
            return False

    def _is_gpt5_model(self) -> bool:
        """GPT-5.x takes a different set of parameters."""
        return 'gpt-5' in self.model_name.lower()

    def _is_gemini_thinking_model(self) -> bool:
        """Gemini's pro line thinks before answering; `stop` empties the reply.

        The flash line does not, and is fine with the ordinary parameters.
        """
        name = self.model_name.lower()
        return 'gemini' in name and 'pro' in name

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One completion.

        Reasoning models need two accommodations: `max_completion_tokens`
        instead of `max_tokens`, and no `stop` — the sequence is trimmed off
        afterwards here instead.
        """
        if self.client is None:
            return {
                "text": "Error: API client not initialized. Call load_model() first.",
                "tokens_generated": 0,
                "latency_ms": 0,
            }

        if stop is None:
            stop = ["Observation:", "\nUser:"]

        start = time.time()
        is_gpt5 = self._is_gpt5_model()
        is_gemini_thinking = self._is_gemini_thinking_model()
        # Neither reasoning family accepts `stop`.
        needs_manual_stop = is_gpt5 or is_gemini_thinking

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_message:
                messages.append({"role": "user", "content": user_message})
            elif prompt:
                messages.append({"role": "user", "content": prompt})

            effective_max_tokens = max_tokens or self.max_tokens
            # Thinking tokens are drawn from the completion budget, so a
            # normal-sized budget can be spent before any text is emitted.
            if is_gpt5 or is_gemini_thinking:
                effective_max_tokens = max(effective_max_tokens, 16384)

            kwargs = {"model": self.model_name, "messages": messages}
            # No `stop` for the reasoning families; the reply is trimmed below.
            if needs_manual_stop:
                kwargs["max_completion_tokens"] = effective_max_tokens
            else:
                kwargs["max_tokens"] = effective_max_tokens
                kwargs["stop"] = stop
            if self._send_temperature:
                kwargs["temperature"] = temperature or self.temperature
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                # Reasoning models accept only their default temperature and
                # answer anything else with a 400. Retry once without it, and
                # remember for the rest of the run.
                if self._send_temperature and "temperature" in str(e).lower():
                    self._send_temperature = False
                    kwargs.pop("temperature", None)
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise

            # A thinking model sometimes comes back with no content at all.
            raw_content = response.choices[0].message.content
            text = strip_reasoning(raw_content.strip()) if raw_content else ""
            usage = response.usage
            output_tokens = usage.completion_tokens if usage else 0
            input_tokens = usage.prompt_tokens if usage else 0
            cached = 0
            if usage:
                # OpenAI: prompt_tokens_details.cached_tokens (inside prompt_tokens).
                # DeepSeek: prompt_cache_hit_tokens (also inside prompt_tokens).
                details = getattr(usage, "prompt_tokens_details", None)
                cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
                cached = cached or (getattr(usage, "prompt_cache_hit_tokens", 0) or 0)

            # Hand the loop something parseable so it asks again rather than
            # crashing on an empty string.
            if not text:
                text = "Thought: (empty response from model)\nAction: \nArgs: {}"

            # Trim at the first stop sequence.
            if needs_manual_stop and stop:
                for stop_seq in stop:
                    idx = text.find(stop_seq)
                    if idx != -1:
                        text = text[:idx].strip()
                        break

            elapsed_ms = (time.time() - start) * 1000

            self._total_calls += 1
            self._total_tokens_generated += output_tokens
            self._total_input_tokens += input_tokens - cached
            self._cache_read_tokens += cached
            self._total_output_tokens += output_tokens
            self._total_time += elapsed_ms

            return {
                "text": text,
                "tokens_generated": output_tokens,
                "latency_ms": elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return {
                "text": f"Error during API call: {str(e)}",
                "tokens_generated": 0,
                "latency_ms": elapsed_ms,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Call counts, token totals and an estimated spend."""
        avg_latency = self._total_time / max(self._total_calls, 1)
        in_cost, out_cost = self.cost_per_m
        # A cache hit is billed at a tenth of the input rate by both OpenAI
        # and DeepSeek; close enough for an estimate.
        estimated_cost = (
            self._total_input_tokens * in_cost / 1_000_000
            + self._cache_read_tokens * in_cost * 0.10 / 1_000_000
            + self._total_output_tokens * out_cost / 1_000_000
        )
        prompt_tokens = self._total_input_tokens + self._cache_read_tokens
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens_generated,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "cache_read_tokens": self._cache_read_tokens,
            "cache_write_tokens": 0,
            "prompt_tokens": prompt_tokens,
            "cache_hit_rate": round(self._cache_read_tokens / prompt_tokens, 3) if prompt_tokens else 0.0,
            "total_time_ms": self._total_time,
            "avg_latency_ms": avg_latency,
            "estimated_cost_usd": round(estimated_cost, 4),
        }

    def reset_stats(self):
        """Zero the counters."""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cache_read_tokens = 0


class ClaudeEngine:
    """Anthropic's Messages API, behind the same interface as OpenAIEngine."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        cost_per_m: tuple = (3.0, 15.0),  # (input, output) per 1M tokens
    ):
        self.model_name = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_per_m = cost_per_m
        self.client = None

        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        # Cached input is billed differently from fresh input, so it has to
        # be counted separately or the cost estimate drifts low.
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        # Tells a caller it may hand the user turn over pre-split so the
        # settled part can be cached. Engines without this keep getting
        # one plain string.
        self.supports_segmented_user = True

    def load_model(self) -> bool:
        """Create the client. Returns False instead of raising."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print(f"Claude API initialized. Model: {self.model_name}")
            return True
        except ImportError:
            print("Error: anthropic not installed. Install with: pip install anthropic")
            return False
        except Exception as e:
            print(f"Error initializing Claude: {e}")
            return False

    def _rejects_sampling_params(self) -> bool:
        """Newer Claude models dropped temperature/top_p/top_k; sending one is a 400."""
        name = self.model_name.lower()
        return any(tag in name for tag in (
            "opus-5", "sonnet-5", "fable-5", "mythos-5", "opus-4-8", "opus-4-7"))

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
        user_segments: Optional[list] = None,
    ) -> Dict[str, Any]:
        """One completion."""
        if self.client is None:
            return {"text": "Error: API client not initialized.", "tokens_generated": 0, "latency_ms": 0}

        if stop is None:
            stop = ["Observation:", "\nUser:"]

        start = time.time()

        try:
            messages = []
            content = user_message or prompt
            if user_segments:
                # The caller split the turn into a settled part and this
                # round's delta. Marking the settled part cacheable lets each
                # round read the history the previous round paid to write,
                # instead of re-buying the whole transcript every time.
                blocks = []
                for seg in user_segments[:-1]:
                    blocks.append({"type": "text", "text": seg,
                                   "cache_control": {"type": "ephemeral"}})
                blocks.append({"type": "text", "text": user_segments[-1]})
                messages.append({"role": "user", "content": blocks})
            elif content:
                messages.append({"role": "user", "content": content})

            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
            }
            # Opus 5 / 4.8 / 4.7, Sonnet 5, Fable 5 removed the sampling params:
            # sending temperature/top_p/top_k returns a 400. Older Claude models
            # still accept them, so keep passing it there.
            if not self._rejects_sampling_params():
                kwargs["temperature"] = temperature or self.temperature
            if system_prompt:
                # Within one ReAct run the system prompt is byte-identical on
                # every round — skills, memory, the project digest and the
                # tool descriptions are fixed once the run starts — yet it is
                # re-sent each time and was charged at full rate. Marking it
                # cacheable lets rounds 2..N read it at a tenth of the price.
                # Below the model's minimum cacheable length it simply does
                # not cache; no error, so this is safe to apply always.
                kwargs["system"] = [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]
            if stop:
                kwargs["stop_sequences"] = stop

            response = self.client.messages.create(**kwargs)

            # Safety classifiers can decline (HTTP 200 + stop_reason "refusal").
            if getattr(response, "stop_reason", None) == "refusal":
                return {"text": "Error: the model declined this request (safety refusal).",
                        "tokens_generated": 0, "latency_ms": (time.time() - start) * 1000}

            # Thinking is on by default on Opus 5, so content[0] may be a thinking
            # block — take the first *text* block rather than assuming index 0.
            text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text = block.text
                    break
            text = text.strip()
            usage = response.usage
            # input_tokens counts only what was NOT served from cache; the
            # whole prompt is input + cache_creation + cache_read.
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

            elapsed_ms = (time.time() - start) * 1000

            self._total_calls += 1
            self._total_tokens_generated += output_tokens
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._cache_read_tokens += cache_read
            self._cache_write_tokens += cache_write
            self._total_time += elapsed_ms

            return {
                "text": text,
                "tokens_generated": output_tokens,
                "latency_ms": elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return {
                "text": f"Error during Claude API call: {str(e)}",
                "tokens_generated": 0,
                "latency_ms": elapsed_ms,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Call counts, token totals and an estimated spend."""
        avg_latency = self._total_time / max(self._total_calls, 1)
        in_cost, out_cost = self.cost_per_m
        # Cached input is not free and not full price: a cache read bills at
        # 0.1x the input rate and writing an entry costs 1.25x. Folding all
        # three buckets into one number would understate what a run actually
        # cost, since the API reports only the uncached remainder as input.
        estimated_cost = (
            self._total_input_tokens * in_cost / 1_000_000
            + self._cache_write_tokens * in_cost * 1.25 / 1_000_000
            + self._cache_read_tokens * in_cost * 0.10 / 1_000_000
            + self._total_output_tokens * out_cost / 1_000_000
        )
        prompt_tokens = (self._total_input_tokens + self._cache_write_tokens
                         + self._cache_read_tokens)
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens_generated,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "cache_read_tokens": self._cache_read_tokens,
            "cache_write_tokens": self._cache_write_tokens,
            "prompt_tokens": prompt_tokens,
            "cache_hit_rate": round(self._cache_read_tokens / prompt_tokens, 3) if prompt_tokens else 0.0,
            "total_time_ms": self._total_time,
            "avg_latency_ms": avg_latency,
            "estimated_cost_usd": round(estimated_cost, 4),
        }

    def reset_stats(self):
        """Zero the counters."""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        # Cached input is billed differently from fresh input, so it has to
        # be counted separately or the cost estimate drifts low.
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0

