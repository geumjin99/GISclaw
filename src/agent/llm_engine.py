"""
LLM (LLM Engine)

Wraps llama-cpp-python so that all backends share a uniform LLM inference interface.
llama.cpp 
"""
import os
import time
from typing import Optional, Dict, Any, List

class LLMEngine:
    """LLM llama-cpp-python """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        verbose: bool = False,
        repeat_penalty: float = 1.0,
    ):
        """
        Initialize LLM engine

        Args:
            model_path: GGUF 
            n_ctx: 
            n_gpu_layers: GPU -1 = 
            temperature: 
            max_tokens: token 
            verbose: 
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.repeat_penalty = repeat_penalty
        self.model = None

        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0

    def load_model(self):
        """Load the model into memory / GPU"""
        try:
            from llama_cpp import Llama

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            print(f"Loading model from {self.model_path}...")
            start = time.time()

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )

            elapsed = time.time() - start
            print(f"Model loaded in {elapsed:.1f}s")
            return True

        except ImportError:
            print(
                "Error: llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate text (supports chat-completion mode)

        Use chat completion when both system_prompt and user_message are supplied
        (recommended for instruct-tuned models); otherwise falls back to text completion.

        Args:
            prompt: promptfallback 
            stop: 
            temperature: 
            max_tokens: token 
            system_prompt: chat 
            user_message: chat 

        Returns:
            A dict containing text, tokens_generated, latency_ms
        """
        if self.model is None:
            return {
                "text": "Error: Model not loaded. Call load_model() first.",
                "tokens_generated": 0,
                "latency_ms": 0,
            }

        if stop is None:
            stop = ["Observation:", "\nUser:"]

        start = time.time()

        try:
            # If both system_prompt and user_message are supplied, use chat mode
            if system_prompt is not None and user_message is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
                output = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    stop=stop,
                    repeat_penalty=self.repeat_penalty,
                )
                text = output["choices"][0]["message"]["content"].strip()
                tokens = output["usage"]["completion_tokens"]
            else:
                # Fallback: text completion 
                output = self.model(
                    prompt,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    stop=stop,
                    echo=False,
                    repeat_penalty=self.repeat_penalty,
                )
                text = output["choices"][0]["text"].strip()
                tokens = output["usage"]["completion_tokens"]

            elapsed_ms = (time.time() - start) * 1000

            # Update statistics
            self._total_calls += 1
            self._total_tokens_generated += tokens
            self._total_time += elapsed_ms

            return {
                "text": text,
                "tokens_generated": tokens,
                "latency_ms": elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return {
                "text": f"Error during generation: {str(e)}",
                "tokens_generated": 0,
                "latency_ms": elapsed_ms,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Return inference performance statistics"""
        avg_latency = self._total_time / max(self._total_calls, 1)
        avg_tokens = self._total_tokens_generated / max(self._total_calls, 1)
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens_generated,
            "total_time_ms": self._total_time,
            "avg_latency_ms": avg_latency,
            "avg_tokens_per_call": avg_tokens,
            "tokens_per_second": (
                self._total_tokens_generated / (self._total_time / 1000)
                if self._total_time > 0 else 0
            ),
        }

    def reset_stats(self):
        """"""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0

class MockLLMEngine:
    """
    Mock LLM engine used for tests and demos.

    When no real model is available, predefined responses simulate the agent.
    """

    def __init__(self):
        self._total_calls = 0
        self._responses = []

    def load_model(self) -> bool:
        print("MockLLMEngine: Using mock responses (no actual model)")
        return True

    def set_responses(self, responses: List[str]):
        """Set the predefined response sequence"""
        self._responses = responses

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return one of the predefined responses"""
        if self._responses:
            text = self._responses.pop(0)
        else:
            text = (
                "Thought: I don't have any more predefined responses.\n"
                "Action: Final Answer\n"
                'Action Input: This is a mock response. Please provide a real model.'
            )

        self._total_calls += 1
        return {
            "text": text,
            "tokens_generated": len(text.split()),
            "latency_ms": 10.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {"total_calls": self._total_calls, "mock": True}

    def reset_stats(self):
        self._total_calls = 0

class OpenAIEngine:
    """
    OpenAI API OpenAI Chat Completion 

     LLMEngine GPT-4o 
     base_url DeepSeek OpenAI API
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

        # LLMEngine 
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def load_model(self) -> bool:
        """Initialize OpenAI client"""
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
        """Detect GPT-5.x models (parameter set differs from earlier OpenAI models)"""
        return 'gpt-5' in self.model_name.lower()

    def _is_gemini_thinking_model(self) -> bool:
        """Detect Gemini thinking models (the stop parameter causes empty responses)"""
        name = self.model_name.lower()
        # gemini-3.1-pro / gemini-3-pro pro thinking model
        # gemini-3-flash thinking model
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
        """
        Call the OpenAI API and return generated text

        Interface is fully compatible with LLMEngine.generate().
        GPT-5.x : max_completion_tokens replacement max_tokens, stop 
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
        # Gemini thinking model GPT-5.x stop 
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
            # (GPT-5.x / Gemini Pro): thinking tokens completion 
            # token thinking + 
            if is_gpt5 or is_gemini_thinking:
                effective_max_tokens = max(effective_max_tokens, 16384)

            # GPT-5.x / Gemini thinking: stop 
            if needs_manual_stop:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=effective_max_tokens,
                    temperature=temperature or self.temperature,
                    # stop — 
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=temperature or self.temperature,
                    stop=stop,
                )

            # Guard against content=None (Gemini thinking models sometimes return empty content)
            raw_content = response.choices[0].message.content
            text = raw_content.strip() if raw_content else ""
            output_tokens = response.usage.completion_tokens if response.usage else 0
            input_tokens = response.usage.prompt_tokens if response.usage else 0

            # Empty-response guard: return a format-error hint so the agent retries
            if not text:
                text = "Thought: (empty response from model)\nAction: \nArgs: {}"

            # stop 
            if needs_manual_stop and stop:
                for stop_seq in stop:
                    idx = text.find(stop_seq)
                    if idx != -1:
                        text = text[:idx].strip()
                        break

            elapsed_ms = (time.time() - start) * 1000

            # Update statistics
            self._total_calls += 1
            self._total_tokens_generated += output_tokens
            self._total_input_tokens += input_tokens
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
        """Return inference performance statistics, with cost estimate"""
        avg_latency = self._total_time / max(self._total_calls, 1)
        in_cost, out_cost = self.cost_per_m
        estimated_cost = (
            self._total_input_tokens * in_cost / 1_000_000
            + self._total_output_tokens * out_cost / 1_000_000
        )
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens_generated,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_time_ms": self._total_time,
            "avg_latency_ms": avg_latency,
            "estimated_cost_usd": round(estimated_cost, 4),
        }

    def reset_stats(self):
        """"""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

class OllamaEngine:
    """
    Ollama Ollama API 

    Ollama OpenAI APIhttp://localhost:11434/v1
     openai 
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0

    def load_model(self) -> bool:
        """Initialize Ollama client via OpenAI-compatible endpoint"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",
            )
            print(f"Ollama engine initialized. Model: {self.model_name}")
            return True
        except ImportError:
            print("Error: openai not installed. Install with: pip install openai")
            return False
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            return False

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call the Ollama API and return generated text

        Interface is fully compatible with LLMEngine.generate().
        """
        if self.client is None:
            return {
                "text": "Error: Ollama client not initialized.",
                "tokens_generated": 0,
                "latency_ms": 0,
            }

        if stop is None:
            stop = ["Observation:", "\nUser:"]

        start = time.time()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_message:
                messages.append({"role": "user", "content": user_message})
            elif prompt:
                messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                stop=stop,
            )

            text = response.choices[0].message.content.strip()
            tokens = response.usage.completion_tokens if response.usage else len(text.split())

            elapsed_ms = (time.time() - start) * 1000

            self._total_calls += 1
            self._total_tokens_generated += tokens
            self._total_time += elapsed_ms

            return {
                "text": text,
                "tokens_generated": tokens,
                "latency_ms": elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return {
                "text": f"Error during Ollama call: {str(e)}",
                "tokens_generated": 0,
                "latency_ms": elapsed_ms,
            }

    def get_stats(self) -> Dict[str, Any]:
        avg_latency = self._total_time / max(self._total_calls, 1)
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens_generated,
            "total_time_ms": self._total_time,
            "avg_latency_ms": avg_latency,
        }

    def reset_stats(self):
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0

class ClaudeEngine:
    """
    Anthropic Claude API 

    Interface is OpenAIEngine-compatible and can be substituted directly.
    """

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

    def load_model(self) -> bool:
        """Initialize Anthropic client"""
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

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the Claude API; signature mirrors OpenAIEngine"""
        if self.client is None:
            return {"text": "Error: API client not initialized.", "tokens_generated": 0, "latency_ms": 0}

        if stop is None:
            stop = ["Observation:", "\nUser:"]

        start = time.time()

        try:
            # Build messages
            messages = []
            content = user_message or prompt
            if content:
                messages.append({"role": "user", "content": content})

            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            # Claude stop_sequences
            if stop:
                kwargs["stop_sequences"] = stop

            response = self.client.messages.create(**kwargs)

            text = response.content[0].text.strip()
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            elapsed_ms = (time.time() - start) * 1000

            # Update statistics
            self._total_calls += 1
            self._total_tokens_generated += output_tokens
            self._total_input_tokens += input_tokens
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
                "text": f"Error during Claude API call: {str(e)}",
                "tokens_generated": 0,
                "latency_ms": elapsed_ms,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Return inference performance statistics, with cost estimate"""
        avg_latency = self._total_time / max(self._total_calls, 1)
        in_cost, out_cost = self.cost_per_m
        estimated_cost = (
            self._total_input_tokens * in_cost / 1_000_000
            + self._total_output_tokens * out_cost / 1_000_000
        )
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens_generated,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_time_ms": self._total_time,
            "avg_latency_ms": avg_latency,
            "estimated_cost_usd": round(estimated_cost, 4),
        }

    def reset_stats(self):
        """"""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

