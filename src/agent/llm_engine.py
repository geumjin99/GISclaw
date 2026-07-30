"""
LLM 推理引擎 (LLM Engine)

封装 llama-cpp-python 的调用，提供统一的 LLM 推理接口。
支持通过 llama.cpp 进行完全离线推理。
"""
import os
import time
from typing import Optional, Dict, Any, List


class LLMEngine:
    """LLM 推理引擎：封装 llama-cpp-python 调用"""

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
        初始化 LLM 引擎

        Args:
            model_path: GGUF 模型文件路径
            n_ctx: 上下文窗口大小
            n_gpu_layers: GPU 加速层数（-1 = 全部）
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            verbose: 是否显示详细日志
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.repeat_penalty = repeat_penalty
        self.model = None

        # 性能统计
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0

    def load_model(self):
        """加载模型到内存/GPU"""
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
        生成文本（支持 chat completion 模式）

        如果提供了 system_prompt 和 user_message，则使用 chat completion
        模式调用（推荐用于 instruct 模型）。否则 fallback 到 text completion。

        Args:
            prompt: 输入 prompt（fallback 模式使用）
            stop: 停止序列列表
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token 数
            system_prompt: 系统提示词（chat 模式）
            user_message: 用户消息（chat 模式）

        Returns:
            包含 text, tokens_generated, latency_ms 的字典
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
            # 如果提供了 system_prompt 和 user_message，使用 chat 模式
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
                # Fallback: 纯 text completion 模式
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

            # 更新统计
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
        """获取推理性能统计"""
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
        """重置统计"""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0


class MockLLMEngine:
    """
    模拟 LLM 引擎（用于测试和演示）

    当没有实际模型时，使用预定义的响应来模拟 Agent 行为。
    """

    def __init__(self):
        self._total_calls = 0
        self._responses = []

    def load_model(self) -> bool:
        print("MockLLMEngine: Using mock responses (no actual model)")
        return True

    def set_responses(self, responses: List[str]):
        """设置预定义的响应序列"""
        self._responses = responses

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """返回预定义的响应"""
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
    OpenAI API 引擎：封装 OpenAI Chat Completion 调用

    与 LLMEngine 接口完全兼容，可无缝替换用于 GPT-4o 等云端模型。
    支持 base_url 参数以兼容 DeepSeek 等 OpenAI 兼容 API。
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

        # 性能统计（兼容 LLMEngine 接口）
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        # 费用追踪
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def load_model(self) -> bool:
        """初始化 OpenAI client"""
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
        """检测是否为 GPT-5.x 系列模型（API 参数不同）"""
        return 'gpt-5' in self.model_name.lower()

    def _is_gemini_thinking_model(self) -> bool:
        """检测是否为 Gemini thinking model（stop 参数会导致空响应）"""
        name = self.model_name.lower()
        # gemini-3.1-pro / gemini-3-pro 等 pro 系列是 thinking model
        # gemini-3-flash 不是 thinking model
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
        调用 OpenAI API 生成文本

        接口与 LLMEngine.generate() 完全兼容。
        GPT-5.x 兼容: 使用 max_completion_tokens 替代 max_tokens, 不使用 stop 参数。
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
        # Gemini thinking model 和 GPT-5.x 都不支持 stop 参数
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
            # 推理模型 (GPT-5.x / Gemini Pro): thinking tokens 占用 completion 配额
            # 需要更大的 token 配额来容纳 thinking + 实际输出
            if is_gpt5 or is_gemini_thinking:
                effective_max_tokens = max(effective_max_tokens, 16384)

            # GPT-5.x / Gemini thinking: 不使用 stop 参数，生成后手动截断
            if needs_manual_stop:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=effective_max_tokens,
                    temperature=temperature or self.temperature,
                    # stop 不支持 — 生成后手动截断
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=effective_max_tokens,
                    temperature=temperature or self.temperature,
                    stop=stop,
                )

            # 防护 content=None（Gemini thinking model 有时返回空内容）
            raw_content = response.choices[0].message.content
            text = raw_content.strip() if raw_content else ""
            output_tokens = response.usage.completion_tokens if response.usage else 0
            input_tokens = response.usage.prompt_tokens if response.usage else 0

            # 空响应防护：返回格式错误提示让 agent 重试
            if not text:
                text = "Thought: (empty response from model)\nAction: \nArgs: {}"

            # 手动截断到第一个 stop 序列
            if needs_manual_stop and stop:
                for stop_seq in stop:
                    idx = text.find(stop_seq)
                    if idx != -1:
                        text = text[:idx].strip()
                        break

            elapsed_ms = (time.time() - start) * 1000

            # 更新统计
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
        """获取推理性能统计（含费用估算）"""
        avg_latency = self._total_time / max(self._total_calls, 1)
        # 使用动态定价
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
        """重置统计"""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0


class OllamaEngine:
    """
    Ollama 推理引擎：通过 Ollama 本地 API 调用模型

    Ollama 提供 OpenAI 兼容 API（http://localhost:11434/v1），
    因此复用 openai 库但指向本地地址。
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

        # 性能统计
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0

    def load_model(self) -> bool:
        """初始化 Ollama client（通过 OpenAI 兼容 API）"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",  # Ollama 不需要真实 key
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
        调用 Ollama API 生成文本

        接口与 LLMEngine.generate() 完全兼容。
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
    Anthropic Claude API 引擎

    接口与 OpenAIEngine 完全兼容，可无缝替换。
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

        # 性能统计
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def load_model(self) -> bool:
        """初始化 Anthropic client"""
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
        """新一代 Claude 模型移除了 temperature/top_p/top_k，传了会 400。"""
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
    ) -> Dict[str, Any]:
        """调用 Claude API 生成文本（兼容 OpenAIEngine 接口）"""
        if self.client is None:
            return {"text": "Error: API client not initialized.", "tokens_generated": 0, "latency_ms": 0}

        if stop is None:
            stop = ["Observation:", "\nUser:"]

        start = time.time()

        try:
            # 构建消息
            messages = []
            content = user_message or prompt
            if content:
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
                kwargs["system"] = system_prompt
            # Claude 支持 stop_sequences
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
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            elapsed_ms = (time.time() - start) * 1000

            # 更新统计
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
        """获取推理性能统计（含费用估算）"""
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
        """重置统计"""
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

