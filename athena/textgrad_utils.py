from textgrad.engine.base import EngineLM, CachedEngine
from textgrad.engine_experimental.litellm import LiteLLMEngine
from textgrad.engine.openai import OpenAI, OLLAMA_BASE_URL
import platformdirs
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Union, List
import tiktoken
import json
import base64
import imghdr


def _detect_image_type(data: bytes) -> str:
    """Best-effort detection of image mime subtype from raw bytes.

    Falls back to ``png`` when the standard library cannot infer the type.
    """

    detected = imghdr.what(None, data)
    return detected or "png"

def _num_tokens_from_messages(messages, model_name="gpt-3.5-turbo"):
    """
    Count tokens in a list of chat-format messages.
    Falls back to cl100k_base when model-specific encoding is unavailable.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens_per_msg = 4      # every message has a 3‑token overhead
    tokens_per_name = 2     # if "name" field present
    num_tokens = 0
    for msg in messages:
        num_tokens += tokens_per_msg
        for k, v in msg.items():
            num_tokens += len(enc.encode(v))
            if k == "name":
                num_tokens += tokens_per_name
    num_tokens += 2  # assistant priming
    return num_tokens

def truncate_messages_to_fit(messages, model_name, context_window, safety_margin=1024):
    """
    Dynamically drop the oldest conversational turns (after the system
    prompt) until the total token count plus safety_margin fits into
    context_window.  The first message (system) is always kept.

    Returns a *new* list; the original is not modified.
    """
    pruned = messages[:]  # shallow copy
    while (
        len(pruned) > 1  # keep at least the system message
        and _num_tokens_from_messages(pruned, model_name) + safety_margin
        > context_window
    ):
        pruned.pop(1)  # remove the oldest non‑system message
    return pruned

LAMBDA_BASE_URL = "https://api.lambda.ai/v1"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

class ChatOpenAI_plus(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str="gpt-3.5-turbo-0613",
        system_prompt: str=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        base_url: str=None,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.base_url = base_url

        if not base_url:
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")

            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif base_url and base_url == OLLAMA_BASE_URL:
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"
            )
        elif base_url and base_url == LAMBDA_BASE_URL:
            self.client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("LAMBDA_API_KEY")
            )
        elif base_url and base_url == DEEPINFRA_BASE_URL:
            self.client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("DEEPINFRA_API_KEY")
            )
        else:
            raise ValueError("Invalid base URL provided. Please use the default OLLAMA base URL or None.")

        self.model_string = model_string
        self.is_multimodal = is_multimodal

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")

            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str=None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt},
        ]

        # ===== Dynamic context‑window safeguard =====
        _CONTEXT_WINDOWS = {
            "qwen3": 40960,
            "qwen": 40960,
            "deepseek": 40960,
            "deepseek-r1": 40960,
            "gpt-4o": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
        }
        context_window = next(
            (size for key, size in _CONTEXT_WINDOWS.items() if key in self.model_string),
            40960  # fallback to a 40 960‑token window when model is unknown
        )

        # --- Dynamically truncate history so prompt + safety fits window ---
        safety_margin = 1024  # keep consistent with later usage
        messages = truncate_messages_to_fit(
            messages, self.model_string, context_window, safety_margin
        )

        prompt_tokens = _num_tokens_from_messages(messages, self.model_string)
        if prompt_tokens >= context_window:
            raise ValueError(
                f"Prompt too long: {prompt_tokens} tokens (limit {context_window})"
            )

        # Adjust max_tokens so prompt + completion stays within the window
        # Leave a small safety buffer because server‑side tokenization can differ
        safety_margin = 1024
        available_tokens = context_window - prompt_tokens - safety_margin
        max_tokens = min(max_tokens, available_tokens)

        if max_tokens < 32:
            raise ValueError(
                f"Prompt too long even after safety margin: "
                f"{prompt_tokens} prompt tokens (+{safety_margin} buffer) "
                f">= {context_window}"
            )

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = _detect_image_type(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        self._save_cache(cache_key, response_text)
        return response_text


__ENGINE_NAME_SHORTCUTS__ = {
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229",
    "sonnet-3.5": "claude-3-5-sonnet-20240620",
    "together-llama-3-70b": "together-meta-llama/Llama-3-70b-chat-hf",
    "vllm-llama-3-8b": "vllm-meta-llama/Meta-Llama-3-8B-Instruct",
}

# Any better way to do this?
__MULTIMODAL_ENGINES__ = ["gpt-4-turbo",
                          "gpt-4o",
                          "claude-3-5-sonnet-20240620",
                          "claude-3-opus-20240229",
                          "claude-3-sonnet-20240229",
                          "claude-3-haiku-20240307",
                          "gpt-4-turbo-2024-04-09",
                          ]

def _check_if_multimodal(engine_name: str):
    return any([name == engine_name for name in __MULTIMODAL_ENGINES__])

def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}")

def get_engine_plus(engine_name: str, **kwargs) -> EngineLM:
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]

    if "seed" in kwargs and "gpt-4" not in engine_name and "gpt-3.5" not in engine_name and "gpt-35" not in engine_name:
        raise ValueError(f"Seed is currently supported only for OpenAI engines, not {engine_name}")

    if "cache" in kwargs and "experimental" not in engine_name:
        raise ValueError(f"Cache is currently supported only for LiteLLM engines, not {engine_name}")

    # check if engine_name starts with "experimental:"
    if engine_name.startswith("experimental:"):
        engine_name = engine_name.split("experimental:")[1]
        return LiteLLMEngine(model_string=engine_name, **kwargs)
    if engine_name.startswith("azure"):
        from textgrad.engine.openai import AzureChatOpenAI
        # remove engine_name "azure-" prefix
        engine_name = engine_name[6:]
        return AzureChatOpenAI(model_string=engine_name, **kwargs)
    elif (("gpt-4" in engine_name) or ("gpt-3.5" in engine_name)):
        from textgrad.engine.openai import ChatOpenAI
        return ChatOpenAI(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "claude" in engine_name:
        from textgrad.engine.anthropic import ChatAnthropic
        return ChatAnthropic(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "gemini" in engine_name:
        from textgrad.engine.gemini import ChatGemini
        return ChatGemini(model_string=engine_name, **kwargs)
    elif "together" in engine_name:
        from textgrad.engine.together import ChatTogether
        engine_name = engine_name.replace("together-", "")
        return ChatTogether(model_string=engine_name, **kwargs)
    elif engine_name in ["command-r-plus", "command-r", "command", "command-light"]:
        from textgrad.engine.cohere import ChatCohere
        return ChatCohere(model_string=engine_name, **kwargs)
    elif engine_name.startswith("ollama"):
        from textgrad.engine.openai import ChatOpenAI, OLLAMA_BASE_URL
        model_string = engine_name.replace("ollama-", "")
        return ChatOpenAI(
            model_string=model_string,
            base_url=OLLAMA_BASE_URL,
            **kwargs
        )
    elif engine_name.startswith("lambda"):
        model_string = engine_name.replace("lambda-", "")
        return ChatOpenAI_plus(
            model_string=model_string,
            base_url=LAMBDA_BASE_URL,
            **kwargs
        )
    elif engine_name.startswith("deepinfra"):
        model_string = engine_name.replace("deepinfra-", "")
        return ChatOpenAI_plus(
            model_string=model_string,
            base_url=DEEPINFRA_BASE_URL,
            **kwargs
        )
    elif "vllm" in engine_name:
        from textgrad.engine.vllm import ChatVLLM
        engine_name = engine_name.replace("vllm-", "")
        return ChatVLLM(model_string=engine_name, **kwargs)
    elif "groq" in engine_name:
        from textgrad.engine.groq import ChatGroq
        engine_name = engine_name.replace("groq-", "")
        return ChatGroq(model_string=engine_name, **kwargs)
    else:
        raise ValueError(f"Engine {engine_name} not supported")
