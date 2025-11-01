"""Chatbot API wrappers used by the Athena pipelines."""

from __future__ import annotations

import json
import os
import re
from typing import List

import openai
import requests
from google import genai
from google.genai import types


class ChatbotAPI:
    """Base class for chatbot providers."""

    def __init__(self, model: str) -> None:
        self.model = model

    def create(self, messages: List[dict]) -> str:
        raise NotImplementedError


class ChatCompletionAPI(ChatbotAPI):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(model)
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chatbot = client.chat.completions

    def create(self, messages: List[dict]) -> str:
        response = self.chatbot.create(model=self.model, messages=messages)
        return response.choices[0].message.content


class OllamaAPI(ChatbotAPI):
    def __init__(self, model: str = "llama3") -> None:
        super().__init__(model)

    def create(self, messages: List[dict]) -> str:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        response_content = json.loads(response.content)["message"]["content"]
        think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        return think_pattern.sub("", response_content).strip()

    def generate(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return json.loads(response.content)["message"]["content"]


class GeminiAPI(ChatbotAPI):
    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        super().__init__(model)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def create(self, messages: List[dict]) -> str:
        system_instruction = None
        user_contents: List[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_instruction = content
            else:
                user_contents.append(content)

        prompt = "\n".join(user_contents)
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)
        else:
            config = types.GenerateContentConfig()

        response = self.client.models.generate_content(
            model=self.model,
            config=config,
            contents=prompt,
        )
        return response.text


class LambdaAPI(ChatbotAPI):
    def __init__(self, model: str = "llama-4-maverick-17b-128e-instruct-fp8") -> None:
        super().__init__(model)
        client = openai.OpenAI(
            api_key=os.getenv("LAMBDA_API_KEY"),
            base_url="https://api.lambda.ai/v1",
        )
        self.chatbot = client.chat.completions

    def create(self, messages: List[dict]) -> str:
        response = self.chatbot.create(model=self.model, messages=messages)
        content = response.choices[0].message.content
        think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        return think_pattern.sub("", content).strip()


class DeepInfraAPI(ChatbotAPI):
    def __init__(self, model: str = "deepseek-r1:32b") -> None:
        super().__init__(model)
        client = openai.OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.chatbot = client.chat.completions

    def create(self, messages: List[dict]) -> str:
        response = self.chatbot.create(model=self.model, messages=messages)
        content = response.choices[0].message.content
        think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        return think_pattern.sub("", content).strip()


__all__ = [
    "ChatbotAPI",
    "ChatCompletionAPI",
    "OllamaAPI",
    "GeminiAPI",
    "LambdaAPI",
    "DeepInfraAPI",
]

