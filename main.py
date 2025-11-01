from __future__ import annotations

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Type

from athena.chatbots import (
    ChatCompletionAPI,
    ChatbotAPI,
    DeepInfraAPI,
    GeminiAPI,
    LambdaAPI,
    OllamaAPI,
)
from athena.pipeline import run_travel_mode, run_vaccine


CHATBOT_FACTORIES: Dict[str, Type[ChatbotAPI]] = {
    "ollama": OllamaAPI,
    "openai": ChatCompletionAPI,
    "gemini": GeminiAPI,
    "lambda": LambdaAPI,
    "deepinfra": DeepInfraAPI,
}


def build_chatbot(name: Optional[str], model: Optional[str]) -> Optional[ChatbotAPI]:
    if name is None:
        return None
    try:
        factory = CHATBOT_FACTORIES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported chatbot provider '{name}'.") from exc

    kwargs = {"model": model} if model else {}
    return factory(**kwargs)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Athena experiments.")
    parser.add_argument(
        "--task",
        choices=["travel-mode", "vaccine"],
        default="travel-mode",
        help="Select which experiment pipeline to run.",
    )
    parser.add_argument(
        "--chatbot",
        choices=sorted(CHATBOT_FACTORIES.keys()),
        default=None,
        help="Chatbot backend to use (defaults to pipeline's built-in choice).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier to pass to the chatbot backend.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to write experiment artefacts.",
    )
    parser.add_argument(
        "--persona-csv",
        default=None,
        help="Path to persona CSV cache (utility task only).",
    )
    parser.add_argument(
        "--selected-pids-file",
        default=None,
        help="Custom selected participant list to use.",
    )
    parser.add_argument("--iterations", type=int, default=30, help="Number of optimisation iterations.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of top candidates kept per iteration.")
    parser.add_argument("--bottom-k", type=int, default=1, help="Number of bottom candidates kept per iteration.")
    parser.add_argument("--n-candidates", type=int, default=5, help="Candidate expressions generated per round.")
    parser.add_argument("--tg-steps", type=int, default=5, help="TextGrad optimisation steps.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    chatbot = build_chatbot(args.chatbot, args.model)

    common_kwargs = {
        "chatbot": chatbot,
        "results_dir": Path(args.results_dir) if args.results_dir else None,
        "selected_pids_file": Path(args.selected_pids_file) if args.selected_pids_file else None,
        "iterations": args.iterations,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
        "n_candidates": args.n_candidates,
        "tg_steps": args.tg_steps,
    }

    if args.task == "travel-mode":
        travel_kwargs = {**common_kwargs}
        travel_kwargs["persona_csv"] = Path(args.persona_csv) if args.persona_csv else None
        run_travel_mode(**travel_kwargs)
    else:
        run_vaccine(**common_kwargs)


if __name__ == "__main__":
    main()
