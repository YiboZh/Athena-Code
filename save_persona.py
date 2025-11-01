from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from athena.chatbots import ChatbotAPI, OllamaAPI
from athena.datasets import DATA_ROOT, Vaccines, load_selected_pids
from athena.logging_config import setup_logging
from athena.textual import loading_persona


logger = logging.getLogger(__name__)


def generate_vaccine_personas(
    *,
    chatbot: ChatbotAPI | None = None,
    results_dir: Path | None = None,
    selected_pids_file: Path | None = None,
) -> None:
    setup_logging("athena.save_persona")
    bot = chatbot or OllamaAPI(model="qwen3:32b")

    results_path = Path(results_dir) if results_dir else Path("results_vac-qwen3")
    results_path.mkdir(parents=True, exist_ok=True)
    persona_path = results_path / "persona.csv"

    selected_path = (
        Path(selected_pids_file)
        if selected_pids_file
        else DATA_ROOT / "vaccine" / "selected_pids.txt"
    )
    selected_memids = set(load_selected_pids(selected_path))

    dataset = Vaccines()
    persona_df = pd.read_csv(persona_path) if persona_path.exists() else pd.DataFrame(columns=["member_id", "persona"])
    processed = set(persona_df["member_id"].tolist())

    rows: list[dict] = []
    for group in dataset.groups:
        for member in group.members:
            if member.pid in processed:
                continue
            if member.pid not in selected_memids:
                continue

            prompt_row = dataset.prompts.iloc[member.pid]
            persona = loading_persona(
                bot,
                None,
                vaccine=True,
                features=prompt_row["demographic_prompt"],
                survey=prompt_row["other_prompt"],
            )
            logger.info("Generated persona for member %s: %s", member.pid, persona)
            rows.append({"member_id": member.pid, "persona": persona})

    if rows:
        new_df = pd.DataFrame(rows)
        persona_df = pd.concat([persona_df, new_df], ignore_index=True)
        persona_df.to_csv(persona_path, index=False)


if __name__ == "__main__":
    generate_vaccine_personas()
