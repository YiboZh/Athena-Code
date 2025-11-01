import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from athena.chatbots import ChatCompletionAPI, ChatbotAPI, GeminiAPI
from athena.datasets import load_selected_pids
from athena.logging_config import setup_logging
from athena.prompts import CHOICES_VAC, PREDICTION_SYS_VAC_POSSIBILITY

main_logger = setup_logging()

# Format prompts for the zero-shot vaccine task
def format_vac_prompts(template: str, profile: str, alternatives: str) -> list:
    return [
        {"role": "system", "content": PREDICTION_SYS_VAC_POSSIBILITY},
        {"role": "user", "content": (
            f"<TEMPLATE>\n{template}\n</TEMPLATE>\n"
            f"<PROFILE>\n{profile}\n</PROFILE>\n"
            f"<ALTERNATIVES>\n{alternatives}\n</ALTERNATIVES>"
            # f"<ALTERNATIVES>\n{alternatives}\n</ALTERNATIVES>\nLet's think step by step."
        )}
    ]


def _canonicalize_vac_label(label: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", label.lower())
    if "unvac" in normalized:
        return "Unvaccinated"
    if "noboost" in normalized or "nobooster" in normalized:
        return "Vaccinated_no_booster"
    if "boost" in normalized:
        return "Booster"
    return None


# Retryable LLM call that extracts and normalizes JSON probabilities
@retry(
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
)
def call_llm(chatbot: ChatbotAPI, messages):
    raw = chatbot.create(messages)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw}")
    preds = json.loads(match.group(0))
    # Ensure all classes present
    mapped: dict[str, float] = {}
    for key, value in preds.items():
        cls = _canonicalize_vac_label(key)
        if cls is not None and cls not in mapped:
            mapped[cls] = value
    for cls in CHOICES_VAC:
        if cls not in mapped:
            raise ValueError(f"Class '{cls}' missing in output JSON: {preds}")
    total = sum(mapped.values())
    if total <= 0:
        raise ValueError(f"Sum of probabilities is non-positive: {mapped}")
    normalized = {k: v / total for k, v in mapped.items()}
    return raw, normalized

# Main prediction loop

def run_vaccine_zero_shot(
    input_csv: str,
    output_csv: str,
    chatbot,
    selected_memids: list,
    choices: list
):
    # Read input (which also contains ground truth)
    df = pd.read_csv(input_csv)

    # Filter to only selected IDs
    df_filtered = df[df['No.'].isin(selected_memids)]
    df_sample = df_filtered.drop_duplicates(subset='No.', keep='first')

    # Prepare output file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['No.', *choices, 'raw_response', 'target']).to_csv(output_csv, index=False)

    # Load any existing outputs to skip
    existing = set(pd.read_csv(output_csv)['No.'].tolist())

    # Iterate through each case
    for _, row in df_sample.iterrows():
        no = row['No.']
        if no in existing:
            continue

        template = row['persona']
        profile = f"{row['demographic_prompt']} {row['other_prompt']}"
        alternatives = ", ".join(choices)

        messages = format_vac_prompts(template, profile, alternatives)
        raw_resp, preds = call_llm(chatbot, messages)

        # Build output entry
        entry = {
            'No.': no,
            **preds,
            'raw_response': raw_resp,
            'target': row['target']  # Assumes target values align with choices indices
        }
        pd.DataFrame([entry]).to_csv(output_csv, mode='a', header=False, index=False)
        existing.add(no)

    # Compute and print metrics
    out_df = pd.read_csv(output_csv)
    y_true = out_df['target'].tolist()
    y_pred = out_df[choices].values.tolist()

    acc = accuracy_score(y_true, [p.index(max(p)) for p in y_pred])
    ce = log_loss(y_true, y_pred, labels=list(range(len(choices))))
    print(f"Completed {len(out_df)} predictions.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cross-Entropy Loss: {ce:.4f}")


if __name__ == "__main__":
    chatbot = GeminiAPI(model="gemini-2.0-flash")
    # chatbot = ChatCompletionAPI(model="gpt-4o-mini")
    selected_memids = load_selected_pids(
        PROJECT_ROOT / "data" / "vaccine" / "selected_pids_subset_100.txt"
    )
    choices = CHOICES_VAC

    run_vaccine_zero_shot(
        input_csv=str(PROJECT_ROOT / "data" / "vaccine" / "processed_prompts_descriptive_3_new.csv"),
        output_csv=str(PROJECT_ROOT / "results_vac" / "vaccine_zeroshot_results.csv"),
        chatbot=chatbot,
        selected_memids=selected_memids,
        choices=choices,
    )
