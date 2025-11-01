import json
import sys
import re
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from athena.chatbots import ChatCompletionAPI, ChatbotAPI
from athena.datasets import load_selected_pids
from athena.logging_config import setup_logging
from athena.prompts import CHOICES, PREDICTION_SYS_ZEROSHOT, PREDICTION_ZEROSHOT

main_logger = setup_logging()

# Format prompts for the zero-shot task
def format_prompts(trip_info: str, transport_options: str) -> list:
    return [
        {"role": "system", "content": PREDICTION_SYS_ZEROSHOT},
        {"role": "user", "content": PREDICTION_ZEROSHOT.format(
            trip_info=trip_info,
            transport_options=transport_options
        )}
    ]

# Call LLM and return both raw and normalized predictions
@retry(
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ValueError, json.JSONDecodeError))
)
def call_llm(chatbot: ChatbotAPI, messages):
    raw = chatbot.create(messages)
    # Extract the JSON object between the first '{' and last '}'
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        raw_json = match.group(0)
    else:
        raise ValueError(f"No JSON object found in response: {raw}")
    preds = json.loads(raw_json)
    # Ensure all modes present
    for mode in CHOICES:
        if mode not in preds:
            raise ValueError(f"Mode '{mode}' missing in output JSON: {preds}")
    total = sum(preds.values())
    if total <= 0:
        raise ValueError(f"Sum of probabilities is non-positive: {preds}")
    normalized = {k: v / total for k, v in preds.items()}
    return raw, normalized

def run_predictions(
    input_csv: str,
    ground_truth_csv: str,
    output_csv: str,
    chatbot,
    mapping: dict,
    choices: list,
    selected_memids: list
):
    # Load input and ground truth
    input_df = pd.read_csv(input_csv)
    truth_df = pd.read_csv(ground_truth_csv)

    # Filter ground truth to only selected IDs
    truth_filtered = truth_df[truth_df['ID'].isin(selected_memids)]
    # For each ID, keep only the first occurrence
    truth_sample = truth_filtered.drop_duplicates(subset='ID', keep='first')

    # Prepare output file if not exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=[
            'ID', 'TRIP_ID', 'raw_response', *choices, 'actual_choice'
        ]).to_csv(output_csv, index=False)

    # Load existing entries to skip
    existing = set()
    existing_df = pd.read_csv(output_csv)
    existing = set(zip(existing_df['ID'], existing_df['TRIP_ID']))

    # Predict for each sampled ground truth row
    for _, row in truth_sample.iterrows():
        memid = row['ID']
        trip_id = row['TRIP_ID']

        # Skip if already processed
        if (memid, trip_id) in existing:
            continue

        # Find corresponding input row where 'TRIP_ID' matches
        input_row = input_df[input_df['TRIP_ID'] == trip_id].iloc[0]
        messages = format_prompts(
            input_row['trip_info'],
            input_row['transport_options']
        )

        raw_resp, preds = call_llm(chatbot, messages)
        actual = row.get('choice', row.get('CHOICE'))

        entry = {
            'ID': memid,
            'TRIP_ID': trip_id,
            'raw_response': raw_resp,
            **{mode: preds.get(mode, 0) for mode in choices},
            'actual_choice': actual
        }
        pd.DataFrame([entry]).to_csv(output_csv, mode='a', header=False, index=False)

        # Add to existing to prevent duplicate if rerun
        existing.add((memid, trip_id))

    # Compute and print summary metrics
    all_df = pd.read_csv(output_csv)
    y_true = []
    for val in all_df['actual_choice']:
        if val in mapping:
            y_true.append(mapping[val])
        else:
            main_logger.warning(f"Value '{val}' not found in mapping. Skipping.")
            y_true.append(-1)  # or some other placeholder value

    y_pred = all_df[choices].values.tolist()
    acc = accuracy_score(y_true, [p.index(max(p)) for p in y_pred])
    ce = log_loss(y_true, y_pred, labels=list(range(len(choices))))
    print(f"Completed {len(all_df)} predictions.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cross-Entropy Loss: {ce:.4f}")


if __name__ == "__main__":
    mapping = {1: 0, 2: 1, 3: 2}
    choices = CHOICES

    chatbot = ChatCompletionAPI(model="gpt-4o-mini")

    selected_memids = load_selected_pids(
        PROJECT_ROOT / "data" / "swissmetro" / "selected_pids_subset_100.txt"
    )

    try:
        run_predictions(
            input_csv=str(PROJECT_ROOT / "data" / "swissmetro" / "prompt_multi_col.csv"),
            ground_truth_csv=str(PROJECT_ROOT / "data" / "swissmetro" / "swissmetro_processed.csv"),
            output_csv=str(PROJECT_ROOT / "results" / "swissmetro_zeroshot_results.csv"),
            chatbot=chatbot,
            mapping=mapping,
            choices=choices,
            selected_memids=selected_memids,
        )
    except Exception as exc:
        main_logger.warning(f"An error occurred in travel mode zeroshot: {exc}")
        raise
