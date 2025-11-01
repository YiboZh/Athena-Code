import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path

import pandas as pd
import textgrad as tg
from sklearn.metrics import accuracy_score, log_loss
from tenacity import retry, stop_after_attempt, wait_none

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))  # Make the project root importable

from athena.datasets import load_selected_pids
from athena.logging_config import setup_logging
from athena.prompts import CHOICES

tg.set_backward_engine(tg.get_engine("gpt-4o-mini"))

main_logger = setup_logging()


# ----------------------------- 1. Core optimizer ----------------------------- #
@retry(stop=stop_after_attempt(3), wait=wait_none())
def optimize_prediction(trip_info: str,
                        transport_options: str,
                        iterations: int = 5):
    """
    Optimize the textgrad prompt and solution for a single trip.
    Return (raw_text_with_prompt, preds_dict_filtered).
    """
    initial_solution = f"""Task: Estimate the probability distribution over three travel modes
(Swissmetro, Train, Car) for a single trip.

<TRIP_INFO>
{trip_info}
</TRIP_INFO>

<TRANSPORT_OPTIONS>
{transport_options}
</TRANSPORT_OPTIONS>

Solution (JSON):
{{"Swissmetro": 0.333, "Train": 0.333, "Car": 0.334}}""".strip()

    # Trainable prompt and answer bundle
    solution = tg.Variable(
        initial_solution,
        requires_grad=True,
        role_description="full prompt+answer"
    )

    # Grader: returns a 0-1 score, higher is more reasonable
    grading_prompt = tg.Variable(
        """You are a transport-economics expert.
Given the trip info, transport options, and predicted probabilities in the
user's message, output a single line ONLY:
Score: <float between 0 and 1>
1 = probabilities look highly reasonable, 0 = implausible.
Remember: THE PREDICTION MUST BE A JSON DICT""",
        requires_grad=False,
        role_description="grading prompt"
    )

    loss_fn   = tg.TextLoss(grading_prompt)
    optimizer = tg.TGD([solution])

    for _ in range(iterations):
        loss = loss_fn(solution)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    final_text = solution.value
    m = re.search(r"\{[\s\S]*\}", final_text)
    if not m:
        raise ValueError("No JSON found after optimization.", final_text)
    full_preds = json.loads(m.group(0))

    # Keep only the three target modes; default missing ones to 0
    filtered = {mode: float(full_preds.get(mode, 0.0)) for mode in CHOICES}
    return final_text, filtered


# ----------------------------- 2. Batch execution ----------------------------- #
def run_travelmode_zero_shot(
        input_csv: str,
        ground_truth_csv: str,
        output_csv: str,
        selected_memids: list,
        choices: list,
        mapping: dict):
    """
    input_csv: prompt_multi_col.csv (contains TRIP_ID, trip_info, transport_options)
    ground_truth_csv: swissmetro_processed.csv (contains ID, TRIP_ID, CHOICE/choice)
    """
    # Load data
    input_df  = pd.read_csv(input_csv)
    truth_df  = pd.read_csv(ground_truth_csv)

    # Keep only selected members and retain the first trip per member
    truth_sample = (truth_df[truth_df['ID'].isin(selected_memids)]
                    .drop_duplicates(subset='ID', keep='first'))

    # Prepare output file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['ID', 'TRIP_ID', 'raw_response', *choices,
                              'actual_choice']).to_csv(output_csv, index=False)

    done = set(zip(*pd.read_csv(output_csv)[['ID', 'TRIP_ID']].values.T))

    # Predict row by row
    for _, row in truth_sample.iterrows():
        memid   = row['ID']
        trip_id = row['TRIP_ID']
        if (memid, trip_id) in done:
            continue

        try:
            prompt_row = input_df[input_df['TRIP_ID'] == trip_id].iloc[0]
        except IndexError:
            main_logger.warning(f"TRIP_ID {trip_id} not found in prompt file; skipped.")
            continue

        try:
            raw, preds = optimize_prediction(prompt_row['trip_info'],
                                             prompt_row['transport_options'])
        except Exception as e:
            main_logger.warning(f"optimize_prediction failed (ID={memid}, "
                              f"TRIP_ID={trip_id}): {e}\n{traceback.format_exc()}")
            continue

        # Normalize
        total = sum(preds.values())
        preds = {k: (v/total) if total else 1/len(choices) for k, v in preds.items()}

        out_row = {'ID': memid,
                   'TRIP_ID': trip_id,
                   'raw_response': raw,
                   **preds,
                   'actual_choice': row.get('choice', row.get('CHOICE'))}
        pd.DataFrame([out_row]).to_csv(output_csv, mode='a',
                                       header=False, index=False,
                                       quoting=csv.QUOTE_MINIMAL)
        done.add((memid, trip_id))

    # ---------------- Evaluation ---------------- #
    res_df = pd.read_csv(output_csv)
    y_true = []
    for val in res_df['actual_choice']:
        if val in mapping:
            y_true.append(mapping[val])
        else:
            main_logger.warning(f"Unknown actual_choice '{val}', skipped.")
            y_true.append(-1)
    y_pred = res_df[choices].values.tolist()

    acc = accuracy_score(y_true, [p.index(max(p)) for p in y_pred])
    ce  = log_loss(y_true, y_pred, labels=list(range(len(choices))))
    print(f"[SUMMARY] Total predictions: {len(res_df)} | "
          f"Accuracy: {acc:.4f} | Cross-Entropy: {ce:.4f}")


# ----------------------------- 3. Entry point ----------------------------- #
if __name__ == "__main__":
    choices = CHOICES
    mapping = {1: 0, 2: 1, 3: 2}

    # Read member list
    selected_memids = load_selected_pids(
        PROJECT_ROOT /
        "data" / "swissmetro" / "selected_pids_subset_100.txt"
    )

    try:
        run_travelmode_zero_shot(
            input_csv=str(PROJECT_ROOT /
                          "data" / "swissmetro" / "prompt_multi_col.csv"),
            ground_truth_csv=str(PROJECT_ROOT /
                                 "data" / "swissmetro" /
                                 "swissmetro_processed.csv"),
            output_csv=str(PROJECT_ROOT /
                           "results" / "swissmetro_zeroshot_textgrad.csv"),
            selected_memids=selected_memids,
            choices=choices,
            mapping=mapping
        )
    except Exception as e:
        main_logger.error(f"Travel-mode zeroshot failed: {e}")
