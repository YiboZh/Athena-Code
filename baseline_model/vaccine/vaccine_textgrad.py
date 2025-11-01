import csv
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import textgrad as tg
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from athena.datasets import load_selected_pids
from athena.logging_config import setup_logging
from athena.prompts import CHOICES_VAC

tg.set_backward_engine(tg.get_engine("gpt-4o-mini"))

main_logger = setup_logging()

def optimize_prediction(
    profile: str,
    alternatives: str,
    choices: list[str],
    iterations: int = 5,
):
    """Return (raw_text_with_prompt, preds_dict_filtered)."""
    initial_solution = f"""Task: Estimate the probability distribution over vaccination options for an individual.

Profile:
{profile}

Alternatives:
{alternatives}

Solution (JSON):
{{"Unvaccinated": 0.333, "Vaccinated_no_booster": 0.334, "Booster": 0.333}}""".strip()

    solution = tg.Variable(initial_solution,
                           requires_grad=True,
                           role_description="full prompt+answer")

    grading_prompt = tg.Variable(
        """You are an expert on vaccine-uptake behavior.
Given the profile, alternatives, and predicted probabilities in the user's message,
output a single line:
Score: <float between 0 and 1>
where 1 means the probabilities look highly reasonable and 0 means implausible.
Only output that line, no extra commentary.""",
        requires_grad=False,
        role_description="grading prompt"
    )

    loss_fn = tg.TextLoss(grading_prompt)
    optimizer = tg.TGD([solution])

    for _ in range(iterations):
        loss = loss_fn(solution)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    final_text = solution.value
    m = re.search(r"\{[\s\S]*\}", final_text)
    if not m:
        raise ValueError("No JSON found after optimization.")
    full_preds = json.loads(m.group(0))

    return final_text, full_preds


def run_vaccine_zero_shot(
    input_csv: str,
    output_csv: str,
    selected_memids: list,
    choices: list
):
    fixed_cols = ['No.', *choices, 'raw_response', 'target']   # Set the column order once up front
    df = pd.read_csv(input_csv)
    df = df[df['No.'].isin(selected_memids)].drop_duplicates('No.')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=fixed_cols).to_csv(output_csv, index=False)

    done = set(pd.read_csv(output_csv)['No.'])
    print(f"Already done {len(done)} rows, total {len(df)} rows to process.")

    for _, row in df.iterrows():
        no = row['No.']
        if no in done:
            continue
        try:
            raw, full_preds = optimize_prediction(
                profile=f"{row['demographic_prompt']} {row['other_prompt']}",
                alternatives=", ".join(choices),
                choices=choices,
            )
        except Exception as e:
            main_logger.error(f"prediction failed No.{no}: {e}")
            continue

        # Keep only the three categories we track; ignore extras
        def safe_float(x):
            """
            Try to convert x into a float. If x is a dict, recursively search for common probability fields.
            Return 0.0 when conversion fails.
            """
            if isinstance(x, (int, float)):
                return float(x)

            if isinstance(x, str):
                try:
                    return float(x)
                except ValueError:
                    return 0.0

            if isinstance(x, dict):
                # Common field names
                for key in ("prob", "value", "p", "score", "Probability"):
                    if key in x:
                        return safe_float(x[key])
                # Otherwise pick the first value that parses to a number
                for v in x.values():
                    n = safe_float(v)
                    if n != 0.0:
                        return n
                return 0.0     # Fallback

            return 0.0

        preds = {c: safe_float(full_preds.get(c, 0.0)) for c in choices}
        total = sum(preds.values())
        if total == 0:
            preds = {k: 1/len(choices) for k in choices}
        else:
            preds = {k: v/total for k, v in preds.items()}

        out_row = {'No.': no, **preds, 'raw_response': raw, 'target': row['target']}
        # Write with the same column order to keep alignment
        pd.DataFrame([out_row], columns=fixed_cols).to_csv(
            output_csv,
            mode='a',
            header=False,
            index=False,
            quoting=csv.QUOTE_MINIMAL
        )
        done.add(no)

    out_df = pd.read_csv(output_csv)
    y_true = out_df['target'].astype(int).tolist()
    y_pred = out_df[choices].values.tolist()
    acc = accuracy_score(y_true, [p.index(max(p)) for p in y_pred])
    ce = log_loss(y_true, y_pred, labels=list(range(len(choices))))
    print(f"Total predictions: {len(out_df)} | Accuracy: {acc:.4f} | Cross-Entropy: {ce:.4f}")

if __name__ == "__main__":
    selected_memids = load_selected_pids(
        PROJECT_ROOT / "data" / "vaccine" / "selected_pids_subset_100.txt"
    )
    run_vaccine_zero_shot(
        input_csv=str(PROJECT_ROOT / "data" / "vaccine" / "processed_prompts_descriptive_3_new.csv"),
        output_csv=str(PROJECT_ROOT / "results_vac" / "vaccine_zeroshot_textgrad_clean.csv"),
        selected_memids=selected_memids,
        choices=CHOICES_VAC,
    )
