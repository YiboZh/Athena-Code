import pandas as pd
import json
import os
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from sklearn.metrics import accuracy_score, log_loss

from athena.chatbots import ChatCompletionAPI, ChatbotAPI
from athena.datasets import load_selected_pids
from athena.logging_config import setup_logging
from athena.prompts import CHOICES, PREDICTION_FEWSHOT, PREDICTION_SYS_FEWSHOT

main_logger = setup_logging()

def format_prompts_fewshot(examples: list[dict], trip_info: str, transport_options: str) -> list:
    content = PREDICTION_FEWSHOT.format(
        trip_info_1=examples[0]['trip_info'],
        transport_options_1=examples[0]['transport_options'],
        choice_1=examples[0]['choice'],
        trip_info_2=examples[1]['trip_info'],
        transport_options_2=examples[1]['transport_options'],
        choice_2=examples[1]['choice'],
        trip_info_3=examples[2]['trip_info'],
        transport_options_3=examples[2]['transport_options'],
        choice_3=examples[2]['choice'],
        trip_info_4=examples[3]['trip_info'],
        transport_options_4=examples[3]['transport_options'],
        choice_4=examples[3]['choice'],
        trip_info_5=examples[4]['trip_info'],
        transport_options_5=examples[4]['transport_options'],
        choice_5=examples[4]['choice'],
        trip_info_6=trip_info,
        transport_options_6=transport_options
    )
    return [
        {"role": "system", "content": PREDICTION_SYS_FEWSHOT},
        {"role": "user",   "content": content}
    ]

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
    for mode in CHOICES:
        if mode not in preds:
            raise ValueError(f"Missing mode '{mode}' in {preds}")
    total = sum(preds.values())
    if total <= 0:
        raise ValueError(f"Non‑positive sum: {preds}")
    return raw, {k: v/total for k, v in preds.items()}

def run_fewshot_predictions(
    input_csv: str,
    ground_truth_csv: str,
    output_csv: str,
    chatbot,
    choices: list[str],
    selected_memids: list[int],
):
    input_df = pd.read_csv(input_csv)             # prompt_multi_col.csv
    truth_df = pd.read_csv(ground_truth_csv)      # swissmetro_processed.csv

    truth_df['CHOICE'] = truth_df['CHOICE'].replace({0: 1})
    choice_map = {1: 0, 2: 1, 3: 2}  # 0=Train,1=Swissmetro,2=Car
    truth_df['choice_idx'] = truth_df['CHOICE'].map(choice_map)

    truth_filtered = (
        truth_df[truth_df['ID'].isin(selected_memids)]
        .drop_duplicates(subset=['ID','TRIP_ID'], keep='first')
        .reset_index(drop=True)
    )

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['ID','TRIP_ID','raw_response', *choices, 'actual_choice_idx']) \
          .to_csv(output_csv, index=False)
    existing = set(zip(*pd.read_csv(output_csv)[['ID','TRIP_ID']].to_numpy().T))

    for _, row in truth_filtered.iterrows():
        memid, trip_id = row['ID'], row['TRIP_ID']
        if (memid, trip_id) in existing:
            continue

        mem_recs = truth_filtered[truth_filtered['ID'] == memid]

        examples = []
        others = mem_recs[mem_recs['TRIP_ID'] != trip_id]
        for _, ex in others.head(5).iterrows():
            inp = input_df.loc[input_df['TRIP_ID'] == ex['TRIP_ID']].iloc[0]
            one_hot = {
                mode: 1.0 if idx == ex['choice_idx'] else 0.0
                for idx, mode in enumerate(choices)
            }
            examples.append({
                'trip_info': inp['trip_info'],
                'transport_options': inp['transport_options'],
                'choice': json.dumps(one_hot)
            })

        inp_target = input_df.loc[input_df['TRIP_ID'] == trip_id].iloc[0]
        messages = format_prompts_fewshot(
            examples,
            trip_info=inp_target['trip_info'],
            transport_options=inp_target['transport_options']
        )
        raw_resp, preds = call_llm(chatbot, messages)

        entry = {
            'ID': memid,
            'TRIP_ID': trip_id,
            'raw_response': raw_resp,
            **{mode: preds[mode] for mode in choices},
            'actual_choice_idx': row['choice_idx']
        }
        pd.DataFrame([entry]).to_csv(output_csv, mode='a', header=False, index=False)
        existing.add((memid, trip_id))

    all_df = pd.read_csv(output_csv)
    y_true = all_df['actual_choice_idx'].tolist()
    y_pred = all_df[choices].values.tolist()
    acc = accuracy_score(y_true, [p.index(max(p)) for p in y_pred])
    ce  = log_loss(y_true, y_pred, labels=list(range(len(choices))))

    print(f"Completed {len(all_df)} few‑shot predictions.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cross‑Entropy Loss: {ce:.4f}")


def run_fewshot_predictions_one_per_member(
    input_csv: str,
    ground_truth_csv: str,
    output_csv: str,
    chatbot,
    choices: list[str],
    selected_memids: list[int],
):
    input_df = pd.read_csv(input_csv)
    truth_df = pd.read_csv(ground_truth_csv)

    truth_df['CHOICE'] = truth_df['CHOICE'].replace({0: 1})
    choice_map = {1: 0, 2: 1, 3: 2}
    truth_df['choice_idx'] = truth_df['CHOICE'].map(choice_map)

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['ID','TRIP_ID','raw_response', *choices, 'actual_choice_idx']) \
          .to_csv(output_csv, index=False)
    existing = set(zip(*pd.read_csv(output_csv)[['ID','TRIP_ID']].to_numpy().T))

    for memid in selected_memids:
        mem_recs = truth_df[truth_df['ID'] == memid]
        if mem_recs.empty:
            continue

        target_row = mem_recs.iloc[0]
        trip_id = target_row['TRIP_ID']
        if (memid, trip_id) in existing:
            continue

        others = mem_recs[mem_recs['TRIP_ID'] != trip_id]
        examples = []
        for _, ex in others.head(5).iterrows():
            inp = input_df.loc[input_df['TRIP_ID'] == ex['TRIP_ID']].iloc[0]
            one_hot = {
                mode: 1.0 if idx == ex['choice_idx'] else 0.0
                for idx, mode in enumerate(choices)
            }
            examples.append({
                'trip_info': inp['trip_info'],
                'transport_options': inp['transport_options'],
                'choice': json.dumps(one_hot)
            })

        inp_t = input_df.loc[input_df['TRIP_ID'] == trip_id].iloc[0]
        messages = format_prompts_fewshot(
            examples,
            trip_info=inp_t['trip_info'],
            transport_options=inp_t['transport_options']
        )

        raw_resp, preds = call_llm(chatbot, messages)

        entry = {
            'ID': memid,
            'TRIP_ID': trip_id,
            'raw_response': raw_resp,
            **{mode: preds[mode] for mode in choices},
            'actual_choice_idx': target_row['choice_idx']
        }
        pd.DataFrame([entry]).to_csv(output_csv, mode='a', header=False, index=False)
        existing.add((memid, trip_id))

    all_df = pd.read_csv(output_csv)
    y_true = all_df['actual_choice_idx'].tolist()
    y_pred = all_df[choices].values.tolist()
    acc = accuracy_score(y_true, [p.index(max(p)) for p in y_pred])
    ce  = log_loss(y_true, y_pred, labels=list(range(len(choices))))

    print(f"Completed {len(all_df)} few‑shot predictions (one per member).")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cross‑Entropy Loss: {ce:.4f}")


if __name__ == "__main__":
    choices = CHOICES
    chatbot = ChatCompletionAPI(model="gpt-4o-mini")
    selected_memids = load_selected_pids(
        PROJECT_ROOT / "data" / "swissmetro" / "selected_pids_subset_100.txt"
    )

    input_csv = str(PROJECT_ROOT / "data" / "swissmetro" / "prompt_multi_col.csv")
    ground_truth_csv = str(PROJECT_ROOT / "data" / "swissmetro" / "swissmetro_processed.csv")
    output_csv = str(PROJECT_ROOT / "results" / "swissmetro_fewshot_results.csv")

    run_fewshot_predictions_one_per_member(
        input_csv=input_csv,
        ground_truth_csv=ground_truth_csv,
        output_csv=output_csv,
        chatbot=chatbot,
        choices=choices,
        selected_memids=selected_memids,
    )
