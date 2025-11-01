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
from athena.prompts import CHOICES_VAC, PREDICTION_SYS_VAC_POSSIBILITY_FEWSHOT

main_logger = setup_logging()

# ------------ new: few-shot user message construction ------------
def format_vac_fewshot_prompts(
    shots: list[dict],
    template: str,
    profile: str,
    alternatives: str
) -> list[dict]:
    """
    shots: list of {
        'template': str,
        'profile': str,
        'alternatives': str,
        'output_json': str
    }
    Returns:
    [
      {"role": "system", "content": PREDICTION_SYS_VAC_POSSIBILITY},
      {"role": "user", "content": <all examples plus the target case>}
    ]
    """
    # First pack each shot into a block
    blocks = []
    for i, shot in enumerate(shots, start=1):
        blocks += [
            "<TEMPLATE>",
            shot['template'],
            "</TEMPLATE>",
            "<PROFILE>",
            shot['profile'],
            "</PROFILE>",
            "<ALTERNATIVES>",
            shot['alternatives'],
            "</ALTERNATIVES>",
            "<CHOICE>",
            shot['output_json'].strip(),
            "</CHOICE>"
        ]
    # Then append the target case
    blocks += [
        "<TEMPLATE>",
        template,
        "</TEMPLATE>",
        "<PROFILE>",
        profile,
        "</PROFILE>",
        "<ALTERNATIVES>",
        alternatives,
        "</ALTERNATIVES>",
        "<CHOICE>",
        "Please predict the probabilities for each vaccination alternative in JSON format.",
        "</CHOICE>"
    ]
    user_content = "\n".join(blocks)
    return [
        {"role": "system", "content": PREDICTION_SYS_VAC_POSSIBILITY_FEWSHOT},
        {"role": "user", "content": user_content}
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


# LLM call and normalization (same as zero-shot)
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

# ------------ few-shot main flow ------------
def run_vaccine_few_shot(
    input_csv: str,
    cluster_csv: str,
    proc_prompts_csv: str,
    output_csv: str,
    chatbot,
    selected_memids: list,
    choices: list
):
    # 1. Read the main dataset
    df = pd.read_csv(input_csv)
    df_filtered = df[df['No.'].isin(selected_memids)].drop_duplicates(subset='No.', keep='first')

    # 2. Load the cluster table and processed prompts
    cluster_df = pd.read_csv(cluster_csv)  # Columns include No., cluster
    proc_df = pd.read_csv(proc_prompts_csv)  # Columns include No., demographic_prompt, other_prompt, persona, target

    # 3. Prepare the output file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['No.', *choices, 'raw_response']).to_csv(output_csv, index=False)
    existing = set(pd.read_csv(output_csv)['No.'].tolist())

    # 4. Iterate row by row
    for _, row in df_filtered.iterrows():
        no = row['No.']
        if no in existing:
            continue

        # Prompt for the current case
        template = proc_df.loc[proc_df['No.'] == no, 'persona'].iloc[0]
        profile = (
            proc_df.loc[proc_df['No.'] == no, 'demographic_prompt'].iloc[0] + " " +
            proc_df.loc[proc_df['No.'] == no, 'other_prompt'].iloc[0]
        )
        alternatives = ", ".join(choices)

        # 5. Build few-shot examples
        this_cluster = cluster_df.loc[cluster_df['No.'] == no, 'cluster'].iloc[0]
        peers = cluster_df[cluster_df['cluster'] == this_cluster]['No.'].tolist()
        peers = [p for p in peers if p != no][:5]  # Take up to 5 entries
        shots = []
        for ex_no in peers:
            ex_row = proc_df.loc[proc_df['No.'] == ex_no].iloc[0]
            ex_tpl = ex_row['persona']
            ex_prof = ex_row['demographic_prompt'] + " " + ex_row['other_prompt']
            ex_alt = ", ".join(choices)
            tgt = int(ex_row['target'])
            onehot = {cls: (1.0 if idx == tgt else 0.0)
                      for idx, cls in enumerate(choices)}
            shots.append({
                'template': ex_tpl,
                'profile': ex_prof,
                'alternatives': ex_alt,
                'output_json': json.dumps(onehot, indent=2)
            })

        # 6. Assemble messages and call the LLM
        messages = format_vac_fewshot_prompts(shots, template, profile, alternatives)
        raw_resp, preds = call_llm(chatbot, messages)

        # 7. Write the result
        entry = {'No.': no, **preds, 'raw_response': raw_resp}
        pd.DataFrame([entry]).to_csv(output_csv, mode='a', header=False, index=False)
        existing.add(no)

    # 8. Compute and print metrics
    out_df = pd.read_csv(output_csv)
    y_true = df_filtered.set_index('No.').loc[out_df['No.'], 'target'].tolist()
    y_pred = out_df[choices].values.tolist()
    acc = accuracy_score(y_true, [p.index(max(p)) for p in y_pred])
    ce = log_loss(y_true, y_pred, labels=list(range(len(choices))))
    print(f"Completed {len(out_df)} predictions.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cross-Entropy Loss: {ce:.4f}")

if __name__ == "__main__":
    # chatbot = GeminiAPI(model="gemini-2.0-flash")
    chatbot = ChatCompletionAPI(model="gpt-4o-mini")
    selected_memids = load_selected_pids(
        PROJECT_ROOT / "data" / "vaccine" / "selected_pids_subset_100.txt"
    )
    choices = CHOICES_VAC

    run_vaccine_few_shot(
        input_csv=str(PROJECT_ROOT / "data" / "vaccine" / "processed_prompts_descriptive_3_new.csv"),
        cluster_csv=str(PROJECT_ROOT / "data" / "vaccine" / "df_cluster_aei.csv"),
        proc_prompts_csv=str(PROJECT_ROOT / "data" / "vaccine" / "processed_prompts_descriptive_3_new.csv"),
        output_csv=str(PROJECT_ROOT / "results_vac" / "vaccine_fewshot_results.csv"),
        chatbot=chatbot,
        selected_memids=selected_memids,
        choices=choices,
    )
