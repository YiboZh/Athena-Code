"""Text-layer optimisation utilities built around LLM interactions."""

from __future__ import annotations

import ast
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import textgrad as tg
from tenacity import retry, retry_if_exception_type, retry_if_result, stop_after_attempt
from textgrad.loss import MultiFieldEvaluation

from .chatbots import ChatbotAPI
from .logging_config import PACKAGE_ROOT
from .prompts import (
    ANALYZE_RESULTS,
    ANALYZE_RESULTS_SYS,
    ANALYZE_RESULTS_VAC,
    CHOICES,
    CHOICES_VAC,
    CROSS_OVER,
    CROSS_OVER_SYS,
    CROSS_OVER_SYS_VAC,
    CROSS_OVER_VAC,
    GEN_RANDOM_N,
    GEN_RANDOM_N2,
    GEN_RANDOM_N_SYS,
    GEN_RANDOM_N_SYS2,
    GEN_RANDOM_N_SYS2_VAC,
    IDENTIFY_FACTOR_RELATION,
    IDENTIFY_FACTOR_RELATION2,
    IDENTIFY_FACTOR_RELATION_SYS,
    IDENTIFY_FACTOR_RELATION_SYS2,
    IDENTIFY_FACTOR_RELATION_SYS2_VAC,
    INIT_CONCEPT_LIB,
    INIT_CONCEPT_LIB_VAC,
    OP_LIB,
    PERSONA,
    PERSONA_SYS,
    PERSONA_SYS_VAC,
    PERSONA_VAC,
    PREDICTION,
    PREDICTION_FEWSHOT,
    PREDICTION_P,
    PREDICTION_P_VAC,
    PREDICTION_SYS,
    PREDICTION_SYS_FEWSHOT,
    PREDICTION_SYS_POSSIBILITY,
    PREDICTION_SYS_VAC,
    PREDICTION_SYS_VAC_POSSIBILITY,
    PREDICTION_SYS_VAC_POSSIBILITY_FEWSHOT,
    PREDICTION_SYS_ZEROSHOT,
    PREDICTION_VAC,
    SELECTION_GUIDANCE,
    SELECTION_GUIDANCE_SYS,
    SELECTION_GUIDANCE_SYS_VAC,
    SW_VAR_LIB,
    SW_VAR_LIB_VAC,
    TEMPLATE_INDIVIDUAL,
    VARIABLE_MAP,
    VARIABLE_MAP_VAC,
)
from .textgrad_utils import get_engine_plus


logger = logging.getLogger(__name__)

RETRY_COUNT = 15
RESULT_FILE = PACKAGE_ROOT.parent / "results_vac_ablation" / "accuracy_results.csv"


class LossParseError(RuntimeError):
    """Raised when an evaluation loss cannot be coerced into a float."""


_LEADING_FLOAT_PATTERN = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _coerce_loss_to_float(value: Any) -> float:
    """Extract a numeric value from loss outputs that may contain text."""

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        match = _LEADING_FLOAT_PATTERN.search(value)
        if match:
            return float(match.group(1))

    raise LossParseError(f"Unable to parse loss value into float: {value!r}")


@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(LossParseError))
def _evaluate_loss_with_retry(
    loss_fn: Callable[[Sequence[tg.Variable]], tg.Variable],
    pred_var: tg.Variable,
    gt_var: tg.Variable,
) -> Tuple[tg.Variable, float]:
    """Evaluate the loss function with retries when parsing fails."""

    loss_var = loss_fn([pred_var, gt_var])
    loss_value = _coerce_loss_to_float(loss_var.value)
    return loss_var, loss_value


def _safe_format(template: str, *, context: Dict[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:
        missing_key = exc.args[0]
        available = ", ".join(sorted(context.keys())) or "<empty>"
        logger.error(
            "Prompt formatting failed. Missing key '%s'. Available keys: %s",
            missing_key,
            available,
        )
        raise ValueError(
            f"Prompt template expects key '{missing_key}'. Provided keys: {available}"
        ) from exc


def _ensure_code_block_list(result: str) -> List[Tuple[str, ...]]:
    """Parse list-like tuples returned by LLMs from flexible formats."""

    cleaned = re.sub(r"^\s*```(?:\w+)?\s*|\s*```$", "", result.strip(), flags=re.MULTILINE)
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("LLM response does not contain a list structure", result)

    payload = cleaned[start:]
    if payload.count("[") > payload.count("]"):
        payload += "]" * (payload.count("[") - payload.count("]"))

    payload = payload.lstrip("[").rstrip("]")
    tuples_raw = re.split(r"\)\s*,\s*\(", payload)

    parsed: List[Tuple[str, ...]] = []
    for entry in tuples_raw:
        entry = entry.strip().lstrip("(").rstrip(")")
        if not entry:
            continue
        parts = [part.strip().strip('\"\'') for part in entry.split(",")]
        parsed.append(tuple(parts))
    return parsed


def _extract_list_literal(result: str) -> List[str]:
    pattern = re.compile(r"```?\s*(\[[\s\S]*?])\s*```?", re.S)
    lists_raw = pattern.findall(result)
    if not lists_raw:
        raise ValueError("Expected a Python list literal in response", result)
    return ast.literal_eval(lists_raw[0])


def identify_factor_relation(chatbot: ChatbotAPI, features: str, knowledge: Sequence[str]) -> str:
    messages = [
        {"role": "system", "content": IDENTIFY_FACTOR_RELATION_SYS},
        {
            "role": "user",
            "content": IDENTIFY_FACTOR_RELATION.format(
                features=features,
                knowledge="\n".join(f"{i + 1}. {item}" for i, item in enumerate(knowledge)),
            ),
        },
    ]
    return chatbot.create(messages)


@retry(
    stop=stop_after_attempt(RETRY_COUNT),
    retry=retry_if_result(lambda result: len(result) == 0)
    | retry_if_exception_type((SyntaxError, IndexError, ValueError)),
)
def identify_factor_relation2(
    chatbot: ChatbotAPI,
    description: str,
    knowledge: Sequence[str],
    variables: Sequence[str],
    *,
    vaccine: bool = False,
) -> List[str]:
    system_prompt = IDENTIFY_FACTOR_RELATION_SYS2 if not vaccine else IDENTIFY_FACTOR_RELATION_SYS2_VAC
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": IDENTIFY_FACTOR_RELATION2.format(
                features=" ".join(variables),
                knowledge="\n".join(f"{i + 1}. {item}" for i, item in enumerate(knowledge)),
                description=description,
            ),
        },
    ]
    response = chatbot.create(messages)
    return _extract_list_literal(response)


@retry(
    stop=stop_after_attempt(RETRY_COUNT),
    retry=retry_if_result(lambda result: len(result) == 0)
    | retry_if_exception_type((SyntaxError, IndexError)),
)
def get_random_n_expressions(
    chatbot: ChatbotAPI,
    suggestions: str,
    n: int,
    variables: Sequence[str],
    operators: Sequence[str],
) -> List[Tuple[str, str]]:
    messages = [
        {
            "role": "system",
            "content": GEN_RANDOM_N_SYS.format(
                N=n, variables=", ".join(variables), operators=", ".join(operators)
            ),
        },
        {
            "role": "user",
            "content": GEN_RANDOM_N.format(suggestions=suggestions, N=n),
        },
    ]
    response = chatbot.create(messages)
    pattern = re.compile(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
    return pattern.findall(response)


@retry(
    stop=stop_after_attempt(RETRY_COUNT),
    retry=retry_if_result(lambda result: len(result) == 0)
    | retry_if_exception_type((SyntaxError, IndexError)),
)
def get_random_n_expressions2(
    chatbot: ChatbotAPI,
    suggestions: str,
    n: int,
    variables: Sequence[str],
    description: str,
    operators: Sequence[str],
    *,
    vaccine: bool = False,
) -> List[Tuple[str, str, str]]:
    system_prompt = GEN_RANDOM_N_SYS2 if not vaccine else GEN_RANDOM_N_SYS2_VAC
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                N=n,
                variables=", ".join(variables),
                operators=", ".join(operators),
                description=description,
            ),
        },
        {
            "role": "user",
            "content": GEN_RANDOM_N2.format(suggestions=suggestions),
        },
    ]
    response = chatbot.create(messages)
    pattern = re.compile(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
    return pattern.findall(response)


def concept_generation(
    chatbot: ChatbotAPI,
    expr: pd.DataFrame,
    description: str,
    n: int,
    *,
    vaccine: bool = False,
) -> str:
    message_template = ANALYZE_RESULTS if not vaccine else ANALYZE_RESULTS_VAC
    system_prompt = ANALYZE_RESULTS_SYS

    prediction_columns = (
        ("train", "car", "metro")
        if not vaccine
        else ("unvaccinated", "vaccinated_no_booster", "booster")
    )

    try:
        if vaccine:
            prompt_values: Dict[str, Any] = {
                "expr_unvax1": expr.iloc[0][prediction_columns[0]],
                "expr_vnb1": expr.iloc[0][prediction_columns[1]],
                "expr_booster1": expr.iloc[0][prediction_columns[2]],
                "expr_unvax2": expr.iloc[1][prediction_columns[0]],
                "expr_vnb2": expr.iloc[1][prediction_columns[1]],
                "expr_booster2": expr.iloc[1][prediction_columns[2]],
            }
        else:
            prompt_values = {
                "texpr1": expr.iloc[0][prediction_columns[0]],
                "cexpr1": expr.iloc[0][prediction_columns[1]],
                "mexpr1": expr.iloc[0][prediction_columns[2]],
                "texpr2": expr.iloc[1][prediction_columns[0]],
                "cexpr2": expr.iloc[1][prediction_columns[1]],
                "mexpr2": expr.iloc[1][prediction_columns[2]],
            }

        prompt_values.update(
            {
                "bexpr1": expr.iloc[2][prediction_columns[0]],
                "bexpr2": expr.iloc[2][prediction_columns[1]],
                "bexpr3": expr.iloc[2][prediction_columns[2]],
                "acc1": expr.iloc[0]["accuracy"],
                "acc2": expr.iloc[1]["accuracy"],
                "acc3": expr.iloc[2]["accuracy"],
                "description": description,
                "N": n,
            }
        )
    except (KeyError, IndexError) as exc:
        logger.error(
            "Failed to build prompt context for concept generation (vaccine=%s): %s",
            vaccine,
            exc,
        )
        raise ValueError(
            "Concept generation requires at least three expressions with expected columns"
        ) from exc

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _safe_format(message_template, context=prompt_values),
        },
    ]
    return chatbot.create(messages)


@retry(
    stop=stop_after_attempt(RETRY_COUNT),
    retry=retry_if_result(lambda result: len(result) == 0)
    | retry_if_exception_type(Exception),
)
def cross_over(
    chatbot: ChatbotAPI,
    expr: pd.DataFrame,
    suggestions: str,
    n: int,
    variables: Sequence[str],
    operators: Sequence[str],
    *,
    vaccine: bool = False,
) -> List[Tuple[str, ...]]:
    system_prompt = CROSS_OVER_SYS if not vaccine else CROSS_OVER_SYS_VAC
    user_prompt = CROSS_OVER if not vaccine else CROSS_OVER_VAC

    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                N=n, variables=", ".join(variables), operators=", ".join(operators)
            ),
        },
    ]

    try:
        expression_columns = (
            ("train", "car", "metro")
            if not vaccine
            else ("unvaccinated", "vaccinated_no_booster", "booster")
        )

        prompt_context: Dict[str, Any] = {"suggestions": suggestions, "N": n}
        if vaccine:
            prompt_context.update(
                {
                    "expr_unvax1": expr.iloc[0][expression_columns[0]],
                    "expr_vnb1": expr.iloc[0][expression_columns[1]],
                    "expr_booster1": expr.iloc[0][expression_columns[2]],
                    "expr_unvax2": expr.iloc[1][expression_columns[0]],
                    "expr_vnb2": expr.iloc[1][expression_columns[1]],
                    "expr_booster2": expr.iloc[1][expression_columns[2]],
                }
            )
        else:
            prompt_context.update(
                {
                    "texpr1": expr.iloc[0][expression_columns[0]],
                    "cexpr1": expr.iloc[0][expression_columns[1]],
                    "mexpr1": expr.iloc[0][expression_columns[2]],
                    "texpr2": expr.iloc[1][expression_columns[0]],
                    "cexpr2": expr.iloc[1][expression_columns[1]],
                    "mexpr2": expr.iloc[1][expression_columns[2]],
                }
            )
    except (IndexError, KeyError) as exc:
        logger.error(
            "Failed to build prompt context for cross-over (vaccine=%s): %s",
            vaccine,
            exc,
        )
        raise ValueError(
            "Cross-over requires at least two expressions with expected columns"
        ) from exc

    messages.append(
        {
            "role": "user",
            "content": _safe_format(user_prompt, context=prompt_context),
        }
    )

    response = chatbot.create(messages)
    logger.debug("Cross-over response: %s", response)
    return _ensure_code_block_list(response)


def select_template(
    chatbot: ChatbotAPI,
    demographics: str,
    utility_function: str,
    *,
    vaccine: bool = False,
) -> str:
    system_prompt = SELECTION_GUIDANCE_SYS if not vaccine else SELECTION_GUIDANCE_SYS_VAC
    user_prompt = SELECTION_GUIDANCE.format(demographics=demographics, utility=utility_function)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = chatbot.create(messages).strip()
    template_name = result.split()[0].upper()
    logger.debug("Selected template %s for demographics=%s", template_name, demographics)
    return template_name


def loading_persona(
    chatbot: ChatbotAPI,
    records: str | None,
    *,
    vaccine: bool = False,
    features: str | None = None,
    survey: str | None = None,
) -> str:
    system_prompt = PERSONA_SYS if not vaccine else PERSONA_SYS_VAC
    if vaccine:
        user_prompt = PERSONA_VAC.format(features=features or "", survey=survey or "")
    else:
        user_prompt = PERSONA.format(records=records or "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = chatbot.create(messages)
    logger.debug("Persona generated: %s", result)
    return result


def build_individual_block(demographics: str, trip_info: str) -> str:
    return TEMPLATE_INDIVIDUAL.format(demographics=demographics, trip_info=trip_info)


def _canonicalise_vaccine_label(text: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", text.lower())
    if "unvac" in normalized:
        return "Unvaccinated"
    if "noboost" in normalized or "nobooster" in normalized:
        return "Vaccinated_no_booster"
    if "boost" in normalized:
        return "Booster"
    return None


def _normalise_prediction_label(result: str, *, vaccine: bool) -> str:
    lowered = result.lower()
    if vaccine:
        canonical = _canonicalise_vaccine_label(result)
        if canonical:
            return canonical
        if "unvac" in lowered:
            return "Unvaccinated"
        if any(token in lowered for token in ("no booster", "no_booster", "nobooster", "no-booster", "no boo", "no-boost")):
            return "Vaccinated_no_booster"
        if "boost" in lowered:
            return "Booster"
        return "Unvaccinated"
    if "metro" in lowered:
        return "Swissmetro"
    if "car" in lowered:
        return "Car"
    if "train" in lowered:
        return "Train"
    return "Swissmetro"


def predict_mode(
    chatbot: ChatbotAPI,
    template_name: str,
    persona: str,
    individual_block: str,
    options: str,
    *,
    vaccine: bool = False,
    prediction_sys_msg: str | None = None,
    prediction_usr_msg: str | None = None,
) -> Tuple[str, str]:
    if vaccine:
        prediction_usr_msg = PREDICTION_P_VAC.format(
            template_name=template_name,
            individual_block=individual_block,
            options=options,
            persona=persona,
        )
        prediction_sys_msg = PREDICTION_SYS_VAC
    else:
        prediction_usr_msg = prediction_usr_msg or PREDICTION_P.format(
            template_name=template_name,
            individual_block=individual_block,
            options=options,
            persona=persona,
        )
        prediction_sys_msg = prediction_sys_msg or PREDICTION_SYS

    messages = [
        {"role": "system", "content": prediction_sys_msg},
        {"role": "user", "content": prediction_usr_msg},
    ]

    result = chatbot.create(messages)
    logger.debug("Prediction response: %s", result)
    label = _normalise_prediction_label(result, vaccine=vaccine)
    return label, result


@retry(stop=stop_after_attempt(5), retry=retry_if_exception_type((ValueError, json.JSONDecodeError)))
def predict_mode_possibility(
    chatbot: ChatbotAPI,
    template_name: str,
    persona: str,
    individual_block: str,
    options: str,
    *,
    vaccine: bool = False,
    prediction_sys_msg: str | None = None,
    prediction_usr_msg: str | None = None,
) -> Tuple[str, dict[str, float], str]:
    if vaccine:
        prediction_usr_msg = PREDICTION_P_VAC.format(
            template_name=template_name,
            individual_block=individual_block,
            options=options,
            persona=persona,
        )
        prediction_sys_msg = PREDICTION_SYS_VAC_POSSIBILITY
    else:
        prediction_usr_msg = prediction_usr_msg or PREDICTION_P.format(
            template_name=template_name,
            individual_block=individual_block,
            options=options,
            persona=persona,
        )
        prediction_sys_msg = prediction_sys_msg or PREDICTION_SYS_POSSIBILITY

    messages = [
        {"role": "system", "content": prediction_sys_msg},
        {"role": "user", "content": prediction_usr_msg},
    ]

    result = chatbot.create(messages)
    logger.debug("Probability prediction response: %s", result)

    match = re.search(r"\{[\s\S]*\}", result)
    if not match:
        raise ValueError(f"No JSON object found in response: {result}")
    preds = json.loads(match.group(0))

    mapped: dict[str, float] = {}
    for key, value in preds.items():
        base_key = key.lower()
        if vaccine:
            canonical = _canonicalise_vaccine_label(key)
            if canonical:
                mapped[canonical] = value
            else:
                if "unvac" in base_key:
                    mapped["Unvaccinated"] = value
                elif any(token in base_key for token in ("no booster", "no_booster", "nobooster", "no-booster", "no boo", "no-boost")):
                    mapped["Vaccinated_no_booster"] = value
                elif "boost" in base_key:
                    mapped["Booster"] = value
        else:
            if "metro" in base_key:
                mapped["Swissmetro"] = value
            elif "car" in base_key:
                mapped["Car"] = value
            elif "train" in base_key:
                mapped["Train"] = value

    expected_modes = CHOICES_VAC if vaccine else CHOICES
    for mode in expected_modes:
        if mode not in mapped:
            raise ValueError(f"Missing mode '{mode}' in {mapped}")

    total = sum(mapped.values())
    if total <= 0:
        raise ValueError(f"Non-positive probability sum: {mapped}")

    normalised = {mode: value / total for mode, value in mapped.items()}
    label = max(normalised, key=normalised.get)
    return label, normalised, result


def record_accuracy(
    group_id: int,
    member_id: int,
    trip_id: int,
    raw_response: str,
    probabilities: dict[str, float],
    predicted_mode: str,
    actual_mode: str,
    accuracy: bool,
    best_sys_prompt: str,
    best_template: str,
    loss_history: Sequence[float],
    *,
    result_file: Path | None = None,
) -> None:
    result_path = result_file or RESULT_FILE
    result_path.parent.mkdir(parents=True, exist_ok=True)

    accuracy_data = {
        "group_id": group_id,
        "member_id": member_id,
        "trip_id": trip_id,
        "raw_response": raw_response,
        "probabilities": json.dumps(probabilities),
        "predicted_mode": predicted_mode,
        "actual_mode": actual_mode,
        "accuracy": accuracy,
        "best_sys_prompt": best_sys_prompt,
        "best_usr_prompt": best_template,
        "loss_history": list(loss_history),
    }
    accuracy_df = pd.DataFrame([accuracy_data])

    exists = result_path.exists()
    if exists:
        existing_header = pd.read_csv(result_path, nrows=0)
        existing_cols = existing_header.columns.tolist()
        new_cols = accuracy_df.columns.tolist()
        if existing_cols != new_cols:
            existing_df = pd.read_csv(result_path)
            combined_df = pd.concat([existing_df, accuracy_df], ignore_index=True)
            combined_df.to_csv(result_path, index=False)
        else:
            accuracy_df.to_csv(result_path, mode="a", header=False, index=False)
    else:
        accuracy_df.to_csv(result_path, index=False)


def _build_evaluator_instruction() -> tg.Variable:
    return tg.Variable(
        "Evaluate the travel mode prediction based on the individual's profile and alternatives. "
        "Compare it to the actual choice and identify any discrepancies. Be concise and focus on why the prediction might be incorrect."
        "Return 0 if they match, 1 otherwise.",
        requires_grad=False,
        role_description="evaluation instruction",
    )


def optimize_prompts(
    initial_sys_prompt: str,
    optimize_sys: bool,
    user_msg: str,
    ground_truth_choice: str,
    *,
    steps: int = 80,
    model_eval: str = "gpt-4o",
    model_pred: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str, List[float]]:
    sys_prompt_var = tg.Variable(
        initial_sys_prompt,
        requires_grad=optimize_sys,
        role_description="system prompt guiding the decision assistant",
    )
    user_msg_var = tg.Variable(
        user_msg,
        requires_grad=optimize_sys,
        role_description="complete <TEMPLATE>/<PROFILE>/<ALTERNATIVES> block sent as user message",
    )

    model_pred = model_pred or model_eval
    eval_engine = get_engine_plus(engine_name=model_eval)
    pred_engine = get_engine_plus(engine_name=model_pred)
    tg.set_backward_engine(eval_engine, override=True)
    predictor = tg.BlackboxLLM(pred_engine, sys_prompt_var)

    eval_instr = _build_evaluator_instruction()
    loss_fn = MultiFieldEvaluation(
        evaluation_instruction=eval_instr,
        role_descriptions=["prediction", "ground truth"],
    )

    parameters = [sys_prompt_var, user_msg_var] if optimize_sys else []
    optimizer = tg.TextualGradientDescent(engine=eval_engine, parameters=parameters)

    best_loss = float("inf")
    best_sys_prompt = sys_prompt_var.get_value()
    best_user_msg = user_msg_var.get_value()
    loss_history: List[float] = []

    for step in range(steps):
        optimizer.zero_grad()
        y_var = tg.Variable(ground_truth_choice, requires_grad=False, role_description="ground truth choice")
        pred_var = predictor(user_msg_var)
        loss_var = loss_fn([pred_var, y_var])

        try:
            loss_val = float(loss_var.value)
        except (TypeError, ValueError):
            loss_val = float("inf")

        loss_history.append(loss_val)
        loss_var.backward()
        optimizer.step()

        if loss_val < best_loss:
            best_loss = loss_val
            best_sys_prompt = sys_prompt_var.get_value()
            best_user_msg = user_msg_var.get_value()

        if verbose and (step % 10 == 0 or step == steps - 1):
            logger.info("Prompt optimisation step %d, loss=%s", step, loss_val)

    return best_sys_prompt, best_user_msg, loss_history


def optimize_partial_prompts(
    chatbot: ChatbotAPI,
    initial_persona: str,
    initial_template_name: str,
    profiles: Sequence[str],
    alternatives: Sequence[str],
    ground_truth_choices: Sequence[str],
    *,
    val_profiles: Sequence[str] | None = None,
    val_alternatives: Sequence[str] | None = None,
    val_ground_truth_choices: Sequence[str] | None = None,
    steps: int = 80,
    model_eval: str = "gpt-4o",
    model_pred: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str, List[float]]:
    persona_var = tg.Variable(initial_persona, requires_grad=True, role_description="optimisable persona text")
    template_var = tg.Variable(initial_template_name, requires_grad=True, role_description="optimisable choice template formula")

    model_pred = model_pred or model_eval
    eval_engine = get_engine_plus(model_eval)
    pred_engine = get_engine_plus(model_pred)
    tg.set_backward_engine(eval_engine, override=True)
    sys_prompt_const = tg.Variable(PREDICTION_SYS, requires_grad=False, role_description="system prompt")
    predictor = tg.BlackboxLLM(pred_engine, sys_prompt_const)

    eval_instr = _build_evaluator_instruction()
    loss_fn = MultiFieldEvaluation(
        evaluation_instruction=eval_instr,
        role_descriptions=["prediction", "ground truth"],
    )

    optimizer = tg.TextualGradientDescent(engine=eval_engine, parameters=[persona_var, template_var])

    best_persona = persona_var.get_value()
    best_template = template_var.get_value()
    best_metric = 0
    loss_history: List[float] = []

    for step in range(steps):
        optimizer.zero_grad()
        batch_losses = []

        for profile, alternative, gt in zip(profiles, alternatives, ground_truth_choices):
            msg_var = persona_var + template_var + tg.Variable(
                f"<PROFILE>\n{profile}\n</PROFILE>\n<ALTERNATIVES>\n{alternative}\n</ALTERNATIVES>",
                requires_grad=False,
                role_description="travel profile and alternatives",
            )
            pred_var = predictor(msg_var)
            gt_var = tg.Variable(gt, requires_grad=False, role_description="ground truth choice")
            batch_losses.append(loss_fn([pred_var, gt_var]))

        total_loss = tg.sum(batch_losses)
        loss_history.append(float(total_loss.value))
        total_loss.backward()
        optimizer.step()

        correct = 0
        if val_profiles and val_alternatives and val_ground_truth_choices:
            for prof, alt, gt in zip(val_profiles, val_alternatives, val_ground_truth_choices):
                mode_prediction, _ = predict_mode(
                    chatbot,
                    template_var.get_value(),
                    persona_var.get_value(),
                    prof,
                    alt,
                )
                if mode_prediction == gt:
                    correct += 1

        if correct >= best_metric:
            best_metric = correct
            best_persona = persona_var.get_value()
            best_template = template_var.get_value()

        if verbose and (step % 10 == 0 or step == steps - 1):
            logger.info(
                "Partial prompt step %d, train_loss=%.4f, val_correct=%s",
                step,
                float(total_loss.value),
                correct,
            )

    return best_persona, best_template, loss_history


def optimize_vaccine_prompts(
    chatbot: ChatbotAPI,
    initial_template_name: str,
    alternatives: str,
    profiles: Sequence[str],
    ground_truth_choices: Sequence[str],
    persona_profiles: Sequence[str],
    *,
    val_profiles: Sequence[str] | None = None,
    val_ground_truth_choices: Sequence[str] | None = None,
    val_persona_profiles: Sequence[str] | None = None,
    steps: int = 80,
    model_eval: str = "gpt-4o",
    model_pred: str | None = None,
    verbose: bool = True,
) -> Tuple[str, List[float]]:
    template_var = tg.Variable(initial_template_name, requires_grad=True, role_description="optimisable choice template formula")

    model_pred = model_pred or model_eval
    eval_engine = get_engine_plus(model_eval)
    pred_engine = get_engine_plus(model_pred)
    tg.set_backward_engine(eval_engine, override=True)
    sys_prompt_const = tg.Variable(PREDICTION_SYS_VAC, requires_grad=False, role_description="system prompt")
    predictor = tg.BlackboxLLM(pred_engine, sys_prompt_const)

    eval_instr = tg.Variable(
        "Evaluate the vaccine choice prediction based on the individual's profile and alternatives. Compare it to the actual choice and identify any discrepancies. Be concise and focus on why the prediction might be incorrect. Return 0 if they match, 1 otherwise.",
        requires_grad=False,
        role_description="evaluation instruction",
    )
    loss_fn = MultiFieldEvaluation(
        evaluation_instruction=eval_instr,
        role_descriptions=["prediction", "ground truth"],
    )

    optimizer = tg.TextualGradientDescent(engine=eval_engine, parameters=[template_var])

    best_template = template_var.get_value()
    best_metric = 0
    loss_history: List[float] = []

    for step in range(steps):
        optimizer.zero_grad()
        batch_losses = []
        batch_loss_values: List[float] = []

        for profile, gt, persona in zip(profiles, ground_truth_choices, persona_profiles):
            msg_var = template_var + tg.Variable(
                f"<PERSONA>{persona}</PERSONA>\n<PROFILE>\n{profile}\n</PROFILE><KNOWLEDGE>{','.join(INIT_CONCEPT_LIB_VAC)}</KNOWLEDGE>\n<ALTERNATIVES>\n{alternatives}\n</ALTERNATIVES>",
                requires_grad=False,
                role_description="vaccination profile and alternatives",
            )
            pred_var = predictor(msg_var)
            gt_var = tg.Variable(gt, requires_grad=False, role_description="ground truth choice")
            loss_var, loss_value = _evaluate_loss_with_retry(loss_fn, pred_var, gt_var)
            batch_losses.append(loss_var)
            batch_loss_values.append(loss_value)

        total_loss = tg.sum(batch_losses)
        numeric_total_loss = float(sum(batch_loss_values))
        loss_history.append(numeric_total_loss)
        total_loss.backward()
        optimizer.step()

        correct = 0
        if val_profiles and val_ground_truth_choices and val_persona_profiles:
            for prof, gt, persona in zip(val_profiles, val_ground_truth_choices, val_persona_profiles):
                mode_prediction, _ = predict_mode(
                    chatbot,
                    template_var.get_value(),
                    persona,
                    prof,
                    alternatives,
                    vaccine=True,
                )
                if mode_prediction == gt:
                    correct += 1

        if correct >= best_metric:
            best_metric = correct
            best_template = template_var.get_value()

        if verbose and (step % 10 == 0 or step == steps - 1):
            logger.info(
                "Vaccine prompt step %d, train_loss=%.4f, val_correct=%s",
                step,
                numeric_total_loss,
                correct,
            )

    return best_template, loss_history


def split_trips(
    trip_ids: Sequence[int],
    df: pd.DataFrame,
    *,
    train_n: int = 5,
    test_n: int = 0,
    val_n: int = 3,
    choices: Sequence[str] | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    choice_labels = list(choices) if choices is not None else CHOICES
    buckets: dict[str, List[int]] = defaultdict(list)
    for tid in trip_ids:
        choice_idx = int(df.iloc[tid]["choice_idx"])
        try:
            choice = choice_labels[choice_idx]
        except IndexError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Choice index {choice_idx} is out of range for choices {choice_labels}") from exc
        buckets[choice].append(tid)

    train_ids: List[int] = []
    for _, tids in sorted(buckets.items(), key=lambda kv: len(kv[1])):
        if tids and len(train_ids) < train_n:
            train_ids.append(tids.pop(0))

    remaining = [tid for tids in buckets.values() for tid in tids]
    random.shuffle(remaining)
    train_ids.extend(remaining[: max(0, train_n - len(train_ids))])
    remaining = remaining[max(0, train_n - len(train_ids)) :]

    test_ids = remaining[:test_n]
    val_ids = remaining[test_n : test_n + val_n]
    return train_ids, test_ids, val_ids


__all__ = [
    "identify_factor_relation",
    "identify_factor_relation2",
    "get_random_n_expressions",
    "get_random_n_expressions2",
    "concept_generation",
    "cross_over",
    "select_template",
    "loading_persona",
    "build_individual_block",
    "predict_mode",
    "predict_mode_possibility",
    "record_accuracy",
    "optimize_prompts",
    "optimize_partial_prompts",
    "optimize_vaccine_prompts",
    "split_trips",
    "RESULT_FILE",
]
