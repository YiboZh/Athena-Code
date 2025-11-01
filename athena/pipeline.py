"""High-level orchestration for the Athena workflows."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from .chatbots import ChatbotAPI, OllamaAPI
from .datasets import DATA_ROOT, SwissMetro, Vaccines, load_selected_pids
from .logging_config import setup_logging
from .numeric import evaluate_utility_function
from .prompts import (
    INIT_CONCEPT_LIB,
    INIT_CONCEPT_LIB_VAC,
    OP_LIB,
    SW_VAR_LIB,
    SW_VAR_LIB_VAC,
    VARIABLE_MAP_VAC,
)
from .textual import (
    CHOICES,
    CHOICES_VAC,
    build_individual_block,
    concept_generation,
    cross_over,
    get_random_n_expressions2,
    identify_factor_relation2,
    loading_persona,
    optimize_partial_prompts,
    optimize_vaccine_prompts,
    predict_mode,
    predict_mode_possibility,
    record_accuracy,
    select_template,
    split_trips,
)


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results-deepseek")
DEFAULT_RESULTS_DIR_VAC = Path("results-deepseek-vac")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _initialise_chatbot(chatbot: ChatbotAPI | None) -> ChatbotAPI:
    if chatbot is not None:
        return chatbot
    logger.info("Initialising default Ollama client (deepseek-r1:32b).")
    return OllamaAPI(model="deepseek-r1:32b")


def _load_persona_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["member_id", "persona"])


def _resume_iterations(results_dir: Path, group_id: int, vaccine: bool = False) -> tuple[List[tuple], int]:
    pattern = results_dir.glob(f"utility_functions_results_group_{group_id}_iteration_*.csv")
    existing = sorted(pattern)
    if not existing:
        return [], 0

    iterations = []
    for file in existing:
        try:
            iterations.append(int(file.stem.split("iteration_")[-1]))
        except ValueError:
            continue

    if not iterations:
        return [], 0

    latest = max(iterations)
    latest_file = results_dir / f"utility_functions_results_group_{group_id}_iteration_{latest}.csv"
    df = pd.read_csv(latest_file)
    if vaccine:
        cols = [
            "unvaccinated",
            "vaccinated_no_booster",
            "booster",
            "accuracy",
            "prec",
            "rec",
            "f1",
            "auc",
            "ce",
            "iteration",
        ]
    else:
        cols = [
            "train",
            "car",
            "metro",
            "accuracy",
            "prec",
            "rec",
            "f1",
            "auc",
            "ce",
            "iteration",
        ]
    rows = df[cols].values.tolist()
    return [tuple(row) for row in rows], latest + 1


def _evaluate_candidates(
    dataset_df: pd.DataFrame,
    trip_ids: Sequence[int],
    group_id: int,
    expressions: Iterable[Sequence[str]],
    iteration_index: int,
    *,
    vaccine: bool = False,
) -> List[tuple]:
    evaluated: List[tuple] = []
    group_df = dataset_df.iloc[trip_ids]
    for item in expressions:
        try:
            if vaccine:
                metrics = evaluate_utility_function(
                    group_df,
                    group_id,
                    item,
                    variable_map=VARIABLE_MAP_VAC,
                )
            else:
                metrics = evaluate_utility_function(group_df, group_id, item)
            if metrics is None:
                continue
            if vaccine:
                no, basic, booster, acc, prec, rec, f1, auc, ce = metrics
                evaluated.append((no, basic, booster, acc, prec, rec, f1, auc, ce, iteration_index))
            else:
                train, car, metro, acc, prec, rec, f1, auc, ce = metrics
                evaluated.append((train, car, metro, acc, prec, rec, f1, auc, ce, iteration_index))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Group %s evaluation error: %s\n%s", group_id, exc, traceback.format_exc())
    return evaluated


def run_travel_mode(
    *,
    chatbot: ChatbotAPI | None = None,
    results_dir: Path | None = None,
    persona_csv: Path | None = None,
    selected_pids_file: Path | None = None,
    iterations: int = 30,
    top_k: int = 2,
    bottom_k: int = 1,
    n_candidates: int = 5,
    tg_steps: int = 5,
) -> None:
    """Run the SwissMetro Athena workflow."""

    setup_logging("athena.pipeline")
    bot = _initialise_chatbot(chatbot)

    results_path = _ensure_dir(Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR)
    persona_path = Path(persona_csv) if persona_csv else results_path / "persona.csv"
    persona_df = _load_persona_frame(persona_path)
    processed_memids = persona_df["member_id"].tolist()

    selection_path = (
        Path(selected_pids_file)
        if selected_pids_file
        else DATA_ROOT / "swissmetro" / "selected_pids_subset_100.txt"
    )
    selected_memids = load_selected_pids(selection_path)

    dataset = SwissMetro()

    for group in dataset.groups:
        logger.info(
            "SwissMetro group %s: %s (members=%s, trips=%s)",
            group.gid,
            group.description,
            len(group.members),
            len(group.get_all_trip_ids()),
        )

        group.concept_library = INIT_CONCEPT_LIB
        utility_results: List[tuple] = []

        final_file = results_path / f"utility_functions_results_group_{group.gid}_iteration_{iterations - 1}.csv"
        skip_utility = final_file.exists()
        start_iteration = 0

        if skip_utility:
            logger.info("Group %s already has final iteration; skipping optimisation.", group.gid)
        else:
            utility_results, start_iteration = _resume_iterations(results_path, group.gid)
            if start_iteration:
                logger.info(
                    "Group %s resuming from iteration %s (loaded %s candidates).",
                    group.gid,
                    start_iteration,
                    len(utility_results),
                )

        if not skip_utility:
            if start_iteration == 0:
                while len(utility_results) < n_candidates:
                    suggestions = identify_factor_relation2(
                        bot,
                        group.description,
                        group.concept_library,
                        SW_VAR_LIB,
                    )
                    candidates = get_random_n_expressions2(
                        bot,
                        " ".join(suggestions),
                        n_candidates,
                        SW_VAR_LIB,
                        group.description,
                        OP_LIB,
                    )

                    evaluated = _evaluate_candidates(
                        dataset.df,
                        group.get_all_trip_ids(),
                        group.gid,
                        candidates,
                        0,
                    )
                    utility_results.extend(evaluated)

            for iteration_index in range(start_iteration, iterations):
                if not utility_results:
                    break

                columns = [
                    "train",
                    "car",
                    "metro",
                    "accuracy",
                    "prec",
                    "rec",
                    "f1",
                    "auc",
                    "ce",
                    "iteration",
                ]
                utility_df = (
                    pd.DataFrame(utility_results, columns=columns)
                    .sort_values(by=["accuracy", "iteration"], ascending=[False, False])
                )
                utility_df.to_csv(
                    results_path / f"utility_functions_results_group_{group.gid}_iteration_{iteration_index}.csv",
                    index=False,
                )

                if len(utility_results) < top_k + bottom_k:
                    logger.warning("Group %s has insufficient candidates for crossover.", group.gid)
                    continue

                top_df = utility_df.head(top_k)
                bottom_df = utility_df.tail(bottom_k)
                concept = concept_generation(
                    bot,
                    pd.concat([top_df, bottom_df]),
                    group.description,
                    n_candidates,
                )

                new_results: List[tuple] = []
                while len(new_results) < n_candidates:
                    crossover_items = cross_over(
                        bot,
                        top_df,
                        concept,
                        n_candidates,
                        SW_VAR_LIB,
                        OP_LIB,
                    )
                    evaluated = _evaluate_candidates(
                        dataset.df,
                        group.get_all_trip_ids(),
                        group.gid,
                        crossover_items,
                        iteration_index + 1,
                    )
                    new_results.extend(evaluated)

                utility_results.extend(new_results)

            columns = [
                "train",
                "car",
                "metro",
                "accuracy",
                "prec",
                "rec",
                "f1",
                "auc",
                "ce",
                "iteration",
            ]
            utility_df = (
                pd.DataFrame(utility_results, columns=columns)
                .sort_values(by=["accuracy", "iteration"], ascending=[False, False])
            )
        else:
            utility_df = pd.read_csv(final_file)

        utility_df.to_csv(final_file, index=False)
        best_utility = utility_df.iloc[0]
        template_name = select_template(
            bot,
            group.description,
            f"train: {best_utility['train']}, car: {best_utility['car']}, metro: {best_utility['metro']}",
        )
        logger.info(
            "Group %s best accuracy %.4f with template %s",
            group.gid,
            best_utility["accuracy"],
            template_name,
        )

        for member in group.members:
            if int(member.pid) in processed_memids:
                continue
            if int(member.pid) not in selected_memids:
                continue

            try:
                travel_history_str = "\n".join(
                    [
                        f"Trip {i + 1}: {dataset.trip_info[member.trip_ids[i]]}, Choice: {CHOICES[int(dataset.df.iloc[member.trip_ids[i]]['choice_idx'])]}"
                        for i in range(len(member.trip_ids))
                    ]
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Member %s travel history error: %s", member.pid, exc)
                continue

            persona_rows = persona_df.loc[persona_df["member_id"] == member.pid, "persona"]
            if persona_rows.empty:
                persona = loading_persona(bot, travel_history_str)
                persona_df = pd.concat(
                    [
                        persona_df,
                        pd.DataFrame({"member_id": [member.pid], "persona": [persona]}),
                    ],
                    ignore_index=True,
                )
                persona_df.to_csv(persona_path, index=False)
            else:
                persona = persona_rows.iloc[0]

            train_profiles: List[str] = []
            test_profiles: List[str] = []
            val_profiles: List[str] = []
            train_alternatives: List[str] = []
            test_alternatives: List[str] = []
            val_alternatives: List[str] = []
            train_gts: List[str] = []
            test_gts: List[str] = []
            val_gts: List[str] = []

            train_ids, test_ids, val_ids = split_trips(
                member.trip_ids,
                dataset.df,
                train_n=5,
                test_n=2,
                val_n=2,
                choices=CHOICES,
            )

            for tid in train_ids:
                train_profiles.append(build_individual_block(group.description, dataset.trip_info[tid]))
                train_alternatives.append(dataset.transport_options[tid])
                train_gts.append(CHOICES[int(dataset.df.iloc[tid]['choice_idx'])])

            for tid in test_ids:
                test_profiles.append(build_individual_block(group.description, dataset.trip_info[tid]))
                test_alternatives.append(dataset.transport_options[tid])
                test_gts.append(CHOICES[int(dataset.df.iloc[tid]['choice_idx'])])

            for tid in val_ids:
                val_profiles.append(build_individual_block(group.description, dataset.trip_info[tid]))
                val_alternatives.append(dataset.transport_options[tid])
                val_gts.append(CHOICES[int(dataset.df.iloc[tid]['choice_idx'])])

            best_persona, best_template, loss_history = optimize_partial_prompts(
                bot,
                persona,
                template_name,
                train_profiles,
                train_alternatives,
                train_gts,
                val_profiles=val_profiles,
                val_alternatives=val_alternatives,
                val_ground_truth_choices=val_gts,
                steps=tg_steps,
            )

            for profile, alternative, gt, trip_id in zip(val_profiles, val_alternatives, val_gts, val_ids):
                predicted_mode, _ = predict_mode(
                    bot, best_template, best_persona, profile, alternative
                )
                _, probabilities, raw_response = predict_mode_possibility(
                    bot,
                    best_template,
                    best_persona,
                    profile,
                    alternative,
                )
                record_accuracy(
                    group.gid,
                    member.pid,
                    trip_id,
                    raw_response,
                    probabilities,
                    predicted_mode,
                    gt,
                    predicted_mode == gt,
                    best_persona,
                    best_template,
                    loss_history,
                    result_file=results_path / "accuracy_results.csv",
                )

            processed_memids.append(int(member.pid))


def run_vaccine(
    *,
    chatbot: ChatbotAPI | None = None,
    results_dir: Path | None = None,
    selected_pids_file: Path | None = None,
    iterations: int = 30,
    top_k: int = 2,
    bottom_k: int = 1,
    n_candidates: int = 5,
    tg_steps: int = 5,
) -> None:
    """Run the vaccination Athena workflow."""

    setup_logging("athena.pipeline")
    bot = _initialise_chatbot(chatbot)

    results_path = _ensure_dir(Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR_VAC)
    selection_path = (
        Path(selected_pids_file)
        if selected_pids_file
        else DATA_ROOT / "vaccine" / "selected_pids_subset_100.txt"
    )
    selected_memids = load_selected_pids(selection_path)

    dataset = Vaccines()

    for group in dataset.groups:
        logger.info("Vaccine group %s: %s", group.gid, group.description)

        group.concept_library = INIT_CONCEPT_LIB_VAC
        utility_results: List[tuple] = []

        final_file = results_path / f"utility_functions_results_group_{group.gid}_iteration_{iterations - 1}.csv"
        skip_utility = final_file.exists()
        start_iteration = 0

        if skip_utility:
            logger.info("Group %s already optimised; skipping search.", group.gid)
        else:
            utility_results, start_iteration = _resume_iterations(results_path, group.gid, vaccine=True)

        if not skip_utility:
            if start_iteration == 0:
                while len(utility_results) < n_candidates:
                    suggestions = identify_factor_relation2(
                        bot,
                        group.description,
                        group.concept_library,
                        SW_VAR_LIB_VAC,
                        vaccine=True,
                    )
                    candidates = get_random_n_expressions2(
                        bot,
                        " ".join(suggestions),
                        n_candidates,
                        SW_VAR_LIB_VAC,
                        group.description,
                        OP_LIB,
                        vaccine=True,
                    )
                    evaluated = _evaluate_candidates(
                        dataset.df,
                        group.get_all_trip_ids(),
                        group.gid,
                        candidates,
                        0,
                        vaccine=True,
                    )
                    utility_results.extend(evaluated)

            for iteration_index in range(start_iteration, iterations):
                if not utility_results:
                    break

                columns = [
                    "unvaccinated",
                    "vaccinated_no_booster",
                    "booster",
                    "accuracy",
                    "prec",
                    "rec",
                    "f1",
                    "auc",
                    "ce",
                    "iteration",
                ]
                utility_df = (
                    pd.DataFrame(utility_results, columns=columns)
                    .sort_values(by=["accuracy", "iteration"], ascending=[False, False])
                )
                utility_df.to_csv(
                    results_path / f"utility_functions_results_group_{group.gid}_iteration_{iteration_index}.csv",
                    index=False,
                )

                if len(utility_results) < top_k + bottom_k:
                    logger.warning("Vaccine group %s lacks candidates for crossover.", group.gid)
                    continue

                top_df = utility_df.head(top_k)
                bottom_df = utility_df.tail(bottom_k)
                concept = concept_generation(
                    bot,
                    pd.concat([top_df, bottom_df]),
                    group.description,
                    n_candidates,
                    vaccine=True,
                )

                new_results: List[tuple] = []
                while len(new_results) < n_candidates:
                    crossover_items = cross_over(
                        bot,
                        top_df,
                        concept,
                        n_candidates,
                        SW_VAR_LIB_VAC,
                        OP_LIB,
                        vaccine=True,
                    )
                    evaluated = _evaluate_candidates(
                        dataset.df,
                        group.get_all_trip_ids(),
                        group.gid,
                        crossover_items,
                        iteration_index + 1,
                        vaccine=True,
                    )
                    new_results.extend(evaluated)

                utility_results.extend(new_results)

            columns = [
                "unvaccinated",
                "vaccinated_no_booster",
                "booster",
                "accuracy",
                "prec",
                "rec",
                "f1",
                "auc",
                "ce",
                "iteration",
            ]
            utility_df = (
                pd.DataFrame(utility_results, columns=columns)
                .sort_values(by=["accuracy", "iteration"], ascending=[False, False])
            )
        else:
            utility_df = pd.read_csv(final_file)

        utility_df.to_csv(final_file, index=False)
        best_utility = utility_df.iloc[0]
        template_name = select_template(
            bot,
            group.description,
            "Unvaccinated: {0}, Vaccinated_no_booster: {1}, Booster: {2}".format(
                best_utility["unvaccinated"],
                best_utility["vaccinated_no_booster"],
                best_utility["booster"],
            ),
            vaccine=True,
        )

        alternatives = ", ".join(CHOICES_VAC)

        for member in group.members:
            if int(member.pid) not in selected_memids:
                continue

            train_profiles: List[str] = []
            val_profiles: List[str] = []
            train_persona: List[str] = []
            val_persona: List[str] = []
            train_gts: List[str] = []
            val_gts: List[str] = []

            train_ids, test_ids, val_ids = split_trips(
                member.trip_ids,
                dataset.df,
                train_n=5,
                test_n=0,
                val_n=3,
                choices=CHOICES_VAC,
            )

            for tid in train_ids:
                prompt_row = dataset.prompts.iloc[tid]
                train_profiles.append(f"{prompt_row['demographic_prompt']} {prompt_row['other_prompt']}")
                train_persona.append(prompt_row["persona"])
                train_gts.append(CHOICES_VAC[int(dataset.df.iloc[tid]['choice_idx'])])

            for tid in val_ids:
                prompt_row = dataset.prompts.iloc[tid]
                val_profiles.append(f"{prompt_row['demographic_prompt']} {prompt_row['other_prompt']}")
                val_persona.append(prompt_row["persona"])
                val_gts.append(CHOICES_VAC[int(dataset.df.iloc[tid]['choice_idx'])])

            best_template, loss_history = optimize_vaccine_prompts(
                bot,
                template_name,
                alternatives,
                train_profiles,
                train_gts,
                train_persona,
                val_profiles=val_profiles,
                val_ground_truth_choices=val_gts,
                val_persona_profiles=val_persona,
                steps=tg_steps,
            )

            for profile, gt, persona_text, trip_id in zip(val_profiles, val_gts, val_persona, val_ids):
                label, probabilities, raw_response = predict_mode_possibility(
                    bot,
                    best_template,
                    persona_text,
                    profile,
                    alternatives,
                    vaccine=True,
                )
                record_accuracy(
                    group.gid,
                    member.pid,
                    trip_id,
                    raw_response,
                    probabilities,
                    label,
                    gt,
                    label == gt,
                    persona_text,
                    best_template,
                    loss_history,
                    result_file=results_path / "accuracy_results.csv",
                )


__all__ = ["run_travel_mode", "run_vaccine"]
