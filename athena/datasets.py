"""Dataset abstractions and loaders for Athena."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .logging_config import PACKAGE_ROOT


logger = logging.getLogger(__name__)
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(slots=True)
class Individual:
    """Represents a person and the indices of their associated trips."""

    pid: int
    trip_ids: List[int]


@dataclass(slots=True)
class Group:
    """A collection of individuals that share a behavioural description."""

    gid: int
    description: str
    members: List[Individual]
    concept_library: List[str] = field(default_factory=list)
    utility_function: List[tuple[str, str]] = field(default_factory=list)

    def get_all_trip_ids(self) -> List[int]:
        trip_ids: set[int] = set()
        for individual in self.members:
            trip_ids.update(individual.trip_ids)
        return sorted(trip_ids)


class Dataset:
    """Abstract dataset wrapper."""

    def __init__(self, groups: Optional[List[Group]] = None) -> None:
        self.groups: List[Group] = groups or []

    def init_members(self) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def init_groups(self) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def get_group(self, gid: int) -> Optional[Group]:
        return next((group for group in self.groups if group.gid == gid), None)


class SwissMetro(Dataset):
    """Utility loader for the SwissMetro mode-choice dataset."""

    _DATA_DIR = DATA_ROOT / "swissmetro"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else self._DATA_DIR
        self._load_dataframe()
        self.init_members()
        self.init_groups()

    def _load_dataframe(self) -> None:
        mapping = {
            "trip_purpose": "PURPOSE",
            "is_first_class_traveler": "FIRST",
            "ticket_payer_type": "TICKET",
            "number_of_luggage_items": "LUGGAGE",
            "traveler_age_group": "AGE",
            "is_male": "MALE",
            "is_female": "FEMALE",
            "annual_income_level": "INCOME",
            "has_ga_travel_pass": "GA",
            "origin_canton_code": "ORIGIN",
            "destination_canton_code": "DEST",
            "is_car_available": "CAR_AV",
            "train_total_travel_time_min": "TRAIN_TT",
            "train_ticket_cost_chf": "TRAIN_CO",
            "train_service_headway_min": "TRAIN_HE",
            "sm_travel_time_min": "SM_TT",
            "sm_ticket_cost_chf": "SM_CO",
            "sm_service_headway_min": "SM_HE",
            "car_travel_time_min": "CAR_TT",
            "car_travel_cost_chf": "CAR_CO",
            "choice": "CHOICE",
        }

        reverse_mapping = {value: key for key, value in mapping.items()}

        csv_path = self.data_dir / "swissmetro_processed.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"SwissMetro data not found at {csv_path}")

        df = pd.read_csv(csv_path)
        df = df[list(mapping.values())]
        df.rename(columns=reverse_mapping, inplace=True)

        choice_map = {1: 0, 2: 1, 3: 2}  # 0=Train, 1=Swissmetro, 2=Car
        df["choice_idx"] = df["choice"].map(choice_map)
        df.drop(columns=["choice"], inplace=True)

        self.df = df

    def init_members(self) -> None:
        prompt_path = self.data_dir / "prompt_points.csv"
        multi_path = self.data_dir / "prompt_multi_col.csv"

        self.prompts = pd.read_csv(prompt_path)
        df_multiple = pd.read_csv(multi_path)
        self.trip_info = df_multiple["trip_info"].tolist()
        self.transport_options = df_multiple["transport_options"].tolist()

    def init_groups(self) -> None:
        group_path = self.data_dir / "group_pid_tid_distribution.json"
        desc_path = self.data_dir / "group_features_description.json"

        with group_path.open("r", encoding="utf-8") as fh:
            grouped_person_features = json.load(fh)

        with desc_path.open("r", encoding="utf-8") as fh:
            group_descriptions = json.load(fh)

        groups: List[Group] = []
        gid_counter = 1

        for feature_key, person_list in grouped_person_features.items():
            description = group_descriptions.get(feature_key, "No description available")
            group_individuals: List[Individual] = []

            for person_dict in person_list:
                for person_id_str, trip_id_strs in person_dict.items():
                    try:
                        person_id = int(person_id_str)
                        trip_ids = [int(tid) for tid in trip_id_strs]
                    except ValueError as exc:
                        logger.warning("Failed to parse IDs for %s: %s", person_id_str, exc)
                        continue

                    group_individuals.append(Individual(pid=person_id, trip_ids=trip_ids))

            if group_individuals:
                groups.append(Group(gid=gid_counter, description=description, members=group_individuals))
                gid_counter += 1
            else:
                logger.warning(
                    "Skipping group %s because no valid individuals were found.",
                    feature_key,
                )

        self.groups = groups


class Vaccines(Dataset):
    """Loader for the vaccination decision dataset."""

    _DATA_DIR = DATA_ROOT / "vaccine"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else self._DATA_DIR
        self.df = self._load_stats()
        self.init_members()
        self.init_groups()

    def _load_stats(self) -> pd.DataFrame:
        stats_file = self.data_dir / "df_cluster_aei.csv"
        if not stats_file.exists():
            raise FileNotFoundError(f"Vaccine statistics file not found at {stats_file}")
        return pd.read_csv(stats_file)

    def init_members(self) -> None:
        prompt_file = self.data_dir / "processed_prompts_descriptive_3_new.csv"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Vaccine prompt file not found at {prompt_file}")
        self.prompts = pd.read_csv(prompt_file)

    def init_groups(self) -> None:
        group_file = self.data_dir / "group_pid_distribution.json"
        desc_file = self.data_dir / "group_features_description.json"

        try:
            with group_file.open("r", encoding="utf-8") as fh:
                grouped_person_ids = json.load(fh)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Cannot find the group PID file: {group_file}") from exc

        if desc_file.exists():
            with desc_file.open("r", encoding="utf-8") as fh:
                group_descriptions = json.load(fh)
        else:
            group_descriptions = {}

        groups: List[Group] = []
        gid_counter = 1

        for feature_key, pid_list in grouped_person_ids.items():
            description = group_descriptions.get(feature_key, "No description available")
            individuals: List[Individual] = []

            for pid in pid_list:
                try:
                    pid_int = int(pid)
                except ValueError as exc:
                    logger.error("Invalid PID %s in group %s: %s", pid, feature_key, exc)
                    continue

                person_row = self.df[self.df["No."] == pid_int]
                if person_row.empty:
                    logger.warning("PID %s not found in stats dataframe.", pid_int)
                    continue
                person_row = person_row.iloc[0]

                cluster_rows = self.df[self.df["cluster"] == person_row["cluster"]].copy()

                def demo_distance(row: pd.Series) -> float:
                    age_dist = abs(row["age"] - person_row["age"])
                    gender_dist = 0 if row["gender"] == person_row["gender"] else 1
                    income_dist = (
                        abs(row["income_below_median"] - person_row["income_below_median"])
                        + abs(row["income_unknown"] - person_row["income_unknown"])
                    )
                    edu_dist = abs(row["have_university_degree"] - person_row["have_university_degree"])
                    return age_dist + 0.5 * gender_dist + 2 * (income_dist + edu_dist)

                cluster_rows["dist"] = cluster_rows.apply(demo_distance, axis=1)
                cluster_rows = cluster_rows.sort_values("dist")

                peer_pids = cluster_rows["No."].tolist()
                peer_pids = [p for p in peer_pids if p != pid_int][:8]
                trip_ids = peer_pids + [pid_int]
                trip_ids = [int(tid) - 1 for tid in trip_ids]

                individuals.append(Individual(pid=pid_int, trip_ids=trip_ids))

            if individuals:
                groups.append(Group(gid=gid_counter, description=description, members=individuals))
                gid_counter += 1

        self.groups = groups


def generate_selection(
    dataset: Dataset,
    *,
    total_participants: int = 500,
    output_file: Path | str | None = None,
    seed: Optional[int] = None,
) -> List[int]:
    """Select a roughly balanced subset of member IDs from a dataset."""

    if seed is not None:
        random.seed(seed)

    sizes = [len(group.members) for group in dataset.groups]
    n_groups = len(dataset.groups)
    base_quota = total_participants // n_groups if n_groups else 0
    quotas = [min(size, base_quota) for size in sizes]
    leftover = total_participants - sum(quotas)

    idx = 0
    while leftover > 0 and n_groups > 0:
        if quotas[idx] < sizes[idx]:
            quotas[idx] += 1
            leftover -= 1
        idx = (idx + 1) % n_groups

    selected_pids: List[int] = []
    for quota, group in zip(quotas, dataset.groups):
        all_pids = [member.pid for member in group.members]
        if len(all_pids) <= quota:
            chosen = all_pids
        else:
            chosen = random.sample(all_pids, quota)
        selected_pids.extend(chosen)

    if output_file is not None:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for pid in selected_pids:
                fh.write(f"{pid}\n")

    return selected_pids


def load_selected_pids(path: Path | str) -> List[int]:
    """Load previously selected participant IDs from disk."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()]


__all__ = [
    "Individual",
    "Group",
    "Dataset",
    "SwissMetro",
    "Vaccines",
    "generate_selection",
    "load_selected_pids",
    "DATA_ROOT",
]

