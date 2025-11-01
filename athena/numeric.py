"""Numeric-layer optimisation and evaluation helpers."""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .prompts import VARIABLE_MAP


logger = logging.getLogger(__name__)


def create_utility_function(formula_str: str, param_prefix: str, variable_map: Dict[str, str]):
    """Compile a string-based utility function into a callable."""

    numpy_to_torch = {
        "np.sqrt": "torch.sqrt",
        "np.log": "torch.log",
        "np.exp": "torch.exp",
        "np.abs": "torch.abs",
        "np.sin": "torch.sin",
        "np.cos": "torch.cos",
        "np.tan": "torch.tan",
        "np.max": "torch.max",
        "np.min": "torch.min",
        "sqrt(": "torch.sqrt(",
        "log(": "torch.log(",
        "exp(": "torch.exp(",
        "abs(": "torch.abs(",
        "sin(": "torch.sin(",
        "cos(": "torch.cos(",
        "tan(": "torch.tan(",
        "max(": "torch.max(",
        "min(": "torch.min(",
        "mod(": "torch.remainder(",
        "np.mod(": "torch.remainder(",
    }

    processed_str = formula_str
    for orig_func, torch_func in numpy_to_torch.items():
        processed_str = processed_str.replace(orig_func, torch_func)

    for raw_var, xdict_access in variable_map.items():
        processed_str = processed_str.replace(raw_var, xdict_access)

    if "K" not in processed_str and "C" not in processed_str:
        raise ValueError("Utility function must contain 'K' and 'C' parameters.")

    processed_str = re.sub(r"\bK\d*\b", lambda _: f"self.{param_prefix}_K", processed_str)
    processed_str = re.sub(r"\bC\d*\b", lambda _: f"self.{param_prefix}_C", processed_str)

    fn_code = f"""
def utility_fn(self, X_dict):
    return {processed_str}
"""
    logger.debug("Utility function code for %s: %s", param_prefix, fn_code)

    temp_dict: Dict[str, object] = {}
    exec(fn_code, {"torch": torch}, temp_dict)
    return temp_dict["utility_fn"]


class MNLModel(nn.Module):
    def __init__(self, variable_map: Dict[str, str], train_formula_str: str, metro_formula_str: str, car_formula_str: str):
        super().__init__()

        self.train_K = nn.Parameter(torch.tensor(0.0))
        self.train_C = nn.Parameter(torch.tensor(0.0))
        self.metro_K = nn.Parameter(torch.tensor(0.0))
        self.metro_C = nn.Parameter(torch.tensor(0.0))
        self.car_K = nn.Parameter(torch.tensor(0.0))
        self.car_C = nn.Parameter(torch.tensor(0.0))

        self.train_utility_fn = create_utility_function(train_formula_str, "train", variable_map)
        self.metro_utility_fn = create_utility_function(metro_formula_str, "metro", variable_map)
        self.car_utility_fn = create_utility_function(car_formula_str, "car", variable_map)

    def forward(self, X_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        U_train = self.train_utility_fn(self, X_dict)
        U_metro = self.metro_utility_fn(self, X_dict)
        U_car = self.car_utility_fn(self, X_dict)
        return torch.stack([U_train, U_metro, U_car], dim=1)

    def predict_proba(self, X_dict: Dict[str, torch.Tensor], *, return_tensor: bool = False):
        self.eval()
        with torch.no_grad():
            utilities = torch.nan_to_num(self(X_dict), nan=0.0, posinf=1e6, neginf=-1e6)
            probs = torch.softmax(utilities, dim=1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                n_classes = probs.size(1)
                mask = torch.isnan(probs).any(dim=1) | torch.isinf(probs).any(dim=1)
                probs[mask] = 1.0 / n_classes

        return probs if return_tensor else probs.cpu().numpy()


class BinaryLogitModel(nn.Module):
    def __init__(self, variable_map: Dict[str, str], vacc_formula_str: str):
        super().__init__()
        self.vacc_K = nn.Parameter(torch.tensor(0.0))
        self.vacc_C = nn.Parameter(torch.tensor(0.0))
        self.vacc_utility_fn = create_utility_function(vacc_formula_str, "vacc", variable_map)

    def forward(self, X_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        U_not = torch.zeros_like(next(iter(X_dict.values())))
        U_vacc = self.vacc_utility_fn(self, X_dict)
        return torch.stack([U_not, U_vacc], dim=1)


def mnl_neg_log_likelihood(U: torch.Tensor, choice_idx: torch.Tensor) -> torch.Tensor:
    log_prob = U - torch.logsumexp(U, dim=1, keepdim=True)
    chosen_lp = log_prob[torch.arange(U.shape[0]), choice_idx]
    return -chosen_lp.mean()


def split_dataset(df, train_ratio: float = 0.8, target_col: str = "choice_idx"):
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]

    train_X = {col: torch.tensor(train_df[col].values, dtype=torch.float32) for col in df.columns if col != target_col}
    train_choice_idx = torch.tensor(train_df[target_col].values, dtype=torch.long)

    test_X = {col: torch.tensor(test_df[col].values, dtype=torch.float32) for col in df.columns if col != target_col}
    test_choice_idx = torch.tensor(test_df[target_col].values, dtype=torch.long)
    return train_X, train_choice_idx, test_X, test_choice_idx


def train_model(model: nn.Module, optimizer: optim.Optimizer, train_X, train_choice_idx, num_epochs: int = 5):
    def closure():
        optimizer.zero_grad()
        U = model(train_X)
        loss = mnl_neg_log_likelihood(U, train_choice_idx)
        loss.backward()
        return loss

    for epoch in range(num_epochs):
        optimizer.step(closure)
        logger.info("Epoch %d, loss=%.4f", epoch, closure().item())

    final_loss = closure()
    logger.info("Final loss = %.4f", final_loss.item())
    return model


def evaluate_model(model: MNLModel, test_X, test_choice_idx):
    model.eval()
    with torch.no_grad():
        U = model(test_X)
        if torch.isnan(U).any() or torch.isinf(U).any():
            logger.warning("NaN or Inf detected in utilities U; sanitizing.")
            U = torch.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)
        probs = torch.softmax(U, dim=1)

        if torch.isnan(probs).any():
            logger.warning("NaN detected in softmax probabilities; replacing with uniform distribution.")
            probs = torch.full_like(probs, 1.0 / probs.size(1))

        predicted_choices = torch.argmax(probs, dim=1)
        proba = probs.cpu().numpy()
        n_classes = proba.shape[1]
        labels = np.arange(n_classes)
        y_true = test_choice_idx.cpu().numpy()
        y_pred = predicted_choices.cpu().numpy()
        ce = log_loss(y_true, proba, labels=labels)
        logger.info("Cross Entropy: %.4f", ce)
        proba = model.predict_proba(test_X)

        if np.isnan(proba).any() or np.isinf(proba).any():
            logger.warning("NaN or Inf detected in predict_proba output; replacing with uniform distribution.")
            row_sums = proba.sum(axis=1, keepdims=True)
            zero_rows = row_sums.squeeze() == 0
            if zero_rows.any():
                proba[zero_rows, :] = 1.0 / n_classes
                row_sums[zero_rows] = 1.0
            proba = proba / row_sums

        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            auc = float("nan")
            logger.warning("Only one class present in y_true; AUC is undefined (set to NaN).")
        elif len(unique_labels) == 2:
            pos_label = unique_labels[1]
            auc = roc_auc_score(y_true, proba[:, pos_label], labels=unique_labels)
        else:
            auc = roc_auc_score(y_true, proba, labels=np.arange(n_classes), multi_class="ovr", average="macro")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        logger.info(
            "Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, Cross-Entropy: %.4f, AUC: %.4f",
            acc,
            prec,
            rec,
            f1,
            ce,
            auc,
        )

        return acc, prec, rec, f1, ce, auc


def evaluate_utility_function(group_df, group_id: int, utility_function: Iterable[str], variable_map=VARIABLE_MAP):
    if len(utility_function) != 3:
        raise ValueError("Expected a tuple of three expressions (train, car, metro).")

    train, car, metro = utility_function

    try:
        model = MNLModel(
            variable_map=variable_map,
            train_formula_str=train,
            metro_formula_str=metro,
            car_formula_str=car,
        )
    except Exception as exc:
        logger.warning("Group %s utility_function error: %s", group_id, exc)
        return None

    train_X, train_choice_idx, test_X, test_choice_idx = split_dataset(group_df)
    optimizer = optim.LBFGS(model.parameters(), lr=0.0001, max_iter=200)
    model = train_model(model, optimizer, train_X, train_choice_idx)
    metrics = evaluate_model(model, test_X, test_choice_idx)
    logger.info("Group %s utility_function accuracy: %.4f", group_id, metrics[0])
    return (train, car, metro, *metrics)


__all__ = [
    "create_utility_function",
    "MNLModel",
    "BinaryLogitModel",
    "mnl_neg_log_likelihood",
    "split_dataset",
    "train_model",
    "evaluate_model",
    "evaluate_utility_function",
]

