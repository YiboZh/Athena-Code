#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 4 baseline choice‑models on SwissMetro for every (AGE, INCOME, MALE) group
and write a CSV:  model, AGE, INCOME, MALE, accuracy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from choice_learn.datasets import load_swissmetro
from choice_learn.data import ChoiceDataset
from choice_learn.models import (
    SimpleMNL,
    ConditionalLogit,
    NestedLogit,
)
from choice_learn.models.latent_class_mnl import LatentClassSimpleMNL

swiss_df = load_swissmetro(as_frame=True, preprocessing="biogeme_nested")

ITEMS = ["TRAIN", "SM", "CAR"]
SHARED_COLS = [
    "GROUP", "SURVEY", "SP", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE",
    "AGE", "MALE", "INCOME", "GA", "ORIGIN", "DEST"
]
ITEM_SUFFIXES = ["TT", "CO", "HE", "SEATS"]
AVAIL_SUFFIX = "AV"
CHOICE_COL = "CHOICE"

def to_dataset(df: pd.DataFrame) -> ChoiceDataset:
    return ChoiceDataset.from_single_wide_df(
        df=df,
        items_id=ITEMS,
        shared_features_columns=SHARED_COLS,
        items_features_suffixes=ITEM_SUFFIXES,
        available_items_suffix=AVAIL_SUFFIX,
        choices_column=CHOICE_COL,
        choice_format="item_index",
    )

results = []

for (age, income, male), gdf in swiss_df.groupby(["AGE", "INCOME", "MALE"]):
    if len(gdf) < 2:
        continue

    train_df, test_df = train_test_split(gdf, test_size=0.2, random_state=0)
    if train_df.empty or test_df.empty:
        continue

    train_data, test_data = map(to_dataset, (train_df, test_df))
    y_true = test_df[CHOICE_COL].to_numpy()

    # ---------- MNL ----------
    mnl = SimpleMNL(intercept="item", optimizer="adam")
    mnl.fit(train_data)
    y_pred = mnl.predict_probas(test_data).numpy().argmax(axis=1)
    results.append(["MNL", age, income, male, (y_pred == y_true).mean()])

    # ---------- Conditional Logit ----------
    clogit = ConditionalLogit(optimizer="adam")
    clogit.add_coefficients("intercept", items_indexes=[1, 2])
    clogit.fit(train_data)
    y_pred = clogit.predict_probas(test_data).numpy().argmax(axis=1)
    results.append(["Clogit", age, income, male, (y_pred == y_true).mean()])

    # ---------- Nested Logit ----------
    nests = [[0, 2], [1]]          # (TRAIN+CAR) vs SM
    nlogit = NestedLogit(items_nests=nests, optimizer="adam")
    nlogit.add_coefficients("intercept", items_indexes=[1, 2])
    nlogit.fit(train_data)
    y_pred = nlogit.predict_probas(test_data).numpy().argmax(axis=1)
    results.append(["Nested Logit", age, income, male, (y_pred == y_true).mean()])

    # ---------- Latent‑Class MNL ----------
    lcmnl = LatentClassSimpleMNL(
        n_latent_classes=2,
        fit_method="mle",
        optimizer="adam",
        epochs=1000,
    )
    lcmnl.fit(train_data)
    y_pred = lcmnl.predict_probas(test_data).numpy().argmax(axis=1)
    results.append(["Latent Class MNL", age, income, male, (y_pred == y_true).mean()])

out_df = pd.DataFrame(
    results, columns=["model", "AGE", "INCOME", "MALE", "accuracy"]
)
out_df.to_csv("model_group_accuracies.csv", index=False)
print("Saved -> model_group_accuracies.csv with", len(out_df), "rows")