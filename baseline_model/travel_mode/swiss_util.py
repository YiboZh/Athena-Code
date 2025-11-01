from choice_learn.datasets import load_swissmetro
from choice_learn.data import ChoiceDataset

def load_swissmetro_demographic(
    *,
    age=None,
    income=None,
    male=None,
    as_frame=False,
    preprocessing="biogeme_nested",
    **other_kwargs,
):
    swiss_df = load_swissmetro(as_frame=True,
                               preprocessing=preprocessing,
                               **other_kwargs)

    if age is not None:
        if isinstance(age, slice):
            swiss_df = swiss_df[swiss_df["AGE"].between(age.start, age.stop)]
        elif isinstance(age, (list, tuple, set)):
            swiss_df = swiss_df[swiss_df["AGE"].isin(age)]
        else:
            swiss_df = swiss_df[swiss_df["AGE"] == age]

    if income is not None:
        if isinstance(income, slice):
            swiss_df = swiss_df[swiss_df["INCOME"].between(income.start, income.stop)]
        elif isinstance(income, (list, tuple, set)):
            swiss_df = swiss_df[swiss_df["INCOME"].isin(income)]
        else:
            swiss_df = swiss_df[swiss_df["INCOME"] == income]

    if male is not None:
        male_val = int(bool(male))
        swiss_df = swiss_df[swiss_df["MALE"] == male_val]

    if as_frame:
        return swiss_df.reset_index(drop=True)

    items = ["TRAIN", "SM", "CAR"]
    shared_cols = [
        "GROUP", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE",
        "AGE", "MALE", "INCOME", "GA", "ORIGIN", "DEST",
    ]
    item_suffix = ["CO", "TT", "HE", "SEATS"]

    return ChoiceDataset.from_single_wide_df(
        df=swiss_df,
        items_id=items,
        shared_features_columns=shared_cols,
        items_features_suffixes=item_suffix,
        available_items_suffix="AV",
        choices_column="CHOICE",
        choice_format="items_index",
    )