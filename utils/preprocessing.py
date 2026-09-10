import numpy as np
import pandas as pd


def prepare_training_data(
    data_asianpaints,
    df_market_full
):

    # --------------------------------------------------
    # 1. Make copies
    # --------------------------------------------------

    data_asianpaints = data_asianpaints.copy()
    df_market_full = df_market_full.copy()


    # --------------------------------------------------
    # 2. Flatten Yahoo Finance MultiIndex columns
    # --------------------------------------------------

    if isinstance(
        data_asianpaints.columns,
        pd.MultiIndex
    ):

        data_asianpaints.columns = (
            data_asianpaints.columns
            .get_level_values(0)
        )


    # --------------------------------------------------
    # 3. Make Date a normal column
    # --------------------------------------------------

    if "Date" not in data_asianpaints.columns:

        data_asianpaints = (
            data_asianpaints.reset_index()
        )


    # --------------------------------------------------
    # 4. Make sure Date exists
    # --------------------------------------------------

    if "Date" not in data_asianpaints.columns:

        raise ValueError(
            "Date column is missing from Asian Paints data. "
            f"Available columns: "
            f"{list(data_asianpaints.columns)}"
        )


    if "Date" not in df_market_full.columns:

        raise ValueError(
            "Date column is missing from sentiment data. "
            f"Available columns: "
            f"{list(df_market_full.columns)}"
        )


    # --------------------------------------------------
    # 5. Normalize dates
    # --------------------------------------------------

    data_asianpaints["Date"] = pd.to_datetime(
        data_asianpaints["Date"],
        errors="coerce"
    ).dt.normalize()

    df_market_full["Date"] = pd.to_datetime(
        df_market_full["Date"],
        errors="coerce"
    ).dt.normalize()


    # --------------------------------------------------
    # 6. Remove invalid dates
    # --------------------------------------------------

    data_asianpaints = (
        data_asianpaints
        .dropna(subset=["Date"])
    )

    df_market_full = (
        df_market_full
        .dropna(subset=["Date"])
    )


    # --------------------------------------------------
    # 7. Keep only dates available in sentiment data
    # --------------------------------------------------

    data_asianpaints_new1 = (
        data_asianpaints[
            data_asianpaints["Date"].isin(
                df_market_full["Date"]
            )
        ]
        .copy()
    )


    # --------------------------------------------------
    # 8. Merge stock + sentiment
    # --------------------------------------------------

    df_final_asian = pd.merge(

        data_asianpaints_new1,

        df_market_full,

        on="Date",

        how="left"

    )


    # --------------------------------------------------
    # 9. Sort chronologically
    # --------------------------------------------------

    df_final_asian = (
        df_final_asian
        .sort_values("Date")
        .reset_index(drop=True)
    )


    # --------------------------------------------------
    # 10. Future return
    # --------------------------------------------------

    df_final_asian["future_return"] = (
        df_final_asian["daily_return"]
        .shift(-1)
    )


    # --------------------------------------------------
    # 11. Target variable
    # --------------------------------------------------

    df_final_asian["target"] = np.where(

        df_final_asian["future_return"] > 0,

        1,

        0

    )


    # --------------------------------------------------
    # 12. Remove NaN rows
    # --------------------------------------------------

    df_final_asian.dropna(
        inplace=True
    )


    # --------------------------------------------------
    # 13. Set Date as index
    # --------------------------------------------------

    df_final_asian = (
        df_final_asian
        .set_index("Date")
    )


    # --------------------------------------------------
    # 14. Remove unwanted price columns
    # --------------------------------------------------

    df_final_asian.drop(

        columns=[
            "Close",
            "High",
            "Low",
            "Open",
            "Volume"
        ],

        inplace=True,

        errors="ignore"

    )


    # --------------------------------------------------
    # 15. Remove EMA
    # --------------------------------------------------

    df_final_asian.drop(

        columns=["EMA_20"],

        inplace=True,

        errors="ignore"

    )


    # --------------------------------------------------
    # 16. Separate target
    # --------------------------------------------------

    y = df_final_asian["target"]


    # --------------------------------------------------
    # 17. Remove target from features
    # --------------------------------------------------

    df_final_asian.drop(

        columns=["target"],

        inplace=True

    )


    # --------------------------------------------------
    # 18. Feature matrix
    # --------------------------------------------------

    X = df_final_asian.copy()


    # --------------------------------------------------
    # 19. Replace infinite values
    # --------------------------------------------------

    X = X.replace(

        [np.inf, -np.inf],

        np.nan

    )


    # --------------------------------------------------
    # 20. Drop NaN
    # --------------------------------------------------

    X = X.dropna()


    # --------------------------------------------------
    # 21. Align target with features
    # --------------------------------------------------

    y = y.loc[X.index]


    # --------------------------------------------------
    # 22. Remove future return
    # --------------------------------------------------

    X.drop(

        columns=["future_return"],

        inplace=True,

        errors="ignore"

    )


    # --------------------------------------------------
    # 23. Print final dataset information
    # --------------------------------------------------

    print(
        f"Prepared dataset shape: {X.shape}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    print(
        f"Number of samples: {len(y)}"
    )


    return X, y
