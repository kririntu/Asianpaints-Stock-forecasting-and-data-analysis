import joblib
import pandas as pd
import numpy as np
import os

from utils.data_collection import (
    download_market_data
)

from utils.technical_indicators import (
    add_technical_indicators
)

from utils.market_features import (
    add_market_features
)

from utils.news_collection import (
    collect_news
)

from utils.sentiment import (
    create_daily_sentiment
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "asian_paints_xgboost.pkl"
FEATURE_PATH = "feature_columns.pkl"

PROCESSED_DATA_PATH = "processed_stock_data.csv"


# ============================================================
# GET PROCESSED DATA
# ============================================================

def get_processed_data(force_refresh=False):
    """
    Return processed Asian Paints data.

    If processed_stock_data.csv exists, load it directly.
    This prevents expensive market/news/sentiment processing
    every time FastAPI or Streamlit requests the data.
    """

    # ========================================================
    # 1. LOAD EXISTING PROCESSED DATA
    # ========================================================

    if (
        os.path.exists(PROCESSED_DATA_PATH)
        and not force_refresh
    ):

        print(
            "Loading existing processed data..."
        )

        df = pd.read_csv(
            PROCESSED_DATA_PATH
        )

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = (
            df.dropna(subset=["Date"])
              .sort_values("Date")
              .reset_index(drop=True)
        )

        print(
            f"Loaded {len(df)} processed rows."
        )

        return df


    # ========================================================
    # 2. PROCESS DATA FROM SCRATCH
    # ========================================================

    print("=" * 60)
    print("PREPARING ASIAN PAINTS DATA")
    print("=" * 60)


    # --------------------------------------------------------
    # Download market data
    # --------------------------------------------------------

    print("\nDownloading market data...")

    (
        data_asianpaints,
        nifty,
        realty,
        crude
    ) = download_market_data()

    print("Market data downloaded.")


    # --------------------------------------------------------
    # Technical indicators
    # --------------------------------------------------------

    print("\nAdding technical indicators...")

    data_asianpaints = add_technical_indicators(
        data_asianpaints
    )

    print("Technical indicators added.")


    # --------------------------------------------------------
    # Market features
    # --------------------------------------------------------

    print("\nAdding market features...")

    data_asianpaints = add_market_features(
        data_asianpaints,
        nifty,
        realty,
        crude
    )

    print("Market features added.")


    # --------------------------------------------------------
    # News
    # --------------------------------------------------------

    print("\nCollecting news...")

    news_df = collect_news()

    print(
        f"News articles collected: {len(news_df)}"
    )


    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    print("\nCreating daily sentiment...")

    df_market_full = create_daily_sentiment(
        news_df,
        data_asianpaints
    )

    print("Daily sentiment created.")


    # ========================================================
    # PREPARE DATA
    # ========================================================

    data_asianpaints = data_asianpaints.copy()


    # --------------------------------------------------------
    # Flatten MultiIndex
    # --------------------------------------------------------

    if isinstance(
        data_asianpaints.columns,
        pd.MultiIndex
    ):

        data_asianpaints.columns = (
            data_asianpaints.columns
            .get_level_values(0)
        )


    # --------------------------------------------------------
    # Date column
    # --------------------------------------------------------

    if "Date" not in data_asianpaints.columns:

        data_asianpaints = (
            data_asianpaints.reset_index()
        )


    if "Date" not in data_asianpaints.columns:

        raise ValueError(
            "Date column is missing from Asian Paints data."
        )


    # --------------------------------------------------------
    # Normalize dates
    # --------------------------------------------------------

    data_asianpaints["Date"] = pd.to_datetime(
        data_asianpaints["Date"],
        errors="coerce"
    ).dt.normalize()


    df_market_full["Date"] = pd.to_datetime(
        df_market_full["Date"],
        errors="coerce"
    ).dt.normalize()


    # ========================================================
    # MERGE MARKET + SENTIMENT
    # ========================================================

    df_final = pd.merge(

        data_asianpaints,

        df_market_full,

        on="Date",

        how="left"

    )


    # ========================================================
    # FILL MISSING SENTIMENT
    # ========================================================

    news_columns = [

        "positive_news",
        "negative_news",
        "neutral_news",

        "positive_score_avg",
        "negative_score_avg",
        "neutral_score_avg",

        "news_count"

    ]


    for column in news_columns:

        if column in df_final.columns:

            df_final[column] = (
                pd.to_numeric(
                    df_final[column],
                    errors="coerce"
                )
                .fillna(0.0)
            )


    # ========================================================
    # SORT
    # ========================================================

    df_final = (
        df_final
        .sort_values("Date")
        .reset_index(drop=True)
    )


    # ========================================================
    # CLEAN NaN / INF
    # ========================================================

    df_final = df_final.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # ========================================================
    # SAVE PROCESSED DATA
    # ========================================================

    df_final.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"\nProcessed rows: {len(df_final)}"
    )

    print(
        f"Date range: "
        f"{df_final['Date'].min().date()} "
        f"to "
        f"{df_final['Date'].max().date()}"
    )

    print(
        f"\nSaved processed data to: "
        f"{PROCESSED_DATA_PATH}"
    )


    return df_final


# ============================================================
# STOCK PREDICTION
# ============================================================

def predict_stock():

    print("=" * 60)
    print("ASIAN PAINTS STOCK PREDICTION")
    print("=" * 60)


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_PATH
    )

    print("Model loaded successfully.")


    # ========================================================
    # GET PROCESSED DATA
    # ========================================================

    df_final = get_processed_data()


    # ========================================================
    # LATEST TRADING DAY
    # ========================================================

    latest = (
        df_final
        .iloc[-1:]
        .copy()
    )


    print(
        f"\nLatest trading date: "
        f"{latest['Date'].iloc[0].date()}"
    )


    # ========================================================
    # REMOVE NON-PREDICTION COLUMNS
    # ========================================================

    columns_to_remove = [

        "Close",
        "High",
        "Low",
        "Open",
        "Volume",

        "EMA_20",

        "Date"

    ]


    latest_features = latest.drop(

        columns=columns_to_remove,

        errors="ignore"

    )


    # ========================================================
    # CHECK FEATURES
    # ========================================================

    missing_features = [

        feature

        for feature in feature_columns

        if feature not in latest_features.columns

    ]


    if missing_features:

        raise ValueError(

            "Missing features required by model:\n"
            + "\n".join(missing_features)

        )


    # ========================================================
    # SELECT FEATURES
    # ========================================================

    X_latest = (
        latest_features[
            feature_columns
        ]
        .copy()
    )


    # ========================================================
    # CLEAN FEATURES
    # ========================================================

    X_latest = X_latest.replace(

        [np.inf, -np.inf],

        np.nan

    )

    X_latest = X_latest.fillna(0.0)


    # ========================================================
    # PREDICTION
    # ========================================================

    print("\nRunning prediction...")

    prediction = (
        model.predict(
            X_latest
        )[0]
    )


    probability = (
        model.predict_proba(
            X_latest
        )[0]
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    down_probability = float(
        probability[0]
    )

    up_probability = float(
        probability[1]
    )


    # ========================================================
    # DIRECTION
    # ========================================================

    if prediction == 1:

        direction = "UP"

    else:

        direction = "DOWN"


    # ========================================================
    # RESULT
    # ========================================================

    return {

        "date": str(
            latest["Date"]
            .iloc[0]
            .date()
        ),

        "prediction": direction,

        "prediction_value": int(
            prediction
        ),

        "up_probability":
            up_probability,

        "down_probability":
            down_probability

    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = predict_stock()

    print("\n")
    print("=" * 60)
    print("ASIAN PAINTS STOCK PREDICTION")
    print("=" * 60)

    print(
        f"Date: "
        f"{result['date']}"
    )

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"UP Probability: "
        f"{result['up_probability']:.4f}"
    )

    print(
        f"DOWN Probability: "
        f"{result['down_probability']:.4f}"
    )

    print("=" * 60)


