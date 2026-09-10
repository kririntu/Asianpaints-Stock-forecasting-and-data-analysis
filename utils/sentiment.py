import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# --------------------------------------------------
# Load FinBERT
# --------------------------------------------------

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()


# FinBERT label order
LABELS = [
    "positive",
    "negative",
    "neutral"
]


# --------------------------------------------------
# Analyze sentiment
# --------------------------------------------------

def analyze_sentiment(text):

    if not text or not str(text).strip():

        return {
            "sentiment": "neutral",
            "score": 0.0
        }

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    sentiment = LABELS[predicted_class]

    score = float(
        probabilities[predicted_class]
    )

    return {
        "sentiment": sentiment,
        "score": score
    }


# --------------------------------------------------
# Convert sentiment to numerical value
# --------------------------------------------------

def sentiment_to_numeric(sentiment):

    if sentiment == "positive":

        return 1

    elif sentiment == "negative":

        return -1

    else:

        return 0


# --------------------------------------------------
# Create daily sentiment
# --------------------------------------------------

def create_daily_sentiment(
    df,
    data_asianpaints
):

    sentiments = []

    print(
        f"Running FinBERT on {len(df)} news articles..."
    )

    # --------------------------------------------------
    # 1. Run FinBERT on every news title
    # --------------------------------------------------

    for title in df["title"]:

        try:

            result = analyze_sentiment(title)

            sentiments.append(result)

        except Exception as e:

            print(f"FinBERT error: {e}")

            sentiments.append({
                "sentiment": "neutral",
                "score": 0.0
            })


    # --------------------------------------------------
    # 2. Add sentiment results
    # --------------------------------------------------

    df_sentiments = pd.DataFrame(
        sentiments
    )

    df = df.copy()

    df["sentiment"] = (
        df_sentiments["sentiment"]
    )

    df["score"] = (
        df_sentiments["score"]
    )


    # --------------------------------------------------
    # 3. Convert sentiment to numerical
    # --------------------------------------------------

    df_market = df.copy()

    df_market["sentiment"] = (
        df_market["sentiment"]
        .apply(sentiment_to_numeric)
    )


    # --------------------------------------------------
    # 4. Convert news date
    # --------------------------------------------------

    df_market["published"] = pd.to_datetime(
        df_market["published"],
        errors="coerce"
    ).dt.normalize()

    df_market.rename(
        columns={
            "published": "Date"
        },
        inplace=True
    )


    # --------------------------------------------------
    # 5. Remove unnecessary columns
    # --------------------------------------------------

    df_market.drop(
        columns=[
            "title",
            "link"
        ],
        inplace=True,
        errors="ignore"
    )


    # --------------------------------------------------
    # 6. Prepare Asian Paints dataframe
    # --------------------------------------------------

    data_asianpaints = (
        data_asianpaints.copy()
    )


    # IMPORTANT:
    # Yahoo Finance may return MultiIndex columns

    if isinstance(
        data_asianpaints.columns,
        pd.MultiIndex
    ):

        data_asianpaints.columns = (
            data_asianpaints.columns
            .get_level_values(0)
        )


    # --------------------------------------------------
    # 7. Convert Date index to Date column
    # --------------------------------------------------

    if "Date" not in data_asianpaints.columns:

        data_asianpaints = (
            data_asianpaints.reset_index()
        )


    # --------------------------------------------------
    # 8. Make Date consistent
    # --------------------------------------------------

    data_asianpaints["Date"] = pd.to_datetime(
        data_asianpaints["Date"],
        errors="coerce"
    ).dt.normalize()


    # --------------------------------------------------
    # 9. Determine news date range
    # --------------------------------------------------

    start_date = df_market["Date"].min()

    end_date = df_market["Date"].max()


    # --------------------------------------------------
    # 10. Get trading days
    # --------------------------------------------------

    trade_days = data_asianpaints[
        (data_asianpaints["Date"] >= start_date)
        &
        (data_asianpaints["Date"] <= end_date)
    ]["Date"]


    trade_days = (
        trade_days
        .drop_duplicates()
        .sort_values()
    )


    # --------------------------------------------------
    # 11. Map news to next trading day
    # --------------------------------------------------

    def next_trading_day(date):

        if pd.isna(date):

            return pd.NaT

        date = pd.Timestamp(
            date
        ).normalize()

        future_days = trade_days[
            trade_days >= date
        ]

        if len(future_days) == 0:

            return pd.NaT

        return future_days.iloc[0]


    df_market["Date"] = (
        df_market["Date"]
        .apply(next_trading_day)
    )


    # --------------------------------------------------
    # 12. Remove unmapped dates
    # --------------------------------------------------

    df_market.dropna(
        subset=["Date"],
        inplace=True
    )


    # --------------------------------------------------
    # 13. Aggregate news by day
    # --------------------------------------------------

    daily_news = (

        df_market
        .groupby("Date")
        .apply(
            lambda x: pd.Series({

                "positive_news":
                    (x["sentiment"] == 1).sum(),

                "negative_news":
                    (x["sentiment"] == -1).sum(),

                "neutral_news":
                    (x["sentiment"] == 0).sum(),

                "positive_score_avg":
                    x.loc[
                        x["sentiment"] == 1,
                        "score"
                    ].mean(),

                "negative_score_avg":
                    x.loc[
                        x["sentiment"] == -1,
                        "score"
                    ].mean(),

                "neutral_score_avg":
                    x.loc[
                        x["sentiment"] == 0,
                        "score"
                    ].mean(),

                "news_count":
                    len(x)

            })
        )
        .reset_index()
        .fillna(0)
    )


    # --------------------------------------------------
    # 14. Create all trading days dataframe
    # --------------------------------------------------

    trade_days_df = pd.DataFrame({
        "Date": trade_days
    })

    trade_days_df.reset_index(
        drop=True,
        inplace=True
    )


    # --------------------------------------------------
    # 15. Merge sentiment with trading days
    # --------------------------------------------------

    df_market_full = pd.merge(

        trade_days_df,

        daily_news,

        on="Date",

        how="left"

    )


    # --------------------------------------------------
    # 16. Fill days without news with zero
    # --------------------------------------------------

    sentiment_columns = [

        "positive_news",
        "negative_news",
        "neutral_news",

        "positive_score_avg",
        "negative_score_avg",
        "neutral_score_avg",

        "news_count"

    ]

    for column in sentiment_columns:

        if column in df_market_full.columns:

            df_market_full[column] = (
                df_market_full[column]
                .fillna(0.0)
            )


    print(
        "Daily sentiment created successfully."
    )

    return df_market_full
