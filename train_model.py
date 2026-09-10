import joblib
import pandas as pd

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

from utils.preprocessing import (
    prepare_training_data
)

from utils.model_utils import (
    train_models
)


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():

    print("=" * 60)
    print("ASIAN PAINTS STOCK MODEL TRAINING")
    print("=" * 60)


    # --------------------------------------------------
    # 1. Download market data
    # --------------------------------------------------

    print("\nDownloading market data...")

    (
        data_asianpaints,
        nifty,
        realty,
        crude
    ) = download_market_data()

    print("Market data downloaded.")


    # --------------------------------------------------
    # 2. Technical indicators
    # --------------------------------------------------

    print("\nAdding technical indicators...")

    data_asianpaints = (
        add_technical_indicators(
            data_asianpaints
        )
    )

    print("Technical indicators added.")


    # --------------------------------------------------
    # 3. Market features
    # --------------------------------------------------

    print("\nAdding market features...")

    data_asianpaints = (
        add_market_features(
            data_asianpaints,
            nifty,
            realty,
            crude
        )
    )

    print("Market features added.")


    # --------------------------------------------------
    # 4. Collect news
    # --------------------------------------------------

    print("\nCollecting news...")

    news_df = collect_news()

    print(
        f"News articles collected: {len(news_df)}"
    )


    # --------------------------------------------------
    # 5. FinBERT sentiment
    # --------------------------------------------------

    print("\nCreating daily sentiment...")

    df_market_full = create_daily_sentiment(

        news_df,

        data_asianpaints

    )


    # --------------------------------------------------
    # 6. Prepare training data
    # --------------------------------------------------

    print("\nPreparing training data...")

    X, y = prepare_training_data(

        data_asianpaints,

        df_market_full

    )


    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )


    # --------------------------------------------------
    # 7. Show target distribution
    # --------------------------------------------------

    print("\nTarget distribution:")

    print(
        y.value_counts()
    )

    print(
        "\nTarget proportion:"
    )

    print(
        y.value_counts(
            normalize=True
        )
    )


    # --------------------------------------------------
    # 8. Train models
    # --------------------------------------------------

    print("\nStarting model training...")

    best_model = train_models(

        X,

        y

    )


    # --------------------------------------------------
    # 9. Save model
    # --------------------------------------------------

    joblib.dump(

        best_model,

        "asian_paints_xgboost.pkl"

    )


    joblib.dump(

        list(X.columns),

        "feature_columns.pkl"

    )


    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)


    print(

        "\nModel saved as:"
        "\nasian_paints_xgboost.pkl"

    )


    print(

        "\nFeatures saved as:"
        "\nfeature_columns.pkl"

    )


# --------------------------------------------------
# Run training
# --------------------------------------------------

if __name__ == "__main__":

    main()
