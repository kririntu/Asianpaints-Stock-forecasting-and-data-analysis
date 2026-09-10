from fastapi import FastAPI
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np

from predict import get_processed_data


app = FastAPI(
    title="Asian Paints Data Analysis API",
    description="Historical Asian Paints stock market data analysis API.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Asian Paints Data Analysis API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/dashboard")
def dashboard_data():

    try:

        # This now loads processed_stock_data.csv
        # instead of downloading and processing everything again.
        df = get_processed_data()

        if df is None or df.empty:
            raise Exception("No stock data available.")

        # -----------------------------
        # Clean data
        # -----------------------------

        df = df.copy()

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(subset=["Date"])

        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        # -----------------------------
        # Basic price statistics
        # -----------------------------

        latest_price = float(df["Close"].iloc[-1])

        highest_price = float(df["Close"].max())

        lowest_price = float(df["Close"].min())

        average_price = float(df["Close"].mean())

        total_days = int(len(df))

        # -----------------------------
        # Daily returns
        # -----------------------------

        df["Return"] = (
            df["Close"]
            .pct_change()
            * 100
        )

        valid_returns = (
            df["Return"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(valid_returns) > 0:

            average_return = float(
                valid_returns.mean()
            )

            volatility = float(
                valid_returns.std()
            )

            up_days = int(
                (valid_returns > 0).sum()
            )

            down_days = int(
                (valid_returns < 0).sum()
            )

        else:

            average_return = 0.0
            volatility = 0.0
            up_days = 0
            down_days = 0

        # -----------------------------
        # Recent 90 trading days
        # -----------------------------

        recent_df = df.tail(90).copy()

        recent_df["Volatility"] = (
            recent_df["Return"]
            .rolling(20)
            .std()
        )

        columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Return",
            "Volatility"
        ]

        available_columns = [
            column
            for column in columns
            if column in recent_df.columns
        ]

        recent_df = recent_df[
            available_columns
        ]

        # -----------------------------
        # Clean NaN / Inf values
        # -----------------------------

        recent_df = recent_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        recent_df["Date"] = (
            recent_df["Date"]
            .dt.strftime("%Y-%m-%d")
        )

        # Convert NaN → None
        recent_df = recent_df.astype(object)

        recent_df = recent_df.where(
            pd.notnull(recent_df),
            None
        )

        # -----------------------------
        # Final response
        # -----------------------------

        return {
            "success": True,

            "summary": {
                "latest_price": latest_price,
                "highest_price": highest_price,
                "lowest_price": lowest_price,
                "average_price": average_price,
                "total_days": total_days,
                "average_return": average_return,
                "volatility": volatility,
                "up_days": up_days,
                "down_days": down_days
            },

            "data": recent_df.to_dict(
                orient="records"
            )
        }

    except Exception as e:

        print(
            f"DASHBOARD API ERROR: {e}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

