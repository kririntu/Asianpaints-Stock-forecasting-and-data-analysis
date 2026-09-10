from fastapi import FastAPI
from fastapi.responses import JSONResponse

from predict import (
    predict_stock,
    get_processed_data
)


app = FastAPI(
    title="Asian Paints Stock Prediction API",
    description=(
        "Stock direction prediction using "
        "XGBoost, technical indicators, "
        "market features and news sentiment."
    ),
    version="1.0.0"
)


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Asian Paints Stock Prediction API",
        "status": "running"
    }


# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==================================================
# PREDICTION
# ==================================================

@app.get("/predict")
def prediction():

    try:

        result = predict_stock()

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# ==================================================
# DATA
# ==================================================

@app.get("/data")
def stock_data():

    try:

        df = get_processed_data()

        # ------------------------------------------
        # Convert Date to string
        # ------------------------------------------

        if "Date" in df.columns:

            df["Date"] = (
                pd.to_datetime(df["Date"])
                .dt.strftime("%Y-%m-%d")
            )


        # ------------------------------------------
        # Replace infinity
        # ------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )


        # ------------------------------------------
        # Replace NaN with None
        # ------------------------------------------

        df = df.astype(object)

        df = df.where(
            pd.notnull(df),
            None
        )


        # ------------------------------------------
        # Convert dataframe to records
        # ------------------------------------------

        data = df.to_dict(
            orient="records"
        )


        return {
            "success": True,
            "data": data
        }


    except Exception as e:

        print(
            f"DATA API ERROR: {e}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
