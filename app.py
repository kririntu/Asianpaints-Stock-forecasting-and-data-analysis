from fastapi import FastAPI
from fastapi.responses import JSONResponse

from predict import predict_stock


app = FastAPI(
    title="Asian Paints Stock Prediction API",
    description=(
        "Stock direction prediction using "
        "XGBoost, technical indicators, "
        "market features and news sentiment."
    ),
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "Asian Paints Stock Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


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