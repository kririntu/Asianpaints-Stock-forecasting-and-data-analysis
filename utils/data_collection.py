import yfinance as yf
from datetime import datetime, timedelta


def download_market_data():

    ticker = "ASIANPAINT.NS"

    tomorrow = datetime.today() + timedelta(days=1)

    end_date = tomorrow.strftime("%Y-%m-%d")

    # Asian Paints
    data_asianpaints = yf.download(
        ticker,
        start="2021-01-01",
        end=end_date
    )

    # NIFTY 50
    nifty = yf.download(
        "^NSEI",
        start="2021-01-01",
        end=end_date
    )

    # NIFTY Realty
    realty = yf.download(
        "^CNXREALTY",
        start="2021-01-01",
        end=end_date
    )

    # Brent Crude
    crude = yf.download(
        "BZ=F",
        start="2021-01-01",
        end=end_date
    )

    return data_asianpaints, nifty, realty, crude