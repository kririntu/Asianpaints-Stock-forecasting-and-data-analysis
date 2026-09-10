from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def add_technical_indicators(data_asianpaints):

    # Daily return
    data_asianpaints["daily_return"] = (
        data_asianpaints["Close"].pct_change()
    )

    # Rolling Mean
    data_asianpaints["Rolling_Mean_10"] = (
        data_asianpaints["daily_return"]
        .rolling(window=10)
        .mean()
    )

    # Rolling Volatility
    data_asianpaints["volatility_10"] = (
        data_asianpaints["daily_return"]
        .rolling(window=10)
        .std()
    )

    close = data_asianpaints["Close"].squeeze()
    high = data_asianpaints["High"].squeeze()
    low = data_asianpaints["Low"].squeeze()

    # RSI
    data_asianpaints["RSI"] = (
        RSIIndicator(close, window=14).rsi()
    )

    # MACD
    macd = MACD(close)

    data_asianpaints["MACD"] = macd.macd()

    data_asianpaints["MACD_signal"] = (
        macd.macd_signal()
    )

    # EMA
    data_asianpaints["EMA_20"] = (
        EMAIndicator(close, window=20)
        .ema_indicator()
    )

    # Bollinger Bands
    bb = BollingerBands(close)

    data_asianpaints["BB_upper"] = (
        bb.bollinger_hband()
    )

    data_asianpaints["BB_lower"] = (
        bb.bollinger_lband()
    )

    # ATR
    atr = AverageTrueRange(
        high,
        low,
        close
    )

    data_asianpaints["ATR"] = (
        atr.average_true_range()
    )

    # Volume change
    data_asianpaints["volume_change"] = (
        data_asianpaints["Volume"].pct_change()
    )

    # Volume Moving Average
    data_asianpaints["volume_MA_20"] = (
        data_asianpaints["Volume"]
        .rolling(20)
        .mean()
    )

    volume = data_asianpaints["Volume"].squeeze()

    volume_ma = (
        data_asianpaints["volume_MA_20"]
        .squeeze()
    )

    data_asianpaints["volume_ratio"] = (
        volume / volume_ma
    )

    return data_asianpaints