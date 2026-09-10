def add_market_features(data_asianpaints, nifty, realty, crude):

    # Daily Returns
    nifty["daily_return"] = nifty["Close"].pct_change()

    realty["daily_return"] = realty["Close"].pct_change()

    crude["daily_return"] = crude["Close"].pct_change()

    # Merge with Asian Paints

    data_asianpaints["nifty_return"] = nifty["daily_return"]

    data_asianpaints["realty_return"] = realty["daily_return"]

    data_asianpaints["crude_return"] = crude["daily_return"]

    return data_asianpaints
