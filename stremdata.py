import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Asian Paints | Data Analysis",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# FASTAPI
# ============================================================

API_URL = "http://127.0.0.1:8000/dashboard"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 20px;
    }

    /* Reduce spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* KPI styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #e5e5e5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 Asian Paints Stock Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Historical market performance and risk analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FETCH DATA FROM FASTAPI
# ============================================================

@st.cache_data(ttl=300)
def get_dashboard_data():

    try:

        response = requests.get(
            API_URL,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("success", False):

            return None, result.get(
                "error",
                "Unknown API error"
            )

        return result, None

    except requests.exceptions.ConnectionError:

        return None, (
            "Could not connect to FastAPI. "
            "Please start FastAPI using "
            "`uvicorn dataapp:app --reload`."
        )

    except requests.exceptions.Timeout:

        return None, (
            "FastAPI request timed out. "
            "Check whether the API is running correctly."
        )

    except Exception as e:

        return None, str(e)


# ============================================================
# LOAD DATA
# ============================================================

result, error = get_dashboard_data()


if error:

    st.error(error)

    st.code(
        "uvicorn dataapp:app --reload",
        language="bash"
    )

    st.stop()


# ============================================================
# EXTRACT DATA
# ============================================================

summary = result["summary"]

df = pd.DataFrame(
    result["data"]
)


# ============================================================
# PREPARE DATAFRAME
# ============================================================

if df.empty:

    st.warning("No historical data available.")

    st.stop()


# Convert Date

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# Convert numeric columns

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "Volatility"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# Clean data

df = (
    df
    .dropna(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# DATA RANGE
# ============================================================

start_date = df["Date"].min().strftime(
    "%d %b %Y"
)

end_date = df["Date"].max().strftime(
    "%d %b %Y"
)


st.caption(
    f"Analysis period: **{start_date} → {end_date}** "
    f"| Showing latest 90 trading days"
)


# ============================================================
# KEY METRICS
# ============================================================

st.subheader("Key Metrics")


# ------------------------------------------------------------
# FIRST ROW
# ------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.metric(
        "Latest Price",
        f"₹{summary['latest_price']:.2f}"
    )


with c2:

    st.metric(
        "Highest Price",
        f"₹{summary['highest_price']:.2f}"
    )


with c3:

    st.metric(
        "Lowest Price",
        f"₹{summary['lowest_price']:.2f}"
    )


with c4:

    st.metric(
        "Average Price",
        f"₹{summary['average_price']:.2f}"
    )


with c5:

    st.metric(
        "Trading Days",
        f"{summary['total_days']:,}"
    )


# ------------------------------------------------------------
# SECOND ROW
# ------------------------------------------------------------

c6, c7, c8, c9 = st.columns(4)


with c6:

    st.metric(
        "Avg Daily Return",
        f"{summary['average_return']:.3f}%"
    )


with c7:

    st.metric(
        "Daily Volatility",
        f"{summary['volatility']:.3f}%"
    )


with c8:

    st.metric(
        "Up Days",
        summary["up_days"]
    )


with c9:

    st.metric(
        "Down Days",
        summary["down_days"]
    )


# ============================================================
# PRICE TREND
# ============================================================

st.subheader("📈 Closing Price Trend")


fig_price = go.Figure()


fig_price.add_trace(

    go.Scatter(

        x=df["Date"],

        y=df["Close"],

        mode="lines",

        name="Close Price",

        line=dict(width=2)

    )

)


fig_price.update_layout(

    height=360,

    xaxis_title="Date",

    yaxis_title="Price (₹)",

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),

    showlegend=False

)


st.plotly_chart(
    fig_price,
    use_container_width=True
)


# ============================================================
# CREATE TWO-COLUMN LAYOUT
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# OHLC / CANDLESTICK
# ============================================================

with col1:

    st.subheader("📊 OHLC & Trading Activity")


    fig_ohlc = go.Figure()


    fig_ohlc.add_trace(

        go.Candlestick(

            x=df["Date"],

            open=df["Open"],

            high=df["High"],

            low=df["Low"],

            close=df["Close"],

            name="Asian Paints"

        )

    )


    fig_ohlc.update_layout(

        height=350,

        xaxis_title="Date",

        yaxis_title="Price (₹)",

        xaxis_rangeslider_visible=False,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )

    )


    st.plotly_chart(
        fig_ohlc,
        use_container_width=True
    )


# ============================================================
# RIGHT COLUMN
# DAILY RETURNS
# ============================================================

with col2:

    st.subheader("📉 Daily Returns")


    fig_returns = go.Figure()


    fig_returns.add_trace(

        go.Bar(

            x=df["Date"],

            y=df["Return"],

            name="Daily Return"

        )

    )


    fig_returns.add_hline(

        y=0,

        line_width=1

    )


    fig_returns.update_layout(

        height=350,

        xaxis_title="Date",

        yaxis_title="Return (%)",

        hovermode="x unified",

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        showlegend=False

    )


    st.plotly_chart(
        fig_returns,
        use_container_width=True
    )


# ============================================================
# SECOND TWO-COLUMN ROW
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ROLLING VOLATILITY
# ============================================================

with col1:

    st.subheader("📊 20-Day Rolling Volatility")


    fig_volatility = go.Figure()


    fig_volatility.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["Volatility"],

            mode="lines",

            name="20-Day Volatility",

            line=dict(width=2)

        )

    )


    fig_volatility.update_layout(

        height=350,

        xaxis_title="Date",

        yaxis_title="Volatility (%)",

        hovermode="x unified",

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        showlegend=False

    )


    st.plotly_chart(

        fig_volatility,

        use_container_width=True

    )


# ============================================================
# RIGHT COLUMN
# MARKET DIRECTION - DONUT CHART
# ============================================================

with col2:

    st.subheader("🥯 Market Direction")


    fig_direction = go.Figure(

        go.Pie(

            labels=[
                "Up Days",
                "Down Days"
            ],

            values=[
                summary["up_days"],
                summary["down_days"]
            ],

            hole=0.55,

            textinfo="label+percent",

            hovertemplate=(
                "%{label}: %{value} days"
                "<extra></extra>"
            )

        )

    )


    fig_direction.update_layout(

        height=350,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        showlegend=True,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),

        annotations=[

            dict(

                text="Market<br>Direction",

                x=0.5,

                y=0.5,

                font=dict(
                    size=16
                ),

                showarrow=False

            )

        ]

    )


    st.plotly_chart(

        fig_direction,

        use_container_width=True

    )


# ============================================================
# HISTORICAL DATA
# ============================================================

st.subheader("🔍 Historical Data")


with st.expander("View Historical Data"):

    display_df = df.copy()


    display_df["Date"] = (

        display_df["Date"]

        .dt.strftime("%Y-%m-%d")

    )


    st.dataframe(

        display_df,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(

    "Asian Paints Data Analysis | "
    "Python • Pandas • Plotly • FastAPI • Streamlit"

)
