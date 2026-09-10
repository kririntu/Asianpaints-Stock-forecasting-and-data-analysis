# 📊 Asian Paints Stock Analysis & Prediction

An end-to-end **Data Science and Machine Learning project for Asian Paints Ltd. (NSE: ASIANPAINT)** combining historical stock-market analysis, technical indicators, financial-news sentiment analysis using **FinBERT**, and machine-learning-based stock-direction prediction.

The project is built with **Python, Pandas, Plotly, FastAPI, Streamlit, Scikit-learn, XGBoost, and FinBERT**.

---

## 🚀 Project Overview

This project has two main Streamlit applications:

### 1. 📊 Stock Data Analysis Dashboard

The data-analysis dashboard provides an interactive view of the historical performance of Asian Paints.

It includes:

* Latest stock price
* Highest and lowest prices
* Average price
* Total trading days
* Average daily return
* Daily volatility
* Number of up days
* Number of down days
* Closing-price trend
* OHLC/candlestick chart
* Daily-return chart
* 20-day rolling volatility
* Market-direction donut chart
* Historical data table

### 2. 📈 Stock Prediction Dashboard

The prediction dashboard uses a trained machine-learning model to predict the next market direction of Asian Paints.

The prediction system combines:

* Historical stock-market data
* Technical indicators
* Trading volume
* Market/sector features
* Financial-news information
* FinBERT sentiment features

The application provides:

* Prediction date
* **UP / DOWN prediction**
* UP probability
* DOWN probability
* Prediction confidence
* Raw API response

---

# 🧠 Machine Learning Approach

The prediction problem is formulated as a **binary classification problem**.

The model predicts whether the stock is expected to move:

```text
UP
```

or

```text
DOWN
```

The project experiments with machine-learning classification models including:

* Logistic Regression
* XGBoost

The trained XGBoost model is integrated into the prediction application.

---

# 📰 Financial News Sentiment Analysis

One of the main features of this project is the integration of **financial-news sentiment analysis**.

Financial news related to Asian Paints is processed using **FinBERT**, a transformer model specifically designed for financial text.

The resulting sentiment information is converted into numerical features and combined with market-based features.

Examples of news-related features include:

* Number of relevant news articles
* Aggregated sentiment scores
* Financial sentiment information

This allows the prediction pipeline to incorporate both:

```text
Market Data + Technical Indicators + News Sentiment
```

rather than relying only on historical prices.

---

# 📈 Market & Technical Features

The project uses historical stock-market information from Asian Paints.

The analysis includes:

* Open
* High
* Low
* Close
* Volume
* Daily Return
* Rolling Volatility

Technical indicators used in the prediction pipeline include:

* RSI
* MACD
* Bollinger Bands
* ATR

Additional market and sector variables are also incorporated into the machine-learning feature set.

---

# 🖥️ Data Analysis Dashboard

The data-analysis Streamlit application provides a Power BI-inspired dashboard layout.

### Key Metrics

The dashboard displays:

| Metric           | Description                            |
| ---------------- | -------------------------------------- |
| Latest Price     | Most recent Asian Paints closing price |
| Highest Price    | Highest observed price                 |
| Lowest Price     | Lowest observed price                  |
| Average Price    | Average closing price                  |
| Trading Days     | Number of available trading days       |
| Avg Daily Return | Average daily percentage return        |
| Daily Volatility | Daily/rolling volatility measure       |
| Up Days          | Number of positive-return trading days |
| Down Days        | Number of negative-return trading days |

### Visualizations

The dashboard contains:

* 📈 Closing Price Trend
* 📊 OHLC / Candlestick Chart
* 📉 Daily Returns
* 📊 20-Day Rolling Volatility
* 🥯 Market Direction Donut Chart

The dashboard also provides the historical dataset through an expandable table.

---

# 🔮 Stock Prediction Dashboard

The prediction application communicates with the FastAPI backend.

The user can click:

```text
🔮 Predict Asian Paints
```

The application then requests the latest prediction from the API.

The result contains:

```text
Prediction Date
Prediction: UP / DOWN
UP Probability
DOWN Probability
Prediction Confidence
```

For example:

```text
Prediction: UP

UP Probability      72.35%
DOWN Probability    27.65%
```

The probabilities are displayed directly in the Streamlit interface.

---

# 🏗️ Project Architecture

```text
                       Asian Paints
                            │
                            ▼
                  ┌───────────────────┐
                  │ Historical Market │
                  │       Data        │
                  └─────────┬─────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │ Technical /      │         │ Financial News   │
    │ Market Features  │         │                  │
    └────────┬─────────┘         └────────┬─────────┘
             │                            │
             │                            ▼
             │                    ┌──────────────────┐
             │                    │     FinBERT      │
             │                    │ Sentiment Model  │
             │                    └────────┬─────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Combined Feature  │
                  │      Dataset      │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Machine Learning  │
                  │                   │
                  │ Logistic Regression│
                  │ XGBoost           │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │     FastAPI       │
                  │      Backend      │
                  └───────┬─────┬─────┘
                          │     │
             ┌────────────┘     └────────────┐
             ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Data Analysis    │           │ Stock Prediction │
    │   Streamlit      │           │    Streamlit     │
    │    Dashboard     │           │    Dashboard     │
    └──────────────────┘           └──────────────────┘
```

---

# 📂 Project Structure

A possible repository structure is:

```text
asian-paints-stock-analysis/
│
├── dataapp.py
├── prediction.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── asian_paints_xgboost.pkl
│   └── feature_columns.pkl
│
├── data/
│   └── processed_data.csv
│
├── images/
│   └── dashboard.png
│
└── ...
```

> Update the Python filenames above if your actual repository uses different names.

---

# 🔧 Technologies Used

## Programming

* Python

## Data Analysis

* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* XGBoost

## Natural Language Processing

* Hugging Face Transformers
* FinBERT

## Financial Data

* yfinance
* Technical indicators
* Market and sector data

## Visualization

* Plotly
* Streamlit

## Backend

* FastAPI
* Uvicorn

## Model Serialization

* Pickle

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/kririntu/asian-paints-stock-analysis.git
```

Move into the project directory:

```bash
cd asian-paints-stock-analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The project requires the **FastAPI backend** to be running before using the Streamlit applications.

## 1. Start FastAPI

For example:

```bash
uvicorn dataapp:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

The prediction application uses:

```text
http://127.0.0.1:8000/predict
```

The data-analysis application uses:

```text
http://127.0.0.1:8000/dashboard
```

The API health endpoint is:

```text
http://127.0.0.1:8000/health
```

---

## 2. Run the Data Analysis Dashboard

Run:

```bash
streamlit run <data-analysis-file>.py
```

This opens the historical stock-analysis dashboard.

---

## 3. Run the Prediction Dashboard

Run:

```bash
streamlit run <prediction-file>.py
```

This opens the machine-learning prediction interface.

---

# 📊 Dashboard Screenshot



```text
screenshot/1_dash.png
screenshot/2.png
```

Then include it in the README:

```markdown
## 📊 Dashboard

![Asian Paints Data Analysis Dashboard](images/dashboard.png)
```

You can also add a separate screenshot for the prediction dashboard:

```markdown
![Asian Paints Stock Prediction Dashboard](images/prediction.png)
```

---

# 📦 Trained Model

The trained XGBoost model is stored as:

```text
asian_paints_xgboost.pkl
```

The feature configuration used by the model is stored as:

```text
feature_columns.pkl
```

Keeping the feature-column configuration is important because the prediction pipeline must provide the model with the same feature structure used during training.

---

# 🔄 End-to-End Workflow

The complete workflow is:

```text
1. Collect historical Asian Paints data
              ↓
2. Process and clean market data
              ↓
3. Generate technical indicators
              ↓
4. Collect relevant financial news
              ↓
5. Apply FinBERT sentiment analysis
              ↓
6. Combine market + technical + sentiment features
              ↓
7. Train classification models
              ↓
8. Save trained XGBoost model
              ↓
9. Load model through FastAPI
              ↓
10. Generate UP/DOWN prediction
              ↓
11. Display prediction in Streamlit
```

---

# 🎯 Project Objective

The primary objective of this project is to demonstrate how multiple Data Science techniques can be combined into a single real-world application:

```text
Data Collection
       +
Data Cleaning
       +
Feature Engineering
       +
Financial Analysis
       +
NLP / Sentiment Analysis
       +
Machine Learning
       +
REST API
       +
Interactive Dashboard
```

This project demonstrates an end-to-end workflow from raw financial data to an interactive machine-learning application.

---

# ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes only**.

Stock-market predictions are inherently uncertain. The predictions generated by this application should **not be considered financial advice or a guarantee of future stock performance**.

---

# 👨‍💻 Author

**Krishnendu Patra**

Physics Researcher → Data Science / Machine Learning

GitHub: **kririntu**

---

## ⭐ Key Skills Demonstrated

* Python
* Pandas
* NumPy
* Exploratory Data Analysis
* Financial Data Analysis
* Feature Engineering
* Technical Indicators
* NLP
* FinBERT
* Sentiment Analysis
* Logistic Regression
* XGBoost
* FastAPI
* Streamlit
* Plotly
* REST API Integration
* Machine Learning Deployment
* Data Visualization

# Asianpaints-Stock-forecasting-and-data-analysis
