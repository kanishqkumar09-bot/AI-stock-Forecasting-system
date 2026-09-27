# AI Financial Intelligence

An AI-powered financial analysis project that uses Machine Learning
to analyze company growth, predict stock prices, evaluate risk,
and generate investment recommendations.

---

## 🚀 Project Overview

AI Financial Intelligence combines financial data, technical
indicators, and Machine Learning models to provide a simple
financial analysis system.

The project contains four major components:

1. Stock Price Prediction
2. Company Growth Prediction
3. Risk Analysis
4. Investment Recommendation

---

## ✨ Features

### 📈 Stock Price Prediction

Predicts the next-day stock price using financial and technical
indicators.

Features include:

- Daily Return
- 5-Day Return
- 20-Day Return
- SMA 20
- SMA 50
- EMA 20
- RSI 14
- MACD
- MACD Signal
- MACD Histogram
- Volatility
- Volume Change
- Price vs SMA20
- Price vs SMA50

---

### 🏢 Company Growth Prediction

Predicts next-year revenue growth using company financial
information.

Features include:

- Revenue Growth
- Profit Growth
- EPS Growth
- EBIT Margin
- Net Profit Margin
- Revenue Growth Change
- Profit Growth Change
- EPS Growth Change

---

### ⚠️ Risk Analysis

Analyzes stock risk using:

- Volatility
- RSI
- Moving averages
- Market trend

The system classifies risk as:

- LOW
- MODERATE
- HIGH

---

### 💡 Investment Recommendation

Combines:

- Company growth
- Market trend
- Risk level

to generate a simple recommendation:

- BUY
- HOLD
- AVOID

---

## 🤖 Machine Learning Models

The project compares three regression models:

- Linear Regression
- Random Forest
- XGBoost

Model performance is evaluated using:

- MAE
- RMSE
- R² Score

The best-performing model is selected based on test performance.

---

## 📊 Project Structure

```text
AI-Financial-Intelligence/
│
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── processed/
│
├── models/
│   ├── company_growth/
│   └── stock_forecasting/
│
├── src/
│   ├── models/
│   │   ├── company_growth/
│   │   └── stock_forecasting/
│   │
│   ├── prediction/
│   │   ├── predict_stock.py
│   │   └── predict_company_growth.py
│   │
│   ├── risk/
│   │   └── risk_analysis.py
│   │
│   └── recommendation/
│       └── investment_recommendation.py
│
└── tests/