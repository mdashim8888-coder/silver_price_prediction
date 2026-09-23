# 🥈 Silver Price Prediction Dashboard

A Streamlit web application that predicts the next-day silver closing price using an **XGBoost Regressor** trained on 500 trading days of historical data (October 2024 – September 2026).

---

## 📌 Features

- **Live next-day price prediction** with change indicator vs. last known close
- **Interactive OHLC candlestick chart** with MA7 / MA20 overlays
- **RSI 14 & Volume subplots** (toggleable from the sidebar)
- **Actual vs. Predicted** comparison chart on the test split
- **Feature importance** horizontal bar chart
- **Raw dataset viewer** with date-range filtering
- Model performance metrics: **MAE**, **RMSE**, **R²**

---

## 🗂️ Project Structure

```
silver_price/
├── silver_price_app.py          # Main application (all modules inlined)
├── silver_price_500_dataset.csv # 500-row historical silver price dataset
├── requirements.txt             # Python dependencies
├── models/
│   └── silver_model.joblib      # Saved XGBoost model (auto-generated on first run)
└── README.md
```

---

## ⚙️ Setup & Installation

1. **Clone / download** the repository.

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the App

```bash
streamlit run silver_price_app.py
```

The app will open in your browser at `http://localhost:8501`.

> **First run:** The model will be trained automatically and saved to `models/silver_model.joblib`. Subsequent runs load the cached model instantly.

---

## 🧠 Model Details

| Property        | Value                          |
|-----------------|-------------------------------|
| Algorithm       | XGBoost Regressor              |
| Train/Test Split| 80 / 20 chronological          |
| Estimators      | 200                            |
| Learning Rate   | 0.05                           |
| Max Depth       | 5                              |
| Target          | `Target_Next_Close` (next-day close price) |

### Input Features

| Feature      | Description                        |
|--------------|------------------------------------|
| `Open`       | Opening price                      |
| `High`       | Daily high price                   |
| `Low`        | Daily low price                    |
| `Close`      | Closing price                      |
| `Volume`     | Trading volume                     |
| `Change_Pct` | Daily percentage change            |
| `MA7`        | 7-day moving average               |
| `MA20`       | 20-day moving average              |
| `RSI14`      | 14-period Relative Strength Index  |

---

## 📦 Dependencies

| Package       | Purpose                        |
|---------------|-------------------------------|
| `streamlit`   | Web dashboard framework        |
| `xgboost`     | Gradient boosting model        |
| `pandas`      | Data loading & manipulation    |
| `numpy`       | Numerical operations           |
| `scikit-learn`| Model evaluation metrics       |
| `plotly`      | Interactive charts             |
| `joblib`      | Model serialisation            |
