# Omniscience — Sports Betting TA & Forecast Engine

Omniscience is a **sports betting analysis engine** built for professional-grade line reading, technical analysis (TA), and recommendation generation.  
It transforms raw odds into **implied probabilities (IP)**, applies a full TA stack (momentum, RSI, moving averages, volatility, Fibonacci, Greeks, steam detection, etc.), and generates **human-readable betting recommendations** with confidence scoring.  

---

## 🚀 Features

- **Strict Parsing System**
  - Supports **5-line blocks** (spread games: NFL, NBA, etc.)
  - Supports **4-line blocks** (moneyline games: MLB, NHL, etc.)
  - Supports **8-line splits blocks** (bet % and money % data)
  - Toggle switch prevents confusion between formats
  - Handles missing values & "even" odds gracefully
  - All odds auto-converted to **Implied Probability (IP)**

- **TA Engine**
  - Momentum Velocity (MOM-V) & Momentum Acceleration (MOM-A)
  - RSI, ATR, Z-Score
  - Adaptive & Classic Moving Averages
  - Bollinger Band Width
  - Fibonacci retracements & extensions
  - Option Greeks-style risk metrics for odds
  - Steam detection & sharp money signals

- **Line Movement Forecast (LMF)**
  - Projects short-term spread/total movement in **points**
  - Projects moneyline movement across key IP thresholds
  - Generates confidence intervals for forecasted moves

- **Recommendation Engine**
  - Final outputs: **Back**, **Fade**, or **Hold**
  - Includes narrative explanation of market pressure
  - Calculates **Expected Value (EV)** & **Kelly stake sizing**
  - Confidence tiers: Strong / Moderate / Weak
  - Sharp/steam confirmations integrated

- **Streamlit App**
  - Paste in odds manually for parsing
  - Separate paste box for splits
  - Toggle parser mode (4-line / 5-line / splits)
  - Outputs parsed tables, TA diagnostics, forecasts, and recs
  - Interactive charts of line movement in IP space

---

## 📂 Project Structure
