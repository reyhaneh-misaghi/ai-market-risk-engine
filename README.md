# AI-Driven Market Risk & Sentiment Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reyhaneh-risk-engine.streamlit.app)

An end-to-end quantitative market risk management and LLM-powered portfolio optimization system built with **Python, SQLite, SciPy, Pydantic, Groq LLM, and Streamlit**.

🔗 **Live Interactive Dashboard:** [reyhaneh-risk-engine.streamlit.app](https://reyhaneh-risk-engine.streamlit.app)

---

## 📌 Project Overview & Architecture

This project bridges classical quantitative risk modeling with modern Natural Language Processing (NLP) to dynamically adjust portfolio allocations in response to real-time macroeconomic and commodity news.

1. **Automated Data Pipeline (`src/data_loader.py`):** Fetches 3 years of historical daily market data via `yfinance` across 5 core cross-asset benchmarks (`SPY`, `FTSEMIB.MI`, Crude Oil `CL=F`, Natural Gas `NG=F`, Gold `GC=F`) and persists structured time-series tables in a relational **SQLite** database.
2. **Quantitative Risk Engine (`src/risk_engine.py`):** Computes 1-day portfolio Value-at-Risk (**VaR**) and Expected Shortfall (**CVaR**) across configurable confidence levels (90%–99%) and validates model accuracy via statistical backtesting.
3. **Asynchronous LLM Sentiment Engine (`src/sentiment_ai.py`):** Uses `asyncio` and **Groq LLM** (`llama-3.3-70b-versatile`) with strict **Pydantic** schema validation to extract structured sentiment scores (`[-1, +1]`), market regimes (`Bullish`, `Bearish`, `Neutral`), and confidence metrics from financial news headlines.
4. **Black-Litterman-Inspired Optimizer (`src/portfolio_opt.py`):** Blends historical expected returns with AI sentiment views (controlled by sensitivity parameter $\lambda$) and solves a constrained Sharpe-ratio maximization problem via **SciPy SLSQP**.
5. **Interactive Analytics Dashboard (`app.py`):** Built with **Streamlit** and **Plotly** for real-time parameter tuning, distribution visualization, and portfolio stress testing.

---

## 📊 Key Quantitative Comparisons & Findings

### 1. Risk Methodology Comparison: Historical vs. Parametric vs. Monte Carlo
* **Parametric (Variance-Covariance):** Assumes multivariate normality, which systematically underestimates tail losses during energy and equity volatility spikes.
* **Historical Simulation:** Captures empirical skewness and fat tails directly from historical return distributions without distributional assumptions.
* **Correlated Monte Carlo Simulation:** Generates 10,000+ correlated return scenarios using **Cholesky decomposition** of the empirical covariance matrix.
* **Model Validation (Kupiec POF Backtest):** Evaluates exceedance ratios against expected failure rates using Likelihood Ratio ($\text{LR}_{\text{POF}}$) testing and $p$-values ($p > 0.05$ acceptance threshold) to verify tail-risk calibration.

### 2. Portfolio Comparison: Base Allocation vs. AI-Adjusted Portfolio
* **Base Portfolio:** Static user-defined weights evaluated solely on backward-looking historical covariance.
* **AI-Optimized Portfolio:** Dynamically tilts expected returns using structured LLM news signals before re-optimizing weights under box constraints ($w_i \in [0.05, 0.40]$, $\sum w_i = 1$).
* **Impact:** Under adverse energy or European equity headlines (e.g., bearish shocks on `CL=F` or `FTSEMIB.MI`), the engine systematically reduces exposure to negatively impacted assets and reallocates toward defensive/bullish assets (such as `GC=F`), lowering overall portfolio **VaR and CVaR**.

---

## 🛠️ Tech Stack
* **Quantitative & Data:** `Python`, `NumPy`, `Pandas`, `SciPy`, `SQLite`, `yfinance`
* **AI & Validation:** `Groq API`, `asyncio`, `Pydantic`
* **Visualization & Deployment:** `Streamlit`, `Plotly`, `GitHub`
