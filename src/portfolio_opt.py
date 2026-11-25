import asyncio
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.risk_engine import QuantitativeRiskEngine
from src.sentiment_ai import AsyncSentimentEngine, NewsSentimentSignal


class SentimentPortfolioOptimizer:
    """Combines historical covariance with LLM sentiment signals to tilt portfolio weights."""

    def __init__(self, risk_engine: QuantitativeRiskEngine):
        self.risk_engine = risk_engine
        self.returns_df = risk_engine.returns_df

    def compute_sentiment_adjusted_returns(
        self,
        tickers: list[str],
        signals: Dict[str, NewsSentimentSignal],
        sentiment_weight: float = 0.15,
    ) -> np.ndarray:
        """Tilts historical mean returns using confidence-weighted LLM sentiment scores."""
        sub_returns = self.returns_df[tickers]
        hist_mu = sub_returns.mean().values
        daily_vol = sub_returns.std().values

        adjusted_mu = hist_mu.copy()
        for idx, ticker in enumerate(tickers):
            if ticker in signals:
                sig = signals[ticker]
                tilt = sentiment_weight * sig.sentiment_score * sig.confidence * daily_vol[idx]
                adjusted_mu[idx] += tilt

        return adjusted_mu

    def optimize_weights(
        self,
        base_weights: Dict[str, float],
        signals: Dict[str, NewsSentimentSignal],
        sentiment_weight: float = 0.15,
        min_weight: float = 0.05,
        max_weight: float = 0.40,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Optimizes portfolio weights maximizing sentiment-adjusted Sharpe ratio."""
        tickers = list(base_weights.keys())
        n_assets = len(tickers)
        w0 = np.array([base_weights[t] for t in tickers])

        adj_mu = self.compute_sentiment_adjusted_returns(tickers, signals, sentiment_weight)
        cov_matrix = self.returns_df[tickers].cov().values

        def neg_sharpe(weights: np.ndarray) -> float:
            port_ret = np.dot(weights, adj_mu) * 252
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
            return -port_ret / (port_vol + 1e-8)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        opt_w = result.x if result.success else w0
        opt_weights_dict = {t: round(float(w), 4) for t, w in zip(tickers, opt_w)}

        comparison_df = pd.DataFrame({
            "Asset": tickers,
            "Base_Weight": [f"{base_weights[t]:.1%}" for t in tickers],
            "AI_Sentiment": [f"{signals[t].sentiment_score:+.2f}" if t in signals else "0.00" for t in tickers],
            "Optimized_Weight": [f"{opt_weights_dict[t]:.1%}" for t in tickers],
            "Weight_Change": [f"{(opt_weights_dict[t] - base_weights[t]):+.1%}" for t in tickers],
        })

        return opt_weights_dict, comparison_df


if __name__ == "__main__":
    BASE_WEIGHTS = {
        "SPY": 0.30,
        "FTSEMIB.MI": 0.25,
        "CL=F": 0.20,
        "NG=F": 0.15,
        "GC=F": 0.10,
    }

    SAMPLE_NEWS = [
        {"ticker": "SPY", "headline": "US tech earnings beat expectations amid resilient consumer demand growth"},
        {"ticker": "FTSEMIB.MI", "headline": "European industrial output faces recession risk as borrowing costs stay elevated"},
        {"ticker": "CL=F", "headline": "Global crude oil prices drop sharply following unexpected surge in supply"},
        {"ticker": "NG=F", "headline": "European natural gas faces severe supply disruption and geopolitical sanctions"},
        {"ticker": "GC=F", "headline": "Gold prices rally to record high as investors hedge geopolitical risk"},
    ]

    risk_eng = QuantitativeRiskEngine()
    sent_eng = AsyncSentimentEngine()
    signals = asyncio.run(sent_eng.analyze_batch(SAMPLE_NEWS))

    optimizer = SentimentPortfolioOptimizer(risk_eng)
    opt_weights, summary_table = optimizer.optimize_weights(BASE_WEIGHTS, signals)

    base_mc = risk_eng.monte_carlo_var_cvar(BASE_WEIGHTS)
    opt_mc = risk_eng.monte_carlo_var_cvar(opt_weights)

    print("=== AI-Driven Portfolio Rebalancing ===")
    print(summary_table.to_string(index=False))
    print("\n=== Risk Impact (Monte Carlo 95%) ===")
    print(f"Before AI Tilt -> VaR: {base_mc['VaR']:.2%}, CVaR: {base_mc['CVaR']:.2%}")
    print(f"After  AI Tilt -> VaR: {opt_mc['VaR']:.2%}, CVaR: {opt_mc['CVaR']:.2%}")