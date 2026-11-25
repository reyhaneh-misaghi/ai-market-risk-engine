import sqlite3
from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats


class QuantitativeRiskEngine:
    """Computes Historical, Parametric, and Monte Carlo VaR/CVaR and Backtesting."""

    def __init__(self, db_path: str = "data/market_risk.db"):
        self.db_path = db_path
        self.returns_df = self._load_returns()

    def _load_returns(self) -> pd.DataFrame:
        """Queries log returns from SQLite and pivots into a clean matrix."""
        query = """
            SELECT date, ticker, log_return
            FROM market_prices
            ORDER BY date ASC
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)

        pivot_df = df.pivot(index="date", columns="ticker", values="log_return").dropna()
        return pivot_df

    def compute_portfolio_returns(self, weights: Dict[str, float]) -> pd.Series:
        """Calculates daily portfolio log returns given asset weights."""
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("Portfolio weights must sum to 1.0")
        return self.returns_df[tickers].dot(w)

    def historical_var_cvar(self, port_returns: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
        """Non-parametric Historical VaR and CVaR (Expected Shortfall)."""
        alpha = 1.0 - confidence
        var = -np.percentile(port_returns, alpha * 100)
        cvar = -port_returns[port_returns <= -var].mean()
        return {"VaR": float(var), "CVaR": float(cvar)}

    def parametric_var_cvar(self, weights: Dict[str, float], confidence: float = 0.95) -> Dict[str, float]:
        """Variance-Covariance Parametric VaR and CVaR assuming normality."""
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])
        sub_returns = self.returns_df[tickers]

        mu_p = float(sub_returns.mean().dot(w))
        cov_matrix = sub_returns.cov().values
        sigma_p = float(np.sqrt(w.T @ cov_matrix @ w))

        alpha = 1.0 - confidence
        z_score = stats.norm.ppf(alpha)
        var = -(mu_p + z_score * sigma_p)
        cvar = -mu_p + sigma_p * (stats.norm.pdf(z_score) / alpha)
        return {"VaR": float(var), "CVaR": float(cvar)}

    def monte_carlo_var_cvar(
        self, weights: Dict[str, float], confidence: float = 0.95, n_sims: int = 10000, seed: int = 42
    ) -> Dict[str, float]:
        """Correlated Monte Carlo simulation using Cholesky decomposition."""
        np.random.seed(seed)
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])
        sub_returns = self.returns_df[tickers]

        mu = sub_returns.mean().values
        cov_matrix = sub_returns.cov().values
        cholesky_l = np.linalg.cholesky(cov_matrix)

        # Generate correlated standard normal shocks
        z = np.random.normal(size=(n_sims, len(tickers)))
        simulated_returns = mu + z @ cholesky_l.T
        sim_port_returns = simulated_returns @ w

        alpha = 1.0 - confidence
        var = -np.percentile(sim_port_returns, alpha * 100)
        cvar = -sim_port_returns[sim_port_returns <= -var].mean()
        return {"VaR": float(var), "CVaR": float(cvar)}

    def kupiec_backtest(self, port_returns: pd.Series, var_estimate: float, confidence: float = 0.95) -> Dict[str, float]:
        """Kupiec POF (Proportion of Failures) backtest for VaR model validation."""
        exceptions = int((port_returns < -var_estimate).sum())
        total_obs = len(port_returns)
        expected_rate = 1.0 - confidence
        observed_rate = exceptions / total_obs

        # Likelihood Ratio test statistic
        if exceptions == 0:
            lr_pof = -2 * total_obs * np.log(1 - expected_rate)
        else:
            lr_pof = -2 * (
                (total_obs - exceptions) * np.log((1 - expected_rate) / (1 - observed_rate))
                + exceptions * np.log(expected_rate / observed_rate)
            )
        p_value = float(1 - stats.chi2.cdf(lr_pof, df=1))
        return {
            "exceptions": exceptions,
            "total_days": total_obs,
            "observed_fail_rate": round(observed_rate * 100, 2),
            "kupiec_p_value": round(p_value, 4),
            "model_accepted": p_value > 0.05,
        }


if __name__ == "__main__":
    # Sample balanced portfolio across Equity, Energy, and Commodities
    SAMPLE_WEIGHTS = {
        "SPY": 0.30,
        "FTSEMIB.MI": 0.25,
        "CL=F": 0.20,
        "NG=F": 0.15,
        "GC=F": 0.10,
    }

    engine = QuantitativeRiskEngine()
    port_ret = engine.compute_portfolio_returns(SAMPLE_WEIGHTS)

    hist = engine.historical_var_cvar(port_ret, confidence=0.95)
    param = engine.parametric_var_cvar(SAMPLE_WEIGHTS, confidence=0.95)
    mc = engine.monte_carlo_var_cvar(SAMPLE_WEIGHTS, confidence=0.95)
    bt = engine.kupiec_backtest(port_ret, hist["VaR"], confidence=0.95)

    print("=== 1-Day 95% Risk Metrics ===")
    print(f"Historical  -> VaR: {hist['VaR']:.2%}, CVaR: {hist['CVaR']:.2%}")
    print(f"Parametric  -> VaR: {param['VaR']:.2%}, CVaR: {param['CVaR']:.2%}")
    print(f"Monte Carlo -> VaR: {mc['VaR']:.2%}, CVaR: {mc['CVaR']:.2%}")
    print("\n=== Kupiec Backtest (Historical VaR) ===")
    print(bt)