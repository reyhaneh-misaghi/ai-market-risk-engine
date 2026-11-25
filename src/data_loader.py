import sqlite3
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataPipeline:
    """Pipeline for fetching historical market data and storing it in SQLite."""

    def __init__(self, db_path: str = "data/market_risk.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes normalized SQL tables for assets and historical prices."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    ticker TEXT PRIMARY KEY,
                    asset_class TEXT NOT NULL,
                    description TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_prices (
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    adj_close REAL NOT NULL,
                    log_return REAL,
                    PRIMARY KEY (date, ticker),
                    FOREIGN KEY (ticker) REFERENCES assets (ticker)
                )
            """)
            conn.commit()

    def register_assets(self, assets_metadata: List[tuple]) -> None:
        """Inserts or updates asset metadata in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO assets (ticker, asset_class, description)
                VALUES (?, ?, ?)
                """,
                assets_metadata,
            )
            conn.commit()

    def fetch_and_store(self, tickers: List[str], period: str = "3y") -> None:
        """Downloads market data, computes log returns, and upserts into SQL."""
        raw_data = yf.download(tickers, period=period, auto_adjust=False, progress=False)
        adj_close_df = raw_data["Adj Close"]

        records = []
        for ticker in tickers:
            series = adj_close_df[ticker].dropna()
            log_returns = np.log(series / series.shift(1))

            df_ticker = pd.DataFrame({
                "date": series.index.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "adj_close": series.values,
                "log_return": log_returns.values,
            }).dropna()

            records.extend(df_ticker.to_records(index=False).tolist())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO market_prices (date, ticker, adj_close, log_return)
                VALUES (?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()
        print(f"Successfully stored {len(records)} rows into {self.db_path}.")


if __name__ == "__main__":
    PORTFOLIO_ASSETS = [
        ("SPY", "Equity", "S&P 500 ETF"),
        ("FTSEMIB.MI", "Equity", "FTSE MIB Index Milan"),
        ("CL=F", "Energy", "WTI Crude Oil Futures"),
        ("NG=F", "Energy", "Natural Gas Futures"),
        ("GC=F", "Commodity", "Gold Futures"),
    ]

    pipeline = MarketDataPipeline()
    pipeline.register_assets(PORTFOLIO_ASSETS)
    pipeline.fetch_and_store([asset[0] for asset in PORTFOLIO_ASSETS], period="3y")