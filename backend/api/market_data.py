"""Provider abstraction for free market-data calls.

The first provider uses yfinance because it is the easiest free connection for
NSE `.NS` symbols. More providers can be added without changing the UI.
"""

from typing import Protocol

import pandas as pd

from utils.stock_data import StockDataFetcher


class MarketDataProvider(Protocol):
    """Interface all market-data providers should implement."""

    def history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Return historical OHLCV candles for a symbol."""

    def profile(self, symbol: str) -> dict:
        """Return company profile and valuation metadata."""


class YahooFinanceProvider:
    """Free Yahoo Finance provider for NSE symbols such as RELIANCE.NS."""

    def __init__(self, fetcher: StockDataFetcher | None = None):
        self.fetcher = fetcher or StockDataFetcher()

    def history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        return self.fetcher.fetch_stock_data(symbol, period=period, interval=interval)

    def profile(self, symbol: str) -> dict:
        return self.fetcher.get_stock_info(symbol)


def get_market_data_provider(name: str = "yahoo_finance") -> MarketDataProvider:
    """Factory for the easiest free API connection used by the app."""
    if name != "yahoo_finance":
        raise ValueError(f"Unsupported provider: {name}")
    return YahooFinanceProvider()
