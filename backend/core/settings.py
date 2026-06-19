"""Central app settings for a free, India-centric deployment."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    market: str = "India"
    currency_symbol: str = "₹"
    risk_free_rate: float = 0.07
    trading_days: int = 252
    cache_ttl_seconds: int = 600
    stock_universe_path: Path = Path("data/config/stock_symbols.csv")
    sectors_path: Path = Path("data/config/sectors.json")
    default_provider: str = "yahoo_finance"


settings = AppSettings()
