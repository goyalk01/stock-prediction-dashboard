"""
Stock Data Fetching and Processing Utilities
Handles real-time and historical stock data retrieval
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockDataFetcher:
    """Comprehensive stock data fetching utility"""
    
    def __init__(self):
        self.cache = {}
        
    def fetch_stock_data(self, symbol: str, period: str = "1y", 
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data using Yahoo Finance
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with stock data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Clean and prepare data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Add additional columns
            data['Symbol'] = symbol
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract key information
            stock_info = {
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'price': info.get('currentPrice', 0),
                'target_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationKey', 'hold')
            }
            
            return stock_info
            
        except Exception as e:
            return {'error': f"Could not fetch info for {symbol}: {str(e)}"}
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        stock_data = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, period)
                stock_data[symbol] = data
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
                continue
                
        return stock_data
    
    def get_realtime_price(self, symbol: str) -> float:
        """Get current stock price"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 0.0
        except:
            return 0.0
    
    def calculate_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Calculate correlation matrix between stocks"""
        try:
            # Fetch data for all symbols
            price_data = pd.DataFrame()
            
            for symbol in symbols:
                data = self.fetch_stock_data(symbol, period)
                price_data[symbol] = data['Close']
            
            # Calculate correlation matrix
            correlation_matrix = price_data.corr()
            return correlation_matrix
            
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

class MarketAnalyzer:
    """Market analysis and insights"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns.mean() * 252 - risk_free_rate
            volatility = returns.std() * np.sqrt(252)
            return excess_returns / volatility if volatility != 0 else 0
        except:
            return 0
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except:
            return 0
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        try:
            return np.percentile(returns.dropna(), confidence_level * 100)
        except:
            return 0
    
    @staticmethod
    def identify_support_resistance(data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Identify support and resistance levels"""
        try:
            high_data = data['High'].rolling(window=window).max()
            low_data = data['Low'].rolling(window=window).min()
            
            resistance = high_data.iloc[-window:].max()
            support = low_data.iloc[-window:].min()
            
            return support, resistance
        except:
            return 0.0, 0.0

# Example usage and testing
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = StockDataFetcher()
    analyzer = MarketAnalyzer()
    
    # Test with popular stocks
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    print("Testing Stock Data Fetcher...")
    print("=" * 40)
    
    # Test single stock
    try:
        aapl_data = fetcher.fetch_stock_data("AAPL", period="3mo")
        print(f"✅ AAPL data fetched: {len(aapl_data)} rows")
        
        # Test stock info
        aapl_info = fetcher.get_stock_info("AAPL")
        print(f"✅ AAPL info: {aapl_info['name']}")
        
        # Test real-time price
        current_price = fetcher.get_realtime_price("AAPL")
        print(f"✅ AAPL current price: ${current_price:.2f}")
        
        # Test analysis
        returns = aapl_data['Returns'].dropna()
        sharpe = analyzer.calculate_sharpe_ratio(returns)
        max_dd = analyzer.calculate_max_drawdown(aapl_data['Close'])
        
        print(f"✅ AAPL Sharpe Ratio: {sharpe:.2f}")
        print(f"✅ AAPL Max Drawdown: {max_dd:.2%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✨ Stock data utilities ready for dashboard integration!")