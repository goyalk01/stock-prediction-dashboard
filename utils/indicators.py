"""
Technical Indicators Calculation Module
Implements popular trading indicators for stock analysis
"""

import pandas as pd
import numpy as np
from typing import Union
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Comprehensive technical indicators calculator"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        return pd.DataFrame({
            'SMA': sma,
            'Upper_Band': sma + (std * std_dev),
            'Lower_Band': sma - (std * std_dev)
        })
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'K%': k_percent,
            'D%': d_percent
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> dict:
        """Fibonacci Retracement Levels"""
        diff = high - low
        levels = {
            '0%': high,
            '23.6%': high - (0.236 * diff),
            '38.2%': high - (0.382 * diff),
            '50%': high - (0.5 * diff),
            '61.8%': high - (0.618 * diff),
            '100%': low
        }
        return levels
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=9).max()
        tenkan_low = low.rolling(window=9).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=26).max()
        kijun_low = low.rolling(window=26).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=52).max()
        senkou_low = low.rolling(window=52).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        return pd.DataFrame({
            'Tenkan_sen': tenkan_sen,
            'Kijun_sen': kijun_sen,
            'Senkou_span_a': senkou_span_a,
            'Senkou_span_b': senkou_span_b,
            'Chikou_span': chikou_span
        })

class PatternRecognition:
    """Chart pattern recognition"""
    
    @staticmethod
    def detect_golden_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
        """Detect Golden Cross pattern (short MA crosses above long MA)"""
        golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        return golden_cross
    
    @staticmethod
    def detect_death_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
        """Detect Death Cross pattern (short MA crosses below long MA)"""
        death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        return death_cross
    
    @staticmethod
    def detect_breakouts(data: pd.Series, window: int = 20, threshold: float = 0.02) -> pd.Series:
        """Detect price breakouts"""
        rolling_max = data.rolling(window=window).max()
        rolling_min = data.rolling(window=window).min()
        
        upper_breakout = data > rolling_max.shift(1) * (1 + threshold)
        lower_breakout = data < rolling_min.shift(1) * (1 - threshold)
        
        return upper_breakout | lower_breakout
    
    @staticmethod
    def detect_doji(open_price: pd.Series, high: pd.Series, low: pd.Series, 
                    close: pd.Series, threshold: float = 0.001) -> pd.Series:
        """Detect Doji candlestick pattern"""
        body = abs(close - open_price)
        range_size = high - low
        
        # Doji when body is very small relative to the range
        doji = body <= (range_size * threshold)
        return doji

class RiskMetrics:
    """Risk analysis and metrics"""
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate stock beta"""
        try:
            covariance = np.cov(stock_returns.dropna(), market_returns.dropna())[0][1]
            market_variance = np.var(market_returns.dropna())
            return covariance / market_variance
        except:
            return 1.0
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        try:
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std()
            return excess_returns.mean() / tracking_error if tracking_error != 0 else 0
        except:
            return 0
    
    @staticmethod
    def calculate_treynor_ratio(portfolio_returns: pd.Series, 
                              risk_free_rate: float, beta: float) -> float:
        """Calculate Treynor Ratio"""
        try:
            excess_return = portfolio_returns.mean() - risk_free_rate / 252
            return excess_return / beta if beta != 0 else 0
        except:
            return 0

# Example usage and comprehensive indicator calculation
def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for a stock dataset"""
    indicators_data = data.copy()
    ti = TechnicalIndicators()
    
    # Moving Averages
    indicators_data['SMA_20'] = ti.sma(data['Close'], 20)
    indicators_data['SMA_50'] = ti.sma(data['Close'], 50)
    indicators_data['EMA_12'] = ti.ema(data['Close'], 12)
    indicators_data['EMA_26'] = ti.ema(data['Close'], 26)
    
    # Momentum Indicators
    indicators_data['RSI'] = ti.rsi(data['Close'])
    macd_data = ti.macd(data['Close'])
    indicators_data = pd.concat([indicators_data, macd_data], axis=1)
    
    # Volatility Indicators
    bb_data = ti.bollinger_bands(data['Close'])
    indicators_data = pd.concat([indicators_data, bb_data], axis=1)
    
    # Volume Indicators
    indicators_data['OBV'] = ti.obv(data['Close'], data['Volume'])
    
    # Pattern Recognition
    pr = PatternRecognition()
    indicators_data['Golden_Cross'] = pr.detect_golden_cross(
        indicators_data['SMA_20'], indicators_data['SMA_50']
    )
    indicators_data['Death_Cross'] = pr.detect_death_cross(
        indicators_data['SMA_20'], indicators_data['SMA_50']
    )
    
    return indicators_data

# Testing
if __name__ == "__main__":
    print("Testing Technical Indicators...")
    print("=" * 40)
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data
    sample_data = pd.DataFrame({
        'Open': 100 + np.random.randn(100).cumsum() * 0.5,
        'High': 100 + np.random.randn(100).cumsum() * 0.5 + np.abs(np.random.randn(100)) * 2,
        'Low': 100 + np.random.randn(100).cumsum() * 0.5 - np.abs(np.random.randn(100)) * 2,
        'Close': 100 + np.random.randn(100).cumsum() * 0.5,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Adjust High and Low to be realistic
    sample_data['High'] = np.maximum(sample_data[['Open', 'Close']].max(axis=1), sample_data['High'])
    sample_data['Low'] = np.minimum(sample_data[['Open', 'Close']].min(axis=1), sample_data['Low'])
    
    # Test indicators
    ti = TechnicalIndicators()
    
    print("✅ SMA calculation:", not ti.sma(sample_data['Close'], 20).isna().all())
    print("✅ RSI calculation:", not ti.rsi(sample_data['Close']).isna().all())
    print("✅ MACD calculation:", len(ti.macd(sample_data['Close']).columns) == 3)
    print("✅ Bollinger Bands:", len(ti.bollinger_bands(sample_data['Close']).columns) == 3)
    
    print("\n✨ Technical indicators module ready!")