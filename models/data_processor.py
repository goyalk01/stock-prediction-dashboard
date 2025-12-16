"""
Data Processing Module for Stock Market Prediction
Comprehensive data cleaning, feature engineering, and preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class StockDataProcessor:
    """Comprehensive stock data processor for ML models"""
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize data processor
        
        Args:
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
        """
        self.scaler_type = scaler_type
        self.feature_scaler = None
        self.target_scaler = None
        self._setup_scalers()
    
    def _setup_scalers(self):
        """Setup scalers based on type"""
        if self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            raise ValueError("Scaler type must be 'minmax', 'standard', or 'robust'")
    
    def clean_data(self, data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Clean and prepare stock data
        
        Args:
            data: Raw stock data DataFrame
            verbose: Print cleaning information
            
        Returns:
            Cleaned DataFrame
        """
        if verbose:
            print(f"📊 Starting data cleaning...")
            print(f"   Original shape: {data.shape}")
        
        # Make a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Remove duplicate dates
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
        
        # Handle missing values
        initial_nulls = cleaned_data.isnull().sum().sum()
        
        if initial_nulls > 0:
            if verbose:
                print(f"   Found {initial_nulls} missing values")
            
            # Forward fill then backward fill for small gaps
            cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
            
            # Drop remaining rows with missing values
            cleaned_data = cleaned_data.dropna()
        
        # Remove outliers (prices that change more than 50% in a day - likely data errors)
        if 'Close' in cleaned_data.columns:
            daily_returns = cleaned_data['Close'].pct_change()
            outlier_mask = (daily_returns.abs() > 0.5)
            
            if outlier_mask.sum() > 0:
                if verbose:
                    print(f"   Removed {outlier_mask.sum()} outlier days")
                cleaned_data = cleaned_data[~outlier_mask]
        
        # Ensure positive prices and volumes
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in cleaned_data.columns:
                cleaned_data = cleaned_data[cleaned_data[col] > 0]
        
        if 'Volume' in cleaned_data.columns:
            cleaned_data = cleaned_data[cleaned_data['Volume'] >= 0]
        
        # Sort by date
        cleaned_data = cleaned_data.sort_index()
        
        if verbose:
            print(f"   Final shape: {cleaned_data.shape}")
            print(f"   ✅ Data cleaning complete!")
        
        return cleaned_data
    
    def add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic financial features
        
        Args:
            data: Clean stock data
            
        Returns:
            DataFrame with added features
        """
        feature_data = data.copy()
        
        # Price-based features
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Daily returns
            feature_data['Returns'] = feature_data['Close'].pct_change()
            
            # Price range
            feature_data['Price_Range'] = (feature_data['High'] - feature_data['Low']) / feature_data['Close']
            
            # Gap between open and previous close
            feature_data['Gap'] = (feature_data['Open'] - feature_data['Close'].shift(1)) / feature_data['Close'].shift(1)
            
            # Body size (candle body as percentage of price)
            feature_data['Body_Size'] = abs(feature_data['Close'] - feature_data['Open']) / feature_data['Close']
            
            # Upper and lower shadows
            feature_data['Upper_Shadow'] = (feature_data['High'] - np.maximum(feature_data['Open'], feature_data['Close'])) / feature_data['Close']
            feature_data['Lower_Shadow'] = (np.minimum(feature_data['Open'], feature_data['Close']) - feature_data['Low']) / feature_data['Close']
        
        # Volume-based features
        if 'Volume' in data.columns:
            # Volume change
            feature_data['Volume_Change'] = feature_data['Volume'].pct_change()
            
            # Price-volume trend
            if 'Returns' in feature_data.columns:
                feature_data['Price_Volume_Trend'] = feature_data['Returns'] * feature_data['Volume_Change']
        
        return feature_data
    
    def add_moving_averages(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Add moving average features
        
        Args:
            data: Stock data
            windows: List of moving average windows
            
        Returns:
            DataFrame with moving averages
        """
        ma_data = data.copy()
        
        for window in windows:
            if window <= len(data):
                # Simple Moving Average
                ma_data[f'SMA_{window}'] = ma_data['Close'].rolling(window=window).mean()
                
                # Exponential Moving Average
                ma_data[f'EMA_{window}'] = ma_data['Close'].ewm(span=window).mean()
                
                # Price relative to MA
                ma_data[f'Price_to_SMA_{window}'] = ma_data['Close'] / ma_data[f'SMA_{window}']
                
                # MA slope (trend strength)
                ma_data[f'SMA_{window}_Slope'] = ma_data[f'SMA_{window}'].diff()
        
        return ma_data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators
        
        Args:
            data: Stock data with OHLCV
            
        Returns:
            DataFrame with technical indicators
        """
        tech_data = data.copy()
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        tech_data['RSI'] = calculate_rsi(tech_data['Close'])
        
        # MACD
        ema_12 = tech_data['Close'].ewm(span=12).mean()
        ema_26 = tech_data['Close'].ewm(span=26).mean()
        tech_data['MACD'] = ema_12 - ema_26
        tech_data['MACD_Signal'] = tech_data['MACD'].ewm(span=9).mean()
        tech_data['MACD_Histogram'] = tech_data['MACD'] - tech_data['MACD_Signal']
        
        # Bollinger Bands
        bb_window = 20
        bb_std_dev = 2
        tech_data['BB_Middle'] = tech_data['Close'].rolling(window=bb_window).mean()
        bb_std = tech_data['Close'].rolling(window=bb_window).std()
        tech_data['BB_Upper'] = tech_data['BB_Middle'] + (bb_std * bb_std_dev)
        tech_data['BB_Lower'] = tech_data['BB_Middle'] - (bb_std * bb_std_dev)
        tech_data['BB_Width'] = (tech_data['BB_Upper'] - tech_data['BB_Lower']) / tech_data['BB_Middle']
        tech_data['BB_Position'] = (tech_data['Close'] - tech_data['BB_Lower']) / (tech_data['BB_Upper'] - tech_data['BB_Lower'])
        
        # Stochastic Oscillator
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            k_window = 14
            lowest_low = tech_data['Low'].rolling(window=k_window).min()
            highest_high = tech_data['High'].rolling(window=k_window).max()
            tech_data['Stoch_K'] = 100 * ((tech_data['Close'] - lowest_low) / (highest_high - lowest_low))
            tech_data['Stoch_D'] = tech_data['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            tr1 = tech_data['High'] - tech_data['Low']
            tr2 = abs(tech_data['High'] - tech_data['Close'].shift())
            tr3 = abs(tech_data['Low'] - tech_data['Close'].shift())
            true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            tech_data['ATR'] = true_range.rolling(window=14).mean()
            tech_data['ATR_Percent'] = tech_data['ATR'] / tech_data['Close']
        
        return tech_data
    
    def add_volatility_features(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add volatility-based features
        
        Args:
            data: Stock data
            windows: List of windows for volatility calculation
            
        Returns:
            DataFrame with volatility features
        """
        vol_data = data.copy()
        
        if 'Returns' in vol_data.columns:
            for window in windows:
                if window <= len(data):
                    # Historical volatility
                    vol_data[f'Volatility_{window}'] = vol_data['Returns'].rolling(window=window).std()
                    
                    # Realized volatility (annualized)
                    vol_data[f'Ann_Volatility_{window}'] = vol_data[f'Volatility_{window}'] * np.sqrt(252)
        
        # GARCH-like volatility (simplified)
        if 'Returns' in vol_data.columns:
            vol_data['Volatility_EWMA'] = vol_data['Returns'].ewm(alpha=0.06).var()
        
        return vol_data
    
    def create_lagged_features(self, data: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series
        
        Args:
            data: Stock data
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        lag_data = data.copy()
        
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    lag_data[f'{col}_lag_{lag}'] = lag_data[col].shift(lag)
        
        return lag_data
    
    def prepare_features(self, data: pd.DataFrame, 
                        target_column: str = 'Close',
                        feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete feature preparation pipeline
        
        Args:
            data: Raw stock data
            target_column: Column to predict
            feature_columns: Specific columns to use as features
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Clean data
        processed_data = self.clean_data(data)
        
        # Add all features
        processed_data = self.add_basic_features(processed_data)
        processed_data = self.add_moving_averages(processed_data)
        processed_data = self.add_technical_indicators(processed_data)
        processed_data = self.add_volatility_features(processed_data)
        
        # Select features
        if feature_columns is None:
            # Default feature selection
            feature_columns = [
                'Close', 'Volume', 'Returns', 'Price_Range',
                'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'BB_Width',
                'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26',
                'Volatility_20', 'ATR_Percent'
            ]
            
            # Keep only existing columns
            feature_columns = [col for col in feature_columns if col in processed_data.columns]
        
        # Extract features and target
        features = processed_data[feature_columns].copy()
        target = processed_data[target_column].copy()
        
        # Remove rows with any missing values
        mask = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[mask]
        target = target[mask]
        
        return features, target
    
    def scale_features(self, features: pd.DataFrame, target: pd.Series, 
                      fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features and target for ML models
        
        Args:
            features: Feature DataFrame
            target: Target Series
            fit_scalers: Whether to fit scalers (True for training, False for prediction)
            
        Returns:
            Tuple of (scaled_features, scaled_target)
        """
        if fit_scalers:
            # Fit and transform
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
        else:
            # Only transform (use fitted scalers)
            if self.feature_scaler is None or self.target_scaler is None:
                raise ValueError("Scalers not fitted. Set fit_scalers=True first.")
            scaled_features = self.feature_scaler.transform(features)
            scaled_target = self.target_scaler.transform(target.values.reshape(-1, 1)).flatten()
        
        return scaled_features, scaled_target
    
    def create_sequences(self, features: np.ndarray, target: np.ndarray, 
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            features: Scaled feature array
            target: Scaled target array
            sequence_length: Number of time steps in each sequence
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            # Use last 'sequence_length' days to predict next day
            X.append(features[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train_test_split_sequential(self, X: np.ndarray, y: np.ndarray, 
                                  train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split time series data maintaining temporal order
        
        Args:
            X: Feature sequences
            y: Target sequences
            train_size: Proportion of data for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_index = int(len(X) * train_size)
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale
        
        Args:
            scaled_target: Scaled target values
            
        Returns:
            Original scale values
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted")
        
        return self.target_scaler.inverse_transform(scaled_target.reshape(-1, 1)).flatten()
    
    def get_feature_importance(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple feature importance based on correlation with target
        
        Args:
            features: Feature DataFrame with target column
            
        Returns:
            DataFrame with feature importance scores
        """
        if 'Close' not in features.columns:
            raise ValueError("Target column 'Close' not found in features")
        
        correlations = features.corr()['Close'].abs().sort_values(ascending=False)
        
        importance_df = pd.DataFrame({
            'Feature': correlations.index,
            'Importance': correlations.values,
            'Type': ['Target' if x == 'Close' else 'Feature' for x in correlations.index]
        })
        
        return importance_df[importance_df['Type'] == 'Feature'].reset_index(drop=True)

# Utility functions for quick processing
def quick_process(data: pd.DataFrame, target_col: str = 'Close', sequence_length: int = 60) -> Dict:
    """
    Quick processing pipeline for stock data
    
    Args:
        data: Raw stock data
        target_col: Target column name
        sequence_length: Sequence length for LSTM
        
    Returns:
        Dictionary with processed data components
    """
    processor = StockDataProcessor()
    
    # Process features
    features, target = processor.prepare_features(data, target_col)
    
    # Scale data
    X_scaled, y_scaled = processor.scale_features(features, target, fit_scalers=True)
    
    # Create sequences
    X_sequences, y_sequences = processor.create_sequences(X_scaled, y_scaled, sequence_length)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.train_test_split_sequential(X_sequences, y_sequences)
    
    return {
        'processor': processor,
        'features': features,
        'target': target,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': features.columns.tolist()
    }

# Testing and example usage
if __name__ == "__main__":
    print("Testing Stock Data Processor...")
    print("=" * 45)
    
    # Create sample stock data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Generate realistic stock price data
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, 500)  # Small daily returns with volatility
    prices = [base_price]
    
    for i in range(1, 500):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(1, new_price))  # Ensure positive prices
    
    # Create OHLCV data
    sample_data = pd.DataFrame(index=dates)
    sample_data['Close'] = prices
    sample_data['Open'] = sample_data['Close'].shift(1) * np.random.uniform(0.995, 1.005, 500)
    sample_data['High'] = np.maximum(sample_data['Open'], sample_data['Close']) * np.random.uniform(1.0, 1.03, 500)
    sample_data['Low'] = np.minimum(sample_data['Open'], sample_data['Close']) * np.random.uniform(0.97, 1.0, 500)
    sample_data['Volume'] = np.random.randint(1000000, 10000000, 500)
    
    # Fill first row
    sample_data.iloc[0, sample_data.columns.get_loc('Open')] = sample_data.iloc[0]['Close']
    
    try:
        print("🧪 Testing data processor...")
        
        # Test processing pipeline
        result = quick_process(sample_data, sequence_length=30)
        
        print(f"✅ Raw data shape: {sample_data.shape}")
        print(f"✅ Features shape: {result['features'].shape}")
        print(f"✅ Training sequences: {result['X_train'].shape}")
        print(f"✅ Test sequences: {result['X_test'].shape}")
        print(f"✅ Feature columns: {len(result['feature_columns'])}")
        
        # Test individual components
        processor = StockDataProcessor()
        
        # Test cleaning
        clean_data = processor.clean_data(sample_data)
        print(f"✅ Data cleaning: {clean_data.shape}")
        
        # Test feature engineering
        with_features = processor.add_technical_indicators(clean_data)
        print(f"✅ Technical indicators: {with_features.shape[1]} columns")
        
        # Test feature importance
        features, target = processor.prepare_features(sample_data)
        importance = processor.get_feature_importance(pd.concat([features, target], axis=1))
        print(f"✅ Feature importance calculated: {len(importance)} features")
        
        print(f"\n🎯 Top 5 Most Important Features:")
        for i, row in importance.head().iterrows():
            print(f"   {i+1}. {row['Feature']}: {row['Importance']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Data processor module ready for ML pipeline integration!")