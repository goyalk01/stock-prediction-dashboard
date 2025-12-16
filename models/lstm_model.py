"""
LSTM Model for Stock Price Prediction
Advanced deep learning model with enhanced features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """Advanced LSTM model for stock price prediction"""
    
    def __init__(self, sequence_length=60, features=['Close', 'Volume', 'RSI', 'MACD']):
        self.sequence_length = sequence_length
        self.features = features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close'):
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with stock data
            target_column: Column to predict
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        # Select features
        feature_data = data[self.features].dropna()
        
        # Prepare target variable
        target_data = data[target_column].values.reshape(-1, 1)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        scaled_target = self.price_scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_target[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self, input_shape):
        """Build advanced LSTM architecture"""
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(units=100, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(units=100, return_sequences=True),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(units=50, return_sequences=False),
            Dropout(0.3),
            
            # Dense layers for final prediction
            Dense(units=25, activation='relu'),
            Dense(units=1, activation='linear')
        ])
        
        # Compile model with advanced optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100):
        """Train the LSTM model with callbacks"""
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X_test):
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X_test, verbose=0)
        
        # Inverse transform to get actual prices
        predictions = self.price_scaler.inverse_transform(predictions.reshape(-1, 1))
        
        return predictions.flatten()
    
    def predict_next_days(self, data: pd.DataFrame, days: int = 30):
        """Predict stock prices for next N days"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare last sequence
        feature_data = data[self.features].dropna()
        scaled_features = self.scaler.transform(feature_data)
        
        # Get last sequence
        last_sequence = scaled_features[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next price
            next_pred = self.model.predict(
                current_sequence.reshape(1, self.sequence_length, len(self.features)),
                verbose=0
            )[0, 0]
            
            predictions.append(next_pred)
            
            # Update sequence for next prediction
            # Note: This is simplified - in practice, you'd need to update other features too
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred  # Assuming Close is first feature
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.price_scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        # Convert to actual prices if scaled
        if hasattr(self.price_scaler, 'inverse_transform'):
            y_true = self.price_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load pre-trained model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)

class EnsemblePredictor:
    """Ensemble model combining multiple predictors"""
    
    def __init__(self):
        self.models = []
        self.weights = []
    
    def add_model(self, model: StockPredictor, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(self, X_test):
        """Make ensemble prediction"""
        if not self.models:
            raise ValueError("No models in ensemble!")
        
        predictions = []
        total_weight = sum(self.weights)
        
        for i, model in enumerate(self.models):
            pred = model.predict(X_test)
            weighted_pred = pred * (self.weights[i] / total_weight)
            predictions.append(weighted_pred)
        
        # Combine predictions
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred

class ModelOptimizer:
    """Hyperparameter optimization for LSTM models"""
    
    @staticmethod
    def grid_search_params():
        """Define parameter grid for optimization"""
        param_grid = {
            'sequence_length': [30, 60, 90],
            'lstm_units': [50, 100, 150],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01, 0.1]
        }
        return param_grid
    
    @staticmethod
    def optimize_model(data: pd.DataFrame, param_grid: dict, cv_folds: int = 3):
        """Perform grid search optimization"""
        best_score = float('inf')
        best_params = None
        
        # This is a simplified version - in practice, you'd use proper cross-validation
        for seq_len in param_grid['sequence_length']:
            for lstm_units in param_grid['lstm_units']:
                for dropout in param_grid['dropout_rate']:
                    for lr in param_grid['learning_rate']:
                        
                        try:
                            # Create and train model with these parameters
                            predictor = StockPredictor(sequence_length=seq_len)
                            
                            # Prepare data
                            X_train, y_train, X_test, y_test = predictor.prepare_data(data)
                            
                            # Build custom model with these parameters
                            model = Sequential([
                                LSTM(units=lstm_units, return_sequences=True, 
                                     input_shape=(seq_len, len(predictor.features))),
                                Dropout(dropout),
                                LSTM(units=lstm_units//2),
                                Dropout(dropout),
                                Dense(units=1)
                            ])
                            
                            model.compile(
                                optimizer=Adam(learning_rate=lr),
                                loss='mse',
                                metrics=['mae']
                            )
                            
                            predictor.model = model
                            
                            # Train with fewer epochs for grid search
                            history = model.fit(
                                X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=20,
                                batch_size=32,
                                verbose=0
                            )
                            
                            # Evaluate
                            val_loss = min(history.history['val_loss'])
                            
                            if val_loss < best_score:
                                best_score = val_loss
                                best_params = {
                                    'sequence_length': seq_len,
                                    'lstm_units': lstm_units,
                                    'dropout_rate': dropout,
                                    'learning_rate': lr
                                }
                        
                        except Exception as e:
                            print(f"Error with params {seq_len}, {lstm_units}, {dropout}, {lr}: {e}")
                            continue
        
        return best_params, best_score

# Example usage and testing
if __name__ == "__main__":
    print("Testing LSTM Stock Prediction Model...")
    print("=" * 45)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate realistic stock data
    price = 100
    prices = [price]
    volumes = []
    
    for _ in range(999):
        change = np.random.normal(0, 0.02)
        price = max(1, price * (1 + change))
        prices.append(price)
        volumes.append(np.random.randint(1000000, 5000000))
    
    volumes.append(np.random.randint(1000000, 5000000))
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'Close': prices,
        'Volume': volumes,
        'RSI': np.random.uniform(20, 80, 1000),
        'MACD': np.random.normal(0, 1, 1000)
    }, index=dates)
    
    try:
        # Test model creation and training
        predictor = StockPredictor()
        
        # Prepare data
        X_train, y_train, X_test, y_test = predictor.prepare_data(sample_data)
        print(f"✅ Data prepared: {X_train.shape}, {X_test.shape}")
        
        # Train model (with fewer epochs for testing)
        history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=5)
        print("✅ Model trained successfully")
        
        # Make predictions
        predictions = predictor.predict(X_test)
        print(f"✅ Predictions made: {len(predictions)} values")
        
        # Evaluate model
        y_test_actual = predictor.price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        metrics = predictor.evaluate_model(y_test_actual, predictions)
        print(f"✅ Model MAPE: {metrics['MAPE']:.2f}%")
        
        # Test future predictions
        future_prices = predictor.predict_next_days(sample_data, days=5)
        print(f"✅ Future predictions: {len(future_prices)} days")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✨ LSTM prediction model ready!")