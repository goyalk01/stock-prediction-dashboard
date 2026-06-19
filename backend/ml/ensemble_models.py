"""Fast baseline ML ensemble models for scalable stock forecasting.

This module adds two lightweight scikit-learn models and combines them. It is
intended for production paths where training an LSTM inside a web request is too
slow or too expensive for free hosting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "Close",
    "Volume",
    "Returns",
    "RSI",
    "SMA_20",
    "SMA_50",
    "MACD",
    "MACD_Signal",
]


@dataclass
class EnsembleForecastResult:
    predictions: np.ndarray
    metrics: dict[str, float]
    model_weights: dict[str, float]


class TabularEnsembleForecaster:
    """Combine Random Forest and Gradient Boosting for quick price forecasts."""

    def __init__(self, features: Iterable[str] | None = None, random_state: int = 42):
        self.features = list(features or DEFAULT_FEATURES)
        self.random_state = random_state
        self.models = {
            "random_forest": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(n_estimators=120, max_depth=8, random_state=random_state, n_jobs=-1)),
                ]
            ),
            "gradient_boosting": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingRegressor(n_estimators=160, learning_rate=0.04, max_depth=3, random_state=random_state)),
                ]
            ),
        }
        self.weights = {"random_forest": 0.55, "gradient_boosting": 0.45}
        self.last_feature_row: pd.DataFrame | None = None

    def _frame(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
available_features = [col for col in self.features if col in data.columns]
if "Close" not in data.columns:
    raise ValueError("Close column is required for forecasting")
if not available_features:
    raise ValueError("No usable model features found")
if "Close" not in available_features:
    available_features = ["Close", *available_features]
frame = data[available_features].copy()
frame["Target"] = frame["Close"].shift(-1)
frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["Target"])
X = frame[available_features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(frame["Target"], errors="coerce")
        valid = y.notna()
        return X.loc[valid], y.loc[valid]

    def fit(self, data: pd.DataFrame) -> dict[str, float]:
        X, y = self._frame(data)
        if len(X) < 80:
            raise ValueError("At least 80 clean rows are required for ensemble training")
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        metrics: dict[str, float] = {}
        predictions = []
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred * self.weights[name])
            metrics[f"{name}_MAE"] = float(mean_absolute_error(y_test, pred))
        ensemble_pred = np.sum(predictions, axis=0)
        metrics.update(
            {
                "MAE": float(mean_absolute_error(y_test, ensemble_pred)),
                "MSE": float(mean_squared_error(y_test, ensemble_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
                "R2": float(r2_score(y_test, ensemble_pred)),
                "MAPE": float(np.mean(np.abs((y_test.to_numpy() - ensemble_pred) / y_test.to_numpy())) * 100),
            }
        )
        self.last_feature_row = X.tail(1).copy()
        for model in self.models.values():
            model.fit(X, y)
        return metrics

    def predict_next_days(self, data: pd.DataFrame, days: int = 30) -> np.ndarray:
        if self.last_feature_row is None:
            self.fit(data)
        current = self.last_feature_row.copy()
        predictions: list[float] = []
        for _ in range(days):
            model_predictions = [self.models[name].predict(current)[0] * self.weights[name] for name in self.models]
            next_price = float(np.sum(model_predictions))
            predictions.append(next_price)
            if "Close" in current.columns:
                previous_close = float(current["Close"].iloc[0])
                current.loc[:, "Returns"] = (next_price / previous_close) - 1 if previous_close else 0
                current.loc[:, "Close"] = next_price
        return np.array(predictions)

    def fit_predict(self, data: pd.DataFrame, days: int = 30) -> EnsembleForecastResult:
        metrics = self.fit(data)
        predictions = self.predict_next_days(data, days=days)
        return EnsembleForecastResult(predictions=predictions, metrics=metrics, model_weights=self.weights.copy())
