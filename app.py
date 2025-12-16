"""
Stock Market Prediction & Analysis Dashboard
Main Streamlit Application - Production Ready & Fully Integrated (Final)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import warnings
import traceback

# --- Import Backend Modules ---
try:
    from utils.stock_data import StockDataFetcher, MarketAnalyzer
    from models.data_processor import StockDataProcessor
    from models.lstm_model import StockPredictor
    from utils.visualization import StockVisualizer
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all module files are in the correct directories (utils/ and models/)")
    st.stop()

warnings.filterwarnings("ignore")


# --- Global Config ---
CONFIG = {
    'MODEL_EPOCHS': 20,
    'SEQUENCE_LENGTH': 60,
    'CACHE_TTL': 600
}

# --- Page Configuration ---
st.set_page_config(
    page_title="📈 Stock Market Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
session_vars = [
    "analysis_complete",
    "raw_data",
    "processed_data",
    "stock_info",
    "selected_symbol",
    "predictor",
    "predictions",
    "model_metrics",
]

for var in session_vars:
    st.session_state.setdefault(var, False if var == "analysis_complete" else None)

# --- Custom CSS ---
st.markdown(
    """
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .success-box { background-color: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; border: 1px solid #c3e6cb; }
    .warning-box { background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 5px; border: 1px solid #ffeaa7; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Helper Functions ---


def _ensure_series(target):
    """
    Ensure target is a pandas Series. If DataFrame with multiple columns,
    pick a sensible single column (prefers 'Target' then last column).
    """
    if target is None:
        return None
    if isinstance(target, pd.DataFrame):
        if target.shape[1] == 1:
            target = target.iloc[:, 0]
        else:
            # Prefer named column 'Target' or 'Close' if present, otherwise take last column
            for prefer in ["Target", "target", "Close", "close", "Next_Close", "next_close"]:
                if prefer in target.columns:
                    target = target[prefer]
                    break
            else:
                target = target.iloc[:, -1]
    if isinstance(target, pd.Series):
        target = target.rename("Target")
    else:
        # If scalar or list-like, try to coerce to Series
        try:
            target = pd.Series(target, name="Target")
        except Exception:
            raise ValueError("Could not coerce target into pd.Series")
    return target


def _coerce_numeric_columns(df: pd.DataFrame, cols: list[str]):
    """Coerce columns in list to numeric (pick first column if a DataFrame slice)"""
    for c in cols:
        if c in df.columns:
            col = df[c]
            if isinstance(col, pd.DataFrame):
                # If accidentally a multi-column slice, pick first
                df[c] = col.iloc[:, 0]
            # coerce to numeric
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_float_extract(series_or_value, default=0.0):
    """Safely extract float value from pandas Series or scalar, using last non-null."""
    try:
        if series_or_value is None:
            return default
        # If DataFrame, pick first column
        if isinstance(series_or_value, pd.DataFrame):
            series_or_value = series_or_value.iloc[:, 0]
        # Series path
        if isinstance(series_or_value, pd.Series):
            s = series_or_value.dropna()
            if len(s) == 0:
                return default
            return float(s.iloc[-1])
        # numpy array or list
        if hasattr(series_or_value, "__len__") and not isinstance(series_or_value, (str, bytes)):
            arr = np.asarray(series_or_value)
            if arr.size == 0:
                return default
            return float(arr.ravel()[-1])
        return float(series_or_value)
    except Exception:
        return default


def safe_column_access(df: pd.DataFrame, column_name: str, default=0.0):
    """Safely access DataFrame column with fallback to default."""
    try:
        if column_name not in df.columns:
            return default
        col = df[column_name]
        return safe_float_extract(col, default=default)
    except Exception:
        return default


# --- Cached Functions ---
@st.cache_data(ttl=CONFIG['CACHE_TTL'], show_spinner=False)
def fetch_and_process_data(symbol: str, period: str):

    """
    Fetch raw data and process it with full feature engineering pipeline.
    Ensures returned processed_data has numeric single-column 'Close' and 'Target' series.
    """
    try:
        fetcher = StockDataFetcher()
        processor = StockDataProcessor()

        # Fetch raw data
        raw_data = fetcher.fetch_stock_data(symbol, period)
        if raw_data is None or raw_data.empty:
            return None, None, "No data available"

        # Ensure index is datetime
        try:
            raw_data.index = pd.to_datetime(raw_data.index)
        except Exception:
            pass

        # Process features - the processor should create features and a target (series)
        features, target = processor.prepare_features(raw_data)

        # Normalize target into Series
        target = _ensure_series(target)

        # Combine features and target properly
        processed_data = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        processed_data[target.name if hasattr(target, "name") else "Target"] = target

        # Ensure we have the original Close column for visualization (prefer processed close, then raw)
        if "Close" not in processed_data.columns and "Close" in raw_data.columns:
            processed_data["Close"] = raw_data["Close"]

        # If Close is multi-column accidentally, pick first and coerce numeric
        processed_data = _coerce_numeric_columns(
            processed_data,
            [
                "Close",
                "Returns",
                "RSI",
                "SMA_20",
                "SMA_50",
                "Upper_Band",
                "Lower_Band",
                "MACD",
                "MACD_Signal",
            ],
        )

        # Defensive: drop rows where Close or Target is null
        if "Close" in processed_data.columns and "Target" in processed_data.columns:
            processed_data = processed_data.dropna(subset=["Close", "Target"])

        processed_data.index = pd.to_datetime(processed_data.index, errors="coerce")
        processed_data = processed_data.dropna(how="all")  # drop rows which are fully NaN

        if processed_data.empty:
            return None, None, "Data processing failed - no valid data after cleaning"

        return raw_data, processed_data, "Success"

    except Exception as e:
        return None, None, f"Error: {str(e)}"


@st.cache_resource(show_spinner=False)
def load_or_train_model(symbol: str, processed_data: pd.DataFrame):
    try:
        processor = StockDataProcessor()
        predictor = StockPredictor(sequence_length=CONFIG['SEQUENCE_LENGTH'])

        # --- Prepare features and target robustly ---
        if "Target" in processed_data.columns and not processed_data["Target"].isnull().all():
            target = processed_data["Target"].dropna()
            features = processed_data.drop(columns=["Target"])
        elif "Close" in processed_data.columns and not processed_data["Close"].isnull().all():
            # Use Close to generate target if Target missing
            target = processed_data["Close"].shift(-1).dropna()
            # Features: all other numeric columns except Close
            features = processed_data.drop(columns=["Close"]).loc[target.index]
        else:
            # Fallback if neither Target nor Close available
            return None, "No valid Target or Close column available for training", None

        # Align indices
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx].copy()
        target = target.loc[common_idx].copy()

        # Coerce numeric
        features = features.apply(pd.to_numeric, errors="coerce")
        target = pd.to_numeric(target, errors="coerce").dropna()
        features = features.loc[target.index]  # align again

        if len(features) < 100:
            return None, "Insufficient data for model training (need >100 samples)", None

        # --- Scale features and target ---
        X_scaled, y_scaled = processor.scale_features(features, target, fit_scalers=True)

        # --- Create sequences AFTER scaling ---
        X_seq, y_seq = processor.create_sequences(X_scaled, y_scaled, sequence_length=CONFIG['SEQUENCE_LENGTH'])

        if len(X_seq) < 50:
            missing = 50 - len(X_seq)
            return None, f"❌ Not enough data. Need {missing} more sequence samples for training", None

        # Train-test split
        X_train, X_test, y_train, y_test = processor.train_test_split_sequential(X_seq, y_seq, train_size=0.8)

        # Train model
        history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=CONFIG['MODEL_EPOCHS'])

        # Evaluate model (with inverse transform if available)
        try:
            y_test_actual = processor.inverse_transform_target(y_test)
            preds_scaled = predictor.predict(X_test)
            preds_actual = processor.inverse_transform_target(preds_scaled)
        except Exception:
            preds_actual = predictor.predict(X_test)
            y_test_actual = y_test

        metrics = predictor.evaluate_model(y_test_actual, preds_actual)

        # Save model
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/{symbol}_lstm_model.keras"
        try:
            predictor.save_model(model_path)
        except Exception as save_error:
            st.warning(f"Could not save model: {save_error}")

        return predictor, "Model trained successfully", metrics

    except Exception as e:
        return None, f"Model error: {str(e)}", None


def generate_investment_recommendation(data: pd.DataFrame, predictions: np.ndarray = None) -> dict:
    """Generate investment recommendation based on multiple factors"""
    try:
        score = 0.0
        factors = []

        # Safely extract values
        current_rsi = safe_column_access(data, "RSI", 50.0)
        sma_20 = safe_column_access(data, "SMA_20", 0.0)
        sma_50 = safe_column_access(data, "SMA_50", 0.0)
        current_price = safe_column_access(data, "Close", 100.0)

        if sma_20 == 0.0:
            sma_20 = current_price
        if sma_50 == 0.0:
            sma_50 = current_price

        # RSI analysis
        if current_rsi < 30:
            score += 2
            factors.append("🟢 Oversold condition - potential buying opportunity")
        elif current_rsi > 70:
            score -= 2
            factors.append("🔴 Overbought condition - potential selling pressure")
        else:
            score += 0.5
            factors.append("🟡 Neutral RSI levels")

        # Moving average trend
        if sma_20 > sma_50:
            score += 2
            factors.append("🟢 Bullish trend - short MA above long MA")
        else:
            score -= 1
            factors.append("🔴 Bearish trend - short MA below long MA")

        # Price vs moving averages
        if current_price > sma_20:
            score += 1
            factors.append("🟢 Price above 20-day moving average")
        else:
            score -= 0.5
            factors.append("🔴 Price below 20-day moving average")

        # Volatility analysis
        if "Returns" in data.columns:
            returns = data["Returns"].dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252) * 100
                if volatility < 20:
                    score += 1
                    factors.append("🟢 Low volatility environment")
                elif volatility > 40:
                    score -= 1
                    factors.append("🔴 High volatility - increased risk")

        # Prediction analysis
        if predictions is not None and len(predictions) > 0:
            try:
                last_prediction = float(predictions[-1])
                future_return = (last_prediction / current_price - 1.0) * 100.0
                if future_return > 5:
                    score += 1.5
                    factors.append(f"🟢 AI model predicts {future_return:.1f}% upside")
                elif future_return < -5:
                    score -= 1.5
                    factors.append(f"🔴 AI model predicts {future_return:.1f}% downside")
            except Exception:
                pass

        # Final recommendation
        if score >= 3:
            recommendation = "🟢 STRONG BUY"
        elif score >= 1:
            recommendation = "🟡 BUY"
        elif score >= -1:
            recommendation = "🟠 HOLD"
        else:
            recommendation = "🔴 SELL"

        return {"recommendation": recommendation, "score": score, "max_score": 6, "factors": factors}

    except Exception as e:
        return {
            "recommendation": "🔄 ANALYSIS INCOMPLETE",
            "score": 0,
            "max_score": 6,
            "factors": [f"Error in analysis: {str(e)}"],
        }


# --- Main Application ---
def main():
    """Main dashboard application"""

    # Header
    st.markdown('<div class="main-header">📈 Stock Market Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        """
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Advanced Stock Analysis & Prediction Platform with Real-time Data Integration
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    st.sidebar.title("🔧 Dashboard Controls")

    # Stock selection
    popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

    selected_symbol = st.sidebar.selectbox("📊 Select Stock Symbol", options=popular_stocks, index=0)

    # Time period selection
    time_periods = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}

    selected_period = st.sidebar.selectbox("⏰ Select Time Period", options=list(time_periods.keys()), index=2)

    # Analysis options
    st.sidebar.markdown("### 🔍 Analysis Options")
    show_technical = st.sidebar.checkbox("Technical Analysis", value=True)
    show_predictions = st.sidebar.checkbox("AI Price Predictions", value=True)
    show_risk = st.sidebar.checkbox("Risk Analysis", value=True)
    show_recommendation = st.sidebar.checkbox("Investment Recommendation", value=True)

    # Analysis button
    if st.sidebar.button("🚀 Analyze Stock", type="primary"):
        st.session_state.analysis_complete = False

        with st.container():
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Data fetching
                status_text.text("🔄 Fetching stock data...")
                progress_bar.progress(15)

                raw_data, processed_data, fetch_status = fetch_and_process_data(
                    selected_symbol, time_periods[selected_period]
                )

                if raw_data is None:
                    st.error(f"❌ {fetch_status}")
                    return

                progress_bar.progress(40)
                status_text.text("📊 Getting company information...")

                # Step 2: Stock info
                try:
                    fetcher = StockDataFetcher()
                    stock_info = fetcher.get_stock_info(selected_symbol)
                except Exception as e:
                    stock_info = {"error": str(e)}
                    st.warning(f"Could not fetch company info: {e}")

                # Step 3: Model handling
                progress_bar.progress(65)
                status_text.text("🤖 Loading AI model...")

                predictor, model_status, model_metrics = None, "Skipped", None
                predictions = None

                if show_predictions:
                    predictor, model_status, model_metrics = load_or_train_model(selected_symbol, processed_data)

                    if predictor:
                        progress_bar.progress(85)
                        status_text.text("🔮 Generating predictions...")
                        try:
                            predictions = predictor.predict_next_days(processed_data, days=30)
                        except Exception as e:
                            st.warning(f"Prediction generation failed: {e}")
                            predictions = None

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")

                # Store results
                st.session_state.raw_data = raw_data
                st.session_state.processed_data = processed_data
                st.session_state.stock_info = stock_info
                st.session_state.selected_symbol = selected_symbol
                st.session_state.predictor = predictor
                st.session_state.predictions = predictions
                st.session_state.model_metrics = model_metrics
                st.session_state.analysis_complete = True

                # Show model status
                if show_predictions:
                    if isinstance(model_status, str) and "successfully" in model_status.lower():
                        st.success(f"✅ {model_status}")
                    else:
                        st.warning(f"⚠️ {model_status}")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                if st.checkbox("Show debug info"):
                    st.code(traceback.format_exc())
                return

    # Display Results
    if st.session_state.get("analysis_complete", False) and st.session_state.processed_data is not None:

        raw_data = st.session_state.raw_data
        processed_data = st.session_state.processed_data
        stock_info = st.session_state.stock_info or {}
        symbol = st.session_state.selected_symbol
        predictions = st.session_state.predictions
        model_metrics = st.session_state.model_metrics

        # Initialize utility classes
        visualizer = StockVisualizer()
        analyzer = MarketAnalyzer()

        # Stock Overview
        st.markdown(f"## 📋 {symbol} Stock Overview")

        col1, col2, col3, col4 = st.columns(4)

        # Safe value extraction
        current_price = safe_column_access(processed_data, "Close", 0.0)
        daily_change = safe_column_access(processed_data, "Returns", 0.0) * 100.0

        with col1:
            st.metric("💰 Current Price", f"${current_price:.2f}", f"{daily_change:+.2f}%")

        with col2:
            market_cap = stock_info.get("market_cap", 0) or 0
            st.metric("📊 Market Cap", f"${market_cap/1e9:.1f}B" if market_cap else "N/A")

        with col3:
            pe_ratio = stock_info.get("pe_ratio", 0)
            st.metric("📈 P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio else "N/A")

        with col4:
            sector = stock_info.get("sector", "Unknown")
            st.metric("🏢 Sector", sector)

        # Technical Analysis
        if show_technical:
            st.markdown("## 📊 Technical Analysis")
            try:
                required_ohlc = ['Open', 'High', 'Low', 'Close']

                # Ensure OHLC columns exist in processed_data
                for col in required_ohlc:
                    if col not in processed_data.columns and raw_data is not None and col in raw_data.columns:
                        # Add missing columns from raw_data
                        processed_data[col] = raw_data[col]

                # Check again if OHLC exists
                has_ohlc = all(col in processed_data.columns for col in required_ohlc)

                indicators = {}
                for col in ["SMA_20", "SMA_50", "RSI", "Upper_Band", "Lower_Band", "MACD", "MACD_Signal"]:
                    if col in processed_data.columns:
                        indicators[col] = processed_data[col]

                if has_ohlc:
                    # Create candlestick chart
                    main_chart = visualizer.create_candlestick_chart(
                        processed_data,
                        title=f"{symbol} Technical Analysis",
                        indicators=indicators
                    )
                else:
                    # Fallback: simple line chart if OHLC missing
                    import plotly.graph_objects as go
                    main_chart = go.Figure()
                    main_chart.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=processed_data['Close'],
                        mode='lines',
                        name='Close Price'
                    ))

                st.plotly_chart(main_chart, use_container_width=True)

            except Exception as e:
                st.error(f"Chart error: {e}")


        # AI Predictions
        if show_predictions and predictions is not None:
            st.markdown("## 🔮 AI Price Predictions")
            try:
                last_date = processed_data.index[-1]
                prediction_dates = pd.bdate_range(last_date + timedelta(days=1), periods=30)

                pred_chart = visualizer.create_prediction_chart(processed_data.tail(60), predictions, prediction_dates, title=f"{symbol} 30-Day Forecast")
                st.plotly_chart(pred_chart, use_container_width=True)

                # Prediction metrics
                col1, col2, col3 = st.columns(3)

                predicted_return = (float(predictions[-1]) / max(current_price, 1e-8) - 1) * 100

                with col1:
                    return_signal = "🟢 Bullish" if predicted_return > 0 else "🔴 Bearish"
                    st.metric("30-Day Return", f"{predicted_return:+.1f}%", return_signal)

                with col2:
                    max_pred = float(np.max(predictions))
                    upside = (max_pred / max(current_price, 1e-8) - 1) * 100
                    st.metric("Max Upside", f"+{upside:.1f}%")

                with col3:
                    min_pred = float(np.min(predictions))
                    downside = (min_pred / max(current_price, 1e-8) - 1) * 100
                    st.metric("Max Downside", f"{downside:.1f}%")

            except Exception as e:
                st.error(f"Prediction display error: {e}")

        # Risk Analysis
        if show_risk:
            st.markdown("## ⚠️ Risk Analysis")
            try:
                returns = processed_data.get("Returns", pd.Series(dtype=float))
                if isinstance(returns, pd.Series) and not returns.dropna().empty:
                    returns = returns.dropna()
                    sharpe_ratio = analyzer.calculate_sharpe_ratio(returns)
                    max_drawdown = abs(analyzer.calculate_max_drawdown(processed_data["Close"])) * 100
                    var_95 = abs(analyzer.calculate_var(returns)) * 100

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    with col2:
                        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
                    with col3:
                        st.metric("VaR (95%)", f"{var_95:.1f}%")
                else:
                    st.info("Not enough returns data for risk metrics.")
            except Exception as e:
                st.error(f"Risk analysis error: {e}")

        # Investment Recommendation
        if show_recommendation:
            st.markdown("## 💡 Investment Recommendation")
            try:
                rec_data = generate_investment_recommendation(processed_data, predictions)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"### {rec_data['recommendation']}")
                    st.markdown(f"**Score: {rec_data['score']:.1f}/{rec_data['max_score']}**")
                with col2:
                    st.markdown("**Key Factors:**")
                    for factor in rec_data["factors"]:
                        st.markdown(f"• {factor}")
            except Exception as e:
                st.error(f"Recommendation error: {e}")

        # Disclaimer
        st.markdown("---")
        st.markdown(
            """
        <div class="warning-box">
        ⚠️ <strong>Educational Use Only:</strong> This dashboard is for learning purposes. 
        Not financial advice. Always consult professionals before investing.
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        if st.checkbox("Show debug info"):
            st.code(traceback.format_exc())
