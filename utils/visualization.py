"""
Advanced Visualization Module for Stock Market Dashboard
Comprehensive plotting utilities using Plotly and Matplotlib
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockVisualizer:
    """Advanced stock market visualization toolkit"""
    
    def __init__(self, theme: str = 'plotly_dark'):
        self.theme = theme
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444', 
            'neutral': '#ffaa00',
            'volume': '#4a90e2',
            'ma_short': '#ff6b6b',
            'ma_long': '#4ecdc4',
            'rsi_over': '#ff4444',
            'rsi_under': '#00ff88'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, title: str = "Stock Price Chart",
                               indicators: Dict = None, height: int = 600) -> go.Figure:
        """
        Create interactive candlestick chart with technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            title: Chart title
            indicators: Dictionary of technical indicators to overlay
            height: Chart height
            
        Returns:
            Plotly figure object
        """
        # Create subplots for main chart and volume
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price', 'Volume', 'RSI')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="OHLC",
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Add technical indicators if provided
        if indicators:
            # Moving averages
            if 'SMA_20' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color=self.colors['ma_short'], width=2)
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color=self.colors['ma_long'], width=2)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if 'Upper_Band' in indicators and 'Lower_Band' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['Upper_Band'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='rgba(128,128,128,0.3)', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['Lower_Band'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='rgba(128,128,128,0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ),
                    row=1, col=1
                )
        
        # Volume chart
        colors = ['red' if close < open_ else 'green' 
                 for close, open_ in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI chart
        if indicators and 'RSI' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['neutral'], width=2)
                ),
                row=3, col=1
            )
            
            # RSI overbought/oversold levels
            fig.add_hline(y=70, line_dash="dash", line_color=self.colors['rsi_over'], 
                         row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=self.colors['rsi_under'], 
                         row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=height,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                  title: str = "Stock Correlation Matrix") -> go.Figure:
        """Create interactive correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdYlBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            width=600,
            height=500
        )
        
        return fig
    
    def create_portfolio_performance_chart(self, portfolio_data: pd.DataFrame,
                                         benchmark_data: pd.DataFrame = None,
                                         title: str = "Portfolio Performance") -> go.Figure:
        """Create portfolio performance comparison chart"""
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data['Cumulative_Returns'],
            mode='lines',
            name='Portfolio',
            line=dict(color=self.colors['bullish'], width=3)
        ))
        
        # Benchmark comparison
        if benchmark_data is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data['Cumulative_Returns'],
                mode='lines',
                name='Benchmark',
                line=dict(color=self.colors['neutral'], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Cumulative Returns (%)",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_risk_metrics_radar(self, metrics: Dict, title: str = "Risk Metrics") -> go.Figure:
        """Create radar chart for risk metrics"""
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Profile',
            line_color=self.colors['bullish']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )),
            title=title,
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_prediction_chart(self, historical_data: pd.DataFrame,
                              predictions: np.ndarray,
                              prediction_dates: pd.DatetimeIndex,
                              title: str = "Price Predictions") -> go.Figure:
        """Create chart showing historical data and predictions"""
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color=self.colors['neutral'], width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=prediction_dates,
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color=self.colors['bullish'], width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add vertical line to separate historical from predictions
        last_date = historical_data.index[-1]
        fig.add_vline(x=last_date, line_dash="solid", line_color="gray")
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_volume_profile(self, data: pd.DataFrame, 
                            title: str = "Volume Profile") -> go.Figure:
        """Create volume profile chart"""
        # Calculate volume at each price level
        price_bins = pd.cut(data['Close'], bins=50)
        volume_profile = data.groupby(price_bins)['Volume'].sum().sort_index()
        
        # Get midpoint of each bin for plotting
        price_levels = [interval.mid for interval in volume_profile.index]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=price_levels,
            x=volume_profile.values,
            orientation='h',
            name='Volume',
            marker_color=self.colors['volume'],
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Volume",
            yaxis_title="Price Level",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_sector_performance_chart(self, sector_data: Dict,
                                      title: str = "Sector Performance") -> go.Figure:
        """Create sector performance comparison"""
        sectors = list(sector_data.keys())
        returns = list(sector_data.values())
        
        colors = [self.colors['bullish'] if ret > 0 else self.colors['bearish'] 
                 for ret in returns]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sectors,
            y=returns,
            marker_color=colors,
            name='Returns',
            text=[f"{ret:.1f}%" for ret in returns],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Sector",
            yaxis_title="Returns (%)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_drawdown_chart(self, data: pd.DataFrame,
                            title: str = "Drawdown Analysis") -> go.Figure:
        """Create drawdown chart"""
        # Calculate drawdown
        peak = data['Close'].expanding().max()
        drawdown = (data['Close'] - peak) / peak * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=['Price with Peaks', 'Drawdown (%)']
        )
        
        # Price chart with peaks
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color=self.colors['neutral'], width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=peak,
                mode='lines',
                name='Peak',
                line=dict(color=self.colors['bullish'], width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        # Drawdown chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color=self.colors['bearish'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255,68,68,0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            showlegend=True
        )
        
        return fig

# Utility functions for quick plotting
def quick_price_chart(data: pd.DataFrame, symbol: str = "Stock") -> go.Figure:
    """Quick price chart creation"""
    visualizer = StockVisualizer()
    return visualizer.create_candlestick_chart(
        data, 
        title=f"{symbol} Price Chart"
    )

def quick_correlation_plot(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Quick correlation heatmap"""
    visualizer = StockVisualizer()
    return visualizer.create_correlation_heatmap(correlation_matrix)

# Testing and example usage
if __name__ == "__main__":
    print("Testing Stock Visualization Module...")
    print("=" * 42)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data
    base_price = 100
    sample_data = pd.DataFrame(index=dates)
    
    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    
    for i in range(1, 100):
        prices.append(prices[-1] * (1 + returns[i]))
    
    sample_data['Close'] = prices
    sample_data['Open'] = sample_data['Close'].shift(1).fillna(base_price)
    sample_data['High'] = sample_data[['Open', 'Close']].max(axis=1) * np.random.uniform(1.0, 1.05, 100)
    sample_data['Low'] = sample_data[['Open', 'Close']].min(axis=1) * np.random.uniform(0.95, 1.0, 100)
    sample_data['Volume'] = np.random.randint(1000000, 5000000, 100)
    
    # Add some indicators
    sample_indicators = {
        'SMA_20': sample_data['Close'].rolling(20).mean(),
        'RSI': pd.Series(np.random.uniform(20, 80, 100), index=dates)
    }
    
    # Test visualizer
    try:
        visualizer = StockVisualizer()
        
        # Test candlestick chart
        fig1 = visualizer.create_candlestick_chart(
            sample_data, 
            title="Test Stock Chart",
            indicators=sample_indicators
        )
        print("✅ Candlestick chart created")
        
        # Test correlation heatmap (create sample correlation matrix)
        correlation_data = pd.DataFrame({
            'AAPL': np.random.normal(0, 0.02, 100),
            'GOOGL': np.random.normal(0, 0.02, 100),
            'MSFT': np.random.normal(0, 0.02, 100)
        }).corr()
        
        fig2 = visualizer.create_correlation_heatmap(correlation_data)
        print("✅ Correlation heatmap created")
        
        # Test prediction chart
        future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        predictions = np.random.uniform(90, 110, 10)
        
        fig3 = visualizer.create_prediction_chart(
            sample_data,
            predictions,
            future_dates
        )
        print("✅ Prediction chart created")
        
        print("\n✨ All visualization components working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Visualization module ready for dashboard integration!")