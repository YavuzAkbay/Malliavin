# 🧠 ML-Enhanced Malliavin Calculus for Financial Analysis

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Python framework that combines Malliavin calculus with deep learning to perform advanced financial sensitivity analysis and risk assessment. This tool uses neural networks to approximate Malliavin derivatives, providing comprehensive investment insights through machine learning-enhanced mathematical finance.

## 🚀 Features

- 🧮 Advanced Malliavin Calculus: Implements mathematical framework for sensitivity analysis
- 🤖 Neural Network Integration: Deep learning models for derivative approximation
- 📊 Comprehensive Risk Metrics: VaR, CVaR, Delta, Gamma, and Vega calculations
- 📈 Multi-Scenario Analysis: Generates thousands of market scenarios for robust predictions
- 🎯 Investment Recommendations: AI-powered guidance for different investor profiles
- 📉 Real-time Data Integration: Fetches live market data via yfinance
- 🎨 Publication-Quality Visualizations: Professional charts and analysis reports
- ⚡ GPU Acceleration: CUDA support for faster neural network training

## 📋 Prerequisites

Before running the analysis, ensure you have Python 3.7+ installed with the following requirements:

PyTorch (with optional CUDA support)
NumPy, Pandas, Matplotlib
scikit-learn, SciPy
yfinance for market data

## 🔧 Installation

1. Clone the repository

```bash
git clone https://github.com/YavuzAkbay/Malliavin
cd Malliavin
```

2. Install required packages

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn scipy yfinance
```

3. For GPU acceleration (optional):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 💻 Usage

Basic Analysis

```python
from malliavin_analysis import EnhancedMalliavinAnalysis

# Initialize analyzer for Apple stock
analyzer = EnhancedMalliavinAnalysis("AAPL", period="2y")

# Run comprehensive analysis with visualizations
scenarios_df, risk_metrics = analyzer.visualize_enhanced_analysis()
```

Custom Configuration

```python
# Advanced configuration
analyzer = EnhancedMalliavinAnalysis(
    ticker="TSLA",
    period="1y",
    device='cuda'  # Use GPU if available
)

# Train neural network with custom parameters
analyzer.fetch_data()
analyzer.train_neural_network(epochs=2000, batch_size=1024, learning_rate=0.001)

# Generate sensitivity analysis
scenarios_df = analyzer.enhanced_sensitivity_analysis(n_scenarios=50000)
risk_metrics = analyzer.advanced_risk_metrics(scenarios_df)
```

## 📊 Analysis Components

Neural Network Architecture
- Input Layer: Market features (price ratios, volatility, time horizons)
- Hidden Layers: 3-layer architecture with ReLU activation and dropout
- Output Layer: Malliavin derivatives (Delta, Gamma, Vega)

Risk Metrics Calculated

| Metric  | Description |
| ------------- | ------------- |
| VaR 95%  | Value at Risk at 95% confidence level  |
| CVaR 95%  | Conditional Value at Risk (Expected Shortfall)  |
| Delta Variance | Price sensitivity volatility |
| Gamma Risk | Convexity risk measure |
| Vega Exposure | Volatility sensitivity |

Investment Recommendations

- Conservative Investors: Low-risk, stable return focus
- Moderate Investors: Balanced risk-return profile
- Aggressive Investors: High-risk, high-return potential

## 🔍 Code Structure

```python
class EnhancedMalliavinAnalysis:
    """
    Main analysis class combining Malliavin calculus with ML
    """
    
    def __init__(self, ticker, period="2y", device='cpu'):
        """Initialize analyzer with stock symbol and parameters"""
    
    def train_neural_network(self, epochs=1000):
        """Train deep learning model on Malliavin derivatives"""
    
    def enhanced_sensitivity_analysis(self, n_scenarios=10000):
        """Generate comprehensive sensitivity analysis"""
    
    def generate_investment_recommendation(self):
        """AI-powered investment guidance"""

```
## ⚙️ Configuration Options

| Parameter  | Description | Default | Range |
| ------------- | ------------- | ------------- | ------------- |
| ticker | Stock symbol to analyze  | 'AAPL' | Any valid ticker |
| period  | Historical data period  | '2y' |	'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y' |
| device | Computation device | 'cpu' |	'cpu', 'cuda' |
| epochs | Neural network training epochs | 1000 |	100-5000 |
| n_scenarios | Monte Carlo scenarios | 10000	| 1000-100000 |

## 📈 Example Output

The analysis generates comprehensive reports including:

- Neural Network Training Performance: Model accuracy and convergence analysis
- Price Distribution Predictions: Expected returns and volatility forecasts
- Sensitivity Analysis: Delta, Gamma, and Vega across time horizons
- Historical Performance Review: Trend analysis and momentum indicators
- Investment Recommendations: Tailored advice for different risk profiles

## 📚 References

- Malliavin, P. (1997). Stochastic Analysis
- Nualart, D. (2006). The Malliavin Calculus and Related Topics
- Goodfellow, I. et al. (2016). Deep Learning

## 🙏 Acknowledgments

- yfinance for financial data access
- PyTorch for deep learning framework
- Malliavin calculus research community for mathematical foundations

## 📧 Contact

Yavuz Akbay - akbay.yavuz@gmail.com

⭐️ If this project helped with your financial analysis, please consider giving it a star!

🚨 Disclaimer
This tool is for educational and research purposes only. Not financial advice. Always consult with qualified financial professionals before making investment decisions.
