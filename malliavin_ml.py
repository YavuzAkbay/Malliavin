import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MalliavinNeuralNetwork(nn.Module):
    """
    Neural network for approximating Malliavin derivatives and sensitivities
    """
    def __init__(self, input_dim, hidden_dims=[64, 128, 64], output_dim=3):
        super(MalliavinNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class EnhancedMalliavinAnalysis:
    def __init__(self, ticker, period="2y", device='cpu'):
        self.ticker = ticker
        self.period = period
        self.device = device
        self.stock_data = None
        self.returns = None
        self.volatility = None
        self.risk_free_rate = None
        self.scaler = StandardScaler()
        self.neural_net = None
        self.training_history = []
        
    def fetch_data(self):
        """Enhanced data fetching with feature engineering"""
        stock = yf.Ticker(self.ticker)
        self.stock_data = stock.history(period=self.period)
        
        self.stock_data['Returns'] = self.stock_data['Close'].pct_change()
        self.stock_data['LogReturns'] = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1))
        self.stock_data['Volatility_5d'] = self.stock_data['Returns'].rolling(5).std()
        self.stock_data['Volatility_20d'] = self.stock_data['Returns'].rolling(20).std()
        self.stock_data['RSI'] = self.calculate_rsi(self.stock_data['Close'])

        self.stock_data = self.stock_data.dropna()
        self.returns = self.stock_data['Returns']
        self.volatility = self.returns.std() * np.sqrt(252)
        
        try:
            treasury = yf.Ticker("^TNX")
            treasury_data = treasury.history(period="5d")
            self.risk_free_rate = treasury_data['Close'].iloc[-1] / 100
        except:
            self.risk_free_rate = 0.05
        
        print(f"Enhanced data fetched for {self.ticker}")
        print(f"Data points: {len(self.stock_data)}")
        print(f"Features: Returns, LogReturns, Volatility_5d, Volatility_20d, RSI")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_training_data(self, n_paths=50000, time_horizons=[0.25, 0.5, 1.0]):
        """
        Generate comprehensive training data for neural network
        combining multiple time horizons and market scenarios
        """
        current_price = self.stock_data['Close'].iloc[-1]
        training_features = []
        training_targets = []
        
        for T in time_horizons:
            for _ in range(n_paths // len(time_horizons)):

                scenario = self.generate_market_scenario(current_price, T)
                
                malliavin_data = self.calculate_analytical_malliavin(scenario)
                
                training_features.append(scenario['features'])
                training_targets.append(malliavin_data)
        
        return np.array(training_features), np.array(training_targets)
    
    def generate_market_scenario(self, S0, T):
        """Generate realistic market scenario with multiple features - FIXED VERSION"""
        dt = T / 252
        n_steps = max(1, int(T * 252))
        
        Z = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_steps)
        
        S = S0
        vol_path = []
        price_path = [S0]
        
        for i in range(n_steps):
            vol_shock = 0.1 * Z[i, 1]
            current_vol = max(0.1, self.volatility + vol_shock)
            vol_path.append(current_vol)
            
            S = S * np.exp((self.risk_free_rate - 0.5 * current_vol**2) * dt + 
                          current_vol * np.sqrt(dt) * Z[i, 0])
            price_path.append(S)
        
        features = [
            S / S0,
            np.mean(vol_path) if vol_path else self.volatility,
            np.max(vol_path) if vol_path else self.volatility,
            np.min(vol_path) if vol_path else self.volatility,
            T,
            1.0
        ]
        
        return {
            'features': features,
            'final_price': S,
            'vol_path': vol_path,
            'price_path': price_path
        }
    
    def calculate_analytical_malliavin(self, scenario):
        """
        Calculate analytical Malliavin derivatives for training targets
        Returns: [delta, gamma, vega] approximations
        """
        S_final = scenario['final_price']
        S0 = self.stock_data['Close'].iloc[-1]
        avg_vol = np.mean(scenario['vol_path']) if scenario['vol_path'] else self.volatility
        
        delta = (S_final - S0) / S0 if S0 != 0 else 0
        
        gamma = abs(delta) * avg_vol
        
        vega = S0 * np.sqrt(scenario['features'][4]) * stats.norm.pdf(delta / avg_vol) if avg_vol > 0 else 0
        
        return [delta, gamma, vega]
    
    def train_neural_network(self, epochs=1000, batch_size=512, learning_rate=0.001):
        """Train neural network on Malliavin calculus data"""
        print("Generating training data...")
        X_train, y_train = self.generate_training_data()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        input_dim = X_train_scaled.shape[1]
        self.neural_net = MalliavinNeuralNetwork(input_dim).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training neural network for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            self.neural_net.train()
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.neural_net(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            self.training_history.append(avg_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        print("Training completed!")
    
    def predict_malliavin_derivatives(self, market_features):
        """Use trained neural network to predict Malliavin derivatives"""
        if self.neural_net is None:
            raise ValueError("Neural network not trained yet!")
        
        features_scaled = self.scaler.transform([market_features])
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        self.neural_net.eval()
        with torch.no_grad():
            predictions = self.neural_net(features_tensor)
        
        return predictions.cpu().numpy()[0]
    
    def enhanced_sensitivity_analysis(self, n_scenarios=10000):
        """
        Enhanced sensitivity analysis using ML-powered Malliavin calculus
        """
        current_price = self.stock_data['Close'].iloc[-1]
        results = []
        
        print("Running enhanced sensitivity analysis...")
        
        for i in range(n_scenarios):
            T = np.random.uniform(1/12, 2.0)
            
            scenario = self.generate_market_scenario(current_price, T)
            
            sensitivities = self.predict_malliavin_derivatives(scenario['features'])
            
            results.append({
                'time_horizon': T,
                'final_price': scenario['final_price'],
                'delta': sensitivities[0],
                'gamma': sensitivities[1],
                'vega': sensitivities[2],
                'avg_volatility': np.mean(scenario['vol_path']) if scenario['vol_path'] else self.volatility
            })
        
        return pd.DataFrame(results)
    
    def advanced_risk_metrics(self, scenarios_df):
        """Calculate advanced risk metrics from ML predictions"""
        current_price = self.stock_data['Close'].iloc[-1]
        
        returns = (scenarios_df['final_price'] / current_price - 1) * 100
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        delta_var = np.var(scenarios_df['delta'])
        gamma_risk = np.mean(np.abs(scenarios_df['gamma']))
        vega_exposure = np.std(scenarios_df['vega'])
        
        return {
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Delta_Variance': delta_var,
            'Gamma_Risk': gamma_risk,
            'Vega_Exposure': vega_exposure,
            'Max_Loss_Probability': len(returns[returns < -10]) / len(returns)
        }
    
    def interpret_neural_network_training(self):
        """Provide detailed interpretation of neural network training results"""
        final_loss = self.training_history[-1]
        initial_loss = self.training_history[0]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        interpretation = f"""
🧠 NEURAL NETWORK TRAINING ANALYSIS

Training Performance:
• Initial Loss: {initial_loss:.6f}
• Final Loss: {final_loss:.6f}
• Improvement: {improvement:.1f}%

What This Means for Investors:
"""
        
        if improvement > 90:
            interpretation += """
✅ EXCELLENT MODEL QUALITY: The AI model learned very effectively, reducing prediction 
   errors by over 90%. This means the sensitivity calculations are highly reliable.
   
📈 Investment Confidence: You can trust the risk metrics and price predictions 
   generated by this model for making investment decisions."""
        elif improvement > 70:
            interpretation += """
✅ GOOD MODEL QUALITY: The AI model shows solid learning with 70%+ error reduction.
   The predictions are reliable for most investment scenarios.
   
📊 Moderate Confidence: The model provides good guidance, but consider combining 
   with other analysis methods for major investment decisions."""
        else:
            interpretation += """
⚠️ MODERATE MODEL QUALITY: The model learned but with limited improvement.
   
🔍 Use with Caution: Consider the predictions as one input among many in your 
   investment decision process."""
        
        return interpretation
    
    def interpret_price_distribution(self, scenarios_df, risk_metrics):
        """Provide detailed interpretation of price distribution analysis"""
        current_price = self.stock_data['Close'].iloc[-1]
        returns = (scenarios_df['final_price'] / current_price - 1) * 100
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        upside_prob = len(returns[returns > 0]) / len(returns)
        big_gain_prob = len(returns[returns > 20]) / len(returns)
        big_loss_prob = len(returns[returns < -20]) / len(returns)
        
        interpretation = f"""
📊 PRICE PREDICTION ANALYSIS

Expected Outcomes:
• Average Expected Return: {mean_return:.1f}%
• Return Volatility: {std_return:.1f}%
• Probability of Gains: {upside_prob:.1%}
• Probability of Big Gains (>20%): {big_gain_prob:.1%}
• Probability of Big Losses (<-20%): {big_loss_prob:.1%}

Risk Assessment:
• Value at Risk (95%): {risk_metrics['VaR_95']:.1f}%
• Expected Shortfall: {risk_metrics['CVaR_95']:.1f}%

Investor Guidance:
"""
        
        if mean_return > 5:
            interpretation += """
🚀 POSITIVE OUTLOOK: The model predicts positive average returns, suggesting 
   potential for capital appreciation.
   
💡 Strategy: Consider this stock for growth-oriented portfolios."""
        elif mean_return > -2:
            interpretation += """
📈 NEUTRAL OUTLOOK: Expected returns are close to zero, indicating sideways movement.
   
💡 Strategy: Suitable for income-focused investors if the stock pays dividends."""
        else:
            interpretation += """
📉 NEGATIVE OUTLOOK: The model predicts negative average returns.
   
⚠️ Strategy: Exercise caution. Consider waiting for better entry points or 
   look for alternative investments."""
        
        if std_return > 30:
            interpretation += f"""
   
🎢 HIGH VOLATILITY: With {std_return:.0f}% volatility, expect significant price swings.
   This stock is suitable for risk-tolerant investors only."""
        elif std_return > 20:
            interpretation += f"""
   
📊 MODERATE VOLATILITY: {std_return:.0f}% volatility indicates normal market risk.
   Suitable for most diversified portfolios."""
        else:
            interpretation += f"""
   
🛡️ LOW VOLATILITY: {std_return:.0f}% volatility suggests stable price movements.
   Good for conservative investors."""
        
        return interpretation
    
    def interpret_delta_sensitivity(self, scenarios_df):
        """Provide detailed interpretation of delta sensitivity analysis"""
        avg_delta = np.mean(scenarios_df['delta'])
        delta_std = np.std(scenarios_df['delta'])
        max_delta = np.max(scenarios_df['delta'])
        min_delta = np.min(scenarios_df['delta'])
        
        # Analyze delta by time horizon
        short_term = scenarios_df[scenarios_df['time_horizon'] <= 0.5]['delta']
        long_term = scenarios_df[scenarios_df['time_horizon'] > 1.0]['delta']
        
        interpretation = f"""
⚡ PRICE SENSITIVITY ANALYSIS (DELTA)

Sensitivity Metrics:
• Average Delta: {avg_delta:.3f}
• Delta Volatility: {delta_std:.3f}
• Range: {min_delta:.3f} to {max_delta:.3f}

Time Horizon Impact:
• Short-term Average Delta: {np.mean(short_term):.3f}
• Long-term Average Delta: {np.mean(long_term):.3f}

What Delta Means for Investors:
"""
        
        if abs(avg_delta) > 0.5:
            interpretation += """
🔥 HIGH SENSITIVITY: This stock is very sensitive to market movements.
   
📈 Opportunity: In bull markets, expect amplified gains
📉 Risk: In bear markets, expect amplified losses
💡 Strategy: Perfect for momentum trading, risky for buy-and-hold"""
        elif abs(avg_delta) > 0.2:
            interpretation += """
📊 MODERATE SENSITIVITY: Normal market responsiveness.
   
⚖️ Balanced: Good balance between growth potential and stability
💡 Strategy: Suitable for diversified portfolios"""
        else:
            interpretation += """
🛡️ LOW SENSITIVITY: This stock moves independently of broader market trends.
   
🎯 Defensive: Less affected by market crashes
💡 Strategy: Good for defensive portfolios and market hedging"""
        
        if np.mean(long_term) > np.mean(short_term):
            interpretation += """
   
⏰ TIME EFFECT: Sensitivity increases over longer time horizons.
   Long-term investments carry higher market risk but also higher return potential."""
        else:
            interpretation += """
   
⏰ TIME EFFECT: Sensitivity decreases over longer time horizons.
   Long-term investments may be more stable than short-term trades."""
        
        return interpretation
    
    def interpret_stock_history(self):
        """Provide interpretation of historical stock performance"""
        current_price = self.stock_data['Close'].iloc[-1]
        start_price = self.stock_data['Close'].iloc[0]
        total_return = (current_price / start_price - 1) * 100
        
        max_price = self.stock_data['Close'].max()
        min_price = self.stock_data['Close'].min()
        current_from_high = (current_price / max_price - 1) * 100
        current_from_low = (current_price / min_price - 1) * 100
        
        recent_data = self.stock_data.tail(30)
        recent_return = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100
        
        interpretation = f"""
📈 HISTORICAL PERFORMANCE ANALYSIS

Overall Performance:
• Total Return: {total_return:.1f}% over {len(self.stock_data)} trading days
• Current Price: ${current_price:.2f}
• Distance from High: {current_from_high:.1f}%
• Distance from Low: {current_from_low:.1f}%

Recent Trend (30 days):
• Recent Performance: {recent_return:.1f}%

Investment Timing Analysis:
"""
        
        if current_from_high > -10:
            interpretation += """
🚀 NEAR ALL-TIME HIGHS: Stock is trading close to its peak levels.
   
⚠️ Caution: May face resistance at current levels
💡 Strategy: Consider waiting for pullbacks or dollar-cost averaging"""
        elif current_from_high > -25:
            interpretation += """
📊 MODERATE PULLBACK: Stock has pulled back from highs but not drastically.
   
⚖️ Opportunity: Potential buying opportunity if fundamentals are strong
💡 Strategy: Good entry point for long-term investors"""
        else:
            interpretation += """
🔥 SIGNIFICANT DISCOUNT: Stock is trading well below recent highs.
   
💎 Value Opportunity: Potentially attractive entry point
💡 Strategy: Investigate why the stock declined - could be a bargain"""
        
        if recent_return > 5:
            interpretation += f"""
   
📈 STRONG MOMENTUM: Recent {recent_return:.1f}% gain shows positive momentum.
   Consider riding the trend but watch for overbought conditions."""
        elif recent_return < -5:
            interpretation += f"""
   
📉 RECENT WEAKNESS: {recent_return:.1f}% decline may indicate selling pressure.
   Could be a buying opportunity or signal of deeper issues."""
        else:
            interpretation += """
   
📊 STABLE RECENT PERFORMANCE: Price has been relatively stable recently.
   Good for investors seeking predictable movements."""
        
        return interpretation
    
    def generate_investment_recommendation(self, scenarios_df, risk_metrics):
        """Generate comprehensive investment recommendation"""
        current_price = self.stock_data['Close'].iloc[-1]
        returns = (scenarios_df['final_price'] / current_price - 1) * 100
        mean_return = np.mean(returns)
        
        if self.volatility > 0.4:
            risk_level = "VERY HIGH"
            risk_score = 5
        elif self.volatility > 0.3:
            risk_level = "HIGH"
            risk_score = 4
        elif self.volatility > 0.2:
            risk_level = "MODERATE"
            risk_score = 3
        elif self.volatility > 0.15:
            risk_level = "LOW-MODERATE"
            risk_score = 2
        else:
            risk_level = "LOW"
            risk_score = 1
        
        if mean_return > 10:
            return_potential = "EXCELLENT"
            return_score = 5
        elif mean_return > 5:
            return_potential = "GOOD"
            return_score = 4
        elif mean_return > 0:
            return_potential = "MODERATE"
            return_score = 3
        elif mean_return > -5:
            return_potential = "POOR"
            return_score = 2
        else:
            return_potential = "VERY POOR"
            return_score = 1
        
        overall_score = (return_score + (6 - risk_score)) / 2
        
        recommendation = f"""
🎯 INVESTMENT RECOMMENDATION FOR {self.ticker}

OVERALL RATING: {overall_score:.1f}/5.0

Risk Level: {risk_level} (Annual Volatility: {self.volatility:.1%})
Return Potential: {return_potential} (Expected: {mean_return:.1f}%)

RECOMMENDATION BY INVESTOR TYPE:

🏛️ Conservative Investors (Low Risk Tolerance):
"""
        
        if risk_score <= 2 and return_score >= 3:
            recommendation += "✅ RECOMMENDED - Good risk-adjusted returns"
        elif risk_score <= 2:
            recommendation += "⚠️ CONSIDER - Low risk but limited upside"
        else:
            recommendation += "❌ NOT RECOMMENDED - Too risky for conservative portfolios"
        
        recommendation += f"""

⚖️ Moderate Investors (Balanced Approach):
"""
        
        if overall_score >= 3.5:
            recommendation += "✅ RECOMMENDED - Good balance of risk and return"
        elif overall_score >= 2.5:
            recommendation += "⚠️ CONSIDER - Acceptable but monitor closely"
        else:
            recommendation += "❌ NOT RECOMMENDED - Poor risk-return profile"
        
        recommendation += f"""

🚀 Aggressive Investors (High Risk Tolerance):
"""
        
        if return_score >= 4:
            recommendation += "✅ RECOMMENDED - High return potential"
        elif return_score >= 3:
            recommendation += "⚠️ CONSIDER - Moderate return potential"
        else:
            recommendation += "❌ NOT RECOMMENDED - Insufficient return potential"
        
        recommendation += f"""

💰 POSITION SIZING GUIDANCE:
"""
        
        if risk_score <= 2:
            recommendation += "• Conservative: Up to 10% of portfolio\n• Moderate: Up to 15% of portfolio\n• Aggressive: Up to 20% of portfolio"
        elif risk_score <= 3:
            recommendation += "• Conservative: Up to 5% of portfolio\n• Moderate: Up to 10% of portfolio\n• Aggressive: Up to 15% of portfolio"
        else:
            recommendation += "• Conservative: Avoid or <2% of portfolio\n• Moderate: Up to 5% of portfolio\n• Aggressive: Up to 10% of portfolio"
        
        return recommendation
    
    def visualize_enhanced_analysis(self):
        """Create comprehensive visualizations with detailed interpretations"""
        self.fetch_data()
        self.train_neural_network()
        
        scenarios_df = self.enhanced_sensitivity_analysis()
        risk_metrics = self.advanced_risk_metrics(scenarios_df)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ML-Enhanced Malliavin Analysis for {self.ticker}', fontsize=16, fontweight='bold')
        
        ax1.plot(self.training_history)
        ax1.set_title('Neural Network Training Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        current_price = self.stock_data['Close'].iloc[-1]
        price_returns = (scenarios_df['final_price'] / current_price - 1) * 100
        ax2.hist(price_returns, bins=50, alpha=0.7, color='blue', density=True)
        ax2.axvline(0, color='red', linestyle='--', label='Current Price')
        ax2.axvline(risk_metrics['VaR_95'], color='orange', linestyle='--', label=f'VaR 95%: {risk_metrics["VaR_95"]:.1f}%')
        ax2.set_title('ML-Predicted Price Distribution', fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.scatter(scenarios_df['time_horizon'], scenarios_df['delta'], 
                   c=scenarios_df['avg_volatility'], cmap='viridis', alpha=0.6)
        plt.colorbar(ax3.collections[0], ax=ax3, label='Average Volatility')
        ax3.set_title('Delta Sensitivity vs Time Horizon', fontweight='bold')
        ax3.set_xlabel('Time Horizon (Years)')
        ax3.set_ylabel('Delta')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(self.stock_data.index, self.stock_data['Close'], linewidth=2, color='blue')
        ax4.set_title('Stock Price History', fontweight='bold')
        ax4.set_ylabel('Price ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*100)
        print("🤖 COMPREHENSIVE ML-ENHANCED MALLIAVIN ANALYSIS REPORT")
        print("="*100)
        
        print(self.interpret_neural_network_training())
        print("\n" + "-"*80)
        print(self.interpret_price_distribution(scenarios_df, risk_metrics))
        print("\n" + "-"*80)
        print(self.interpret_delta_sensitivity(scenarios_df))
        print("\n" + "-"*80)
        print(self.interpret_stock_history())
        print("\n" + "-"*80)
        print(self.generate_investment_recommendation(scenarios_df, risk_metrics))
        
        print("\n" + "="*100)
        print("📊 TECHNICAL SUMMARY")
        print("="*100)
        print(f"• Analysis Date: May 25, 2025")
        print(f"• Scenarios Analyzed: {len(scenarios_df):,}")
        print(f"• Neural Network Accuracy: {((self.training_history[0] - self.training_history[-1]) / self.training_history[0] * 100):.1f}% improvement")
        print(f"• Data Points Used: {len(self.stock_data):,}")
        print(f"• Risk-Free Rate: {self.risk_free_rate:.2%}")
        print("• Method: ML-Enhanced Malliavin Calculus with Deep Learning")
        
        return scenarios_df, risk_metrics

if __name__ == "__main__":
    analyzer = EnhancedMalliavinAnalysis("AAPL")
    scenarios_df, risk_metrics = analyzer.visualize_enhanced_analysis()
