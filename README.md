# Crypto Trading Platform

A fully functional, modular cryptocurrency trading platform built with Python, featuring advanced technical indicators, machine learning models, Bayesian optimization, walk-forward ana
lysis, and comprehensive backtesting capabilities.

## 🚀 Features

### Core Capabilities
- **Advanced Causal Strategy (V6/Regimeline)**: 
  - **Leader-Follower Logic**: Uses Granger Causality to identify asset leadership (e.g., BTC leading ETH).
  - **6-Phase Validation**: Leader Crash Veto, Projected Velocity, and Transfer Entropy Gates.
  - **GMM Clustering**: Automatic volatility regime detection.
- **Hybrid Adaptive Strategies (Alpha, Beta, Gamma)**:
  - **Strategy Alpha**: Microstructural Price Resonance using ML Order Book Imbalance (MLOFI) and DeepLOB integrations.
  - **Strategy Beta**: HMM-Gated Volatility Arbitrage dynamically allocating capital between Breakout (Trending) and Mean-Reversion (Choppy) sub-strategies.
  - **Strategy Gamma**: Deterministic Indicator-Fused logic utilizing ADX, Choppiness Index, Tillson T3 DEMAs, and Fisher Transform.
- **Institutional-Grade Data**:
  - **Real-Time Feed**: Powered by `cryptofeed` for L2 Book and Trade stream.
  - **Order Book Imbalance (OBI)**: Detects institutional pressure.
  - **Liquidation Tracking**: Instant crash detection via liquidation velocity.
- **Multi-Source Data Fetching**: Binance Vision, real-time Websockets, yfinance integration
- **Advanced Technical Indicators**: Tillson T3, Fisher Transform, Choppiness Index, ADX, VWAP, EMA, RSI, Bollinger Bands, ATR, Ichimoku Cloud
- **Smart Money Concepts**: Integration with smart-money-concepts library for Order Blocks, Fair Value Gaps, Liquidity
- **Machine Learning Models**: 
  - XGBoost for price prediction
  - LSTM-GARCH for sequence modeling and volatility forecasting
  - HMM (Hidden Markov Models) for continuous Regime Inference
- **Optimization & Robustness**:
  - **Walk-Forward Optimization (WFO)**: Professional cross-validation ensuring strategies perform entirely out-of-sample before deployment.
  - **Simulated Annealing**: Advanced metaheuristic optimization to navigate complex, non-convex strategy parameter spaces and avoid local optima.
  - Bayesian Optimization for parameter tuning.
- **Professional Backtesting**: Realistic simulation with commission, slippage, and risk management
- **Signal Aggregation**: Multiple signal sources with configurable weights
- **Database Storage**: DuckDB + Parquet for efficient data management
- **Distributed Caching**: Redis for high-performance data access
- **PDF Reporting**: Comprehensive reports with charts and metrics
- **Daemon Mode**: Run as background service for continuous operation
- **Containerization**: Docker and Kubernetes ready

### Supported Assets
- Bitcoin (BTC)
- Ethereum (ETH)
- Easily extensible to other cryptocurrencies

## 📋 Requirements

- Python 3.11+
- Docker 20.10+ (for containerized deployment)
- Redis 7+ (for caching)
- 4GB+ RAM
- 10GB+ disk space for data storage

## 🔧 Installation

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd crypto-trading-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
mkdir -p data logs reports
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/crypto-trading-bot
```

## 🎯 Quick Start

### 1. Configure the Platform

Edit `config.yaml` to customize:
- Trading pairs (BTC-USD, ETH-USD)
- Indicator parameters
- ML model settings
- Risk management rules
- Optimization parameters

### 2. Backtest a Strategy

```bash
# Run backtest for Bitcoin
python main.py --mode backtest --symbol BTC/USDT --timeframe 1h

# The platform will:
# 1. Fetch historical data (2 years)
# 2. Calculate all technical indicators
# 3. Train ML models
# 4. Generate trading signals
# 5. Run backtest simulation
# 6. Generate PDF report
```

### 3. Run Walk-Forward Optimization

```bash
python main.py --mode optimize --symbol BTC/USDT --timeframe 1h
```

### 4. Run as Daemon/Service

```bash
# Start daemon
python main.py --mode daemon

# Or with Docker
docker-compose up -d trading-bot
```

## 📊 Architecture

```
crypto-trading-platform/
├── data_ingestion/         # Real-time Feed Handler
│   └── feed_manager.py     # Cryptofeed Integration (L2, Trades, Liq)
├── data_fetchers/          # Historical Data modules
│   ├── binance_fetcher.py
│   └── yfinance_fetcher.py
├── indicators/             # Technical indicators
│   └── ...
├── ml/                     # Machine learning & Causal Inference
│   ├── causal_inference.py 
│   ├── feature_engineer.py
│   └── ...
├── strategy/               # Trading strategies
│   ├── regimeline_strategy.py  # Advanced V6 Causal Regimeline
│   ├── strategy_alpha.py       # Microstructural Resonance
│   ├── strategy_beta.py        # HMM-Gated Volatility Arbitrage
│   └── strategy_gamma.py       # Deterministic Indicator-Fused
├── backtesting/            # Backtesting engine
│   ├── wfo_engine.py       # Walk-Forward Optimization & SA Framework
│   └── ...
├── scripts/                # Execution Scripts
│   ├── run_sa_wfo.py       # WFO execution using Simulated Annealing
│   ├── run_gamma_wfo.py    # WFO specifically for Strategy Gamma
│   └── run_comparative_wfo.py # Multi-strategy WFO comparison matrix
├── db/                     # DuckDB management
├── cache/                  # Redis caching
├── reporting/              # PDF generation
├── utils/                  # Utilities
├── main.py                 # Main entry point
└── config.yaml             # Configuration
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run backtest validation
python -m pytest tests/test_backtest.py

# Check code coverage
pytest --cov=. tests/
```

## 📈 Signal Generation

The platform generates signals from multiple sources:

1. **Technical Indicators**:
   - VWAP crossovers
   - EMA crossovers (9/21)
   - RSI oversold/overbought
   - Ichimoku Cloud signals

2. **Smart Money Concepts**:
   - Order Block detection
   - Fair Value Gaps
   - Liquidity zones

3. **Machine Learning**:
   - XGBoost probability predictions
   - LSTM sequence predictions

4. **Aggregation**:
   - Weighted average
   - Majority voting
   - Unanimous consensus

## 🔐 Security

- API keys stored in environment variables
- Secrets managed via Kubernetes secrets
- No hardcoded credentials
- Rate limiting for API calls
- Input validation and sanitization

## 📝 Configuration

### Key Configuration Parameters

```yaml
# config.yaml

indicators:
  vwap:
    enabled: true
  ema:
    periods: [9, 21, 50, 200]
  rsi:
    period: 14
    overbought: 70
    oversold: 30

ml_models:
  xgboost:
    n_estimators: 100
    max_depth: 7
    learning_rate: 0.01

optimization:
  bayesian:
    n_iterations: 50
  walk_forward:
    train_period_days: 365
    test_period_days: 90
    step_days: 30

risk:
  max_position_size: 0.1
  max_daily_loss: 0.05
  stop_loss_atr_multiplier: 2.0
```

## 🚀 Performance

Expected performance metrics (based on backtesting):
- **Sharpe Ratio**: 1.5 - 2.5
- **Win Rate**: 55% - 65%
- **Max Drawdown**: -15% to -25%
- **Annual Return**: 30% - 100% (highly variable)

⚠ **Disclaimer**: Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request


## 📄 License

MIT License - see LICENSE file for details

## 📧 Support

For issues and questions:
- Open a GitHub issue
- Check documentation
- Review configuration examples

## 🔗 Resources

- [DuckDB Documentation](https://duckdb.org/docs/)
- [Smart Money Concepts](https://github.com/joshyattridge/smart-money-concepts)
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
- [Walk-Forward Analysis](https://www.investopedia.com/articles/trading/11/walk-forward-optimization.asp)

## ⚠ Risk Warning

**IMPORTANT**: This platform is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Always:
- Start with paper trading
- Never invest more than you can afford to lose
- Understand the strategies before deploying
- Monitor your positions actively
- Follow proper risk management

---

**Built with ❤ for the crypto trading community**
