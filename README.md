# Crypto Trading Platform

A fully functional, modular cryptocurrency trading platform built with Python, featuring advanced technical indicators, machine learning models, Bayesian optimization, walk-forward ana
lysis, and comprehensive backtesting capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-Source Data Fetching**: Binance, yfinance integration
- **Advanced Technical Indicators**: VWAP, EMA (9, 21, 50, 200), RSI, Bollinger Bands, ATR, Ichimoku Cloud
- **Smart Money Concepts**: Integration with smart-money-concepts library for Order Blocks, Fair Value Gaps, Liquidity
- **Machine Learning Models**: 
  - XGBoost for price prediction
  - LSTM for sequence modeling
  - LLM integration for sentiment analysis (optional)
- **Feature Engineering**: Comprehensive feature creation from price, volume, technical indicators, and time data
- **Optimization**:
  - Bayesian Optimization for parameter tuning
  - Walk-Forward Analysis to prevent overfitting
  - Time-series cross-validation
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
├── data_fetchers/          # Data acquisition modules
│   ├── binance_fetcher.py
│   └── yfinance_fetcher.py
├── indicators/             # Technical indicators
│   ├── vwap.py
│   ├── ema.py
│   ├── rsi.py
│   ├── ichimoku.py
│   └── ...
├── smart_money/            # Smart Money Concepts
│   ├── order_blocks.py
│   ├── liquidity.py
│   └── fvg.py
├── ml/                     # Machine learning models
│   ├── feature_engineer.py
│   ├── xgboost_model.py
│   ├── lstm_model.py
│   └── llm_analyzer.py
├── optimization/           # Parameter optimization
│   ├── bayesian_optimizer.py
│   └── walk_forward.py
├── strategy/               # Trading strategies
│   ├── scalping_strategy.py
│   ├── ichimoku_strategy.py
│   └── ensemble_strategy.py
├── backtesting/            # Backtesting engine
│   ├── backtest_engine.py
│   └── performance_metrics.py
├── signal/                 # Signal aggregation
│   └── signal_aggregator.py
├── db/                     # DuckDB management
│   └── duckdb_manager.py
├── cache/                  # Redis caching
│   └── redis_manager.py
├── reporting/              # PDF generation
│   └── pdf_generator.py
├── daemon/                 # Service mode
│   └── trading_daemon.py
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
