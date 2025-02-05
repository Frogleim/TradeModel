import backtrader as bt
import pandas as pd
import pandas_ta as ta
from bayes_opt import BayesianOptimization
import random

# Load historical data
df = pd.read_csv("historical_data.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Compute initial indicators
df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
df['atr'] = ta.atr(df['high'], df['low'], df['close'])
df['rsi'] = ta.rsi(df['close'])

# Drop NaN values
df.dropna(inplace=True)


class OptimizedStrategy(bt.Strategy):
    params = dict(ema_short=20, ema_long=50, adx_threshold=20, rsi_low=30, rsi_high=70)

    def __init__(self):
        self.ema_short = bt.indicators.EMA(period=int(self.params.ema_short))
        self.ema_long = bt.indicators.EMA(period=int(self.params.ema_long))
        self.adx = bt.indicators.ADX()
        self.atr = bt.indicators.ATR()
        self.rsi = bt.indicators.RSI()

    def next(self):
        if self.ema_short > self.ema_long and self.adx > self.params.adx_threshold and self.rsi > self.params.rsi_low:
            self.buy()
        elif self.ema_short < self.ema_long and self.adx > self.params.adx_threshold and self.rsi < self.params.rsi_high:
            self.sell()


# Define objective function for Bayesian Optimization
def optimize_strategy(ema_short, ema_long, adx_threshold, rsi_low, rsi_high):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(OptimizedStrategy,
                        ema_short=int(ema_short),
                        ema_long=int(ema_long),
                        adx_threshold=int(adx_threshold),
                        rsi_low=int(rsi_low),
                        rsi_high=int(rsi_high))

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    results = cerebro.run()
    strategy = results[0]

    wins = 0
    losses = 0

    for order in strategy.broker.orders:
        if order.status == bt.Order.Completed:
            if order.executed.pnl > 0:
                wins += 1
            else:
                losses += 1

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    return win_rate  # We maximize win rate


# Bayesian Optimization
optimizer = BayesianOptimization(
    f=optimize_strategy,
    pbounds={
        "ema_short": (10, 50),
        "ema_long": (30, 100),
        "adx_threshold": (10, 40),
        "rsi_low": (20, 40),
        "rsi_high": (60, 80),
    },
    random_state=42,
)

optimizer.maximize(init_points=5, n_iter=30)

best_params = optimizer.max['params']
print("\n✅ Best Trading Strategy Parameters:")
print(best_params)