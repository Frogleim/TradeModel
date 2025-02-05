import csv
import random

# Define the CSV filename
filename = "trades.csv"

# Define possible symbols and trade sides
symbols = ["BTCUSDT"]
sides = ["BUY", "SELL"]

# Generate and write the data
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow([
        "id", "symbol", "entry_price", "exit_price", "pnl", "long_ema", "short_ema",
        "adx", "atr", "rsi", "volume", "side"
    ])

    # Generate 200 trade entries
    for i in range(1, 100001):
        symbol = random.choice(symbols)
        entry_price = round(random.uniform(100000, 105000), 2)
        exit_price = round(entry_price * random.uniform(0.95, 1.05), 2)
        pnl = round(exit_price - entry_price, 2)
        long_ema = round(random.uniform(10, 50), 2)
        short_ema = round(random.uniform(5, 30), 2)
        adx = round(random.uniform(20, 60) if pnl > 0 else random.uniform(10, 40), 2)
        atr = round(entry_price * random.uniform(0.002, 0.01), 2)
        rsi = round(random.uniform(40, 80) if pnl > 0 else random.uniform(10, 50), 2)
        volume = round(random.uniform(100000, 105000), 2)
        side = random.choice(sides)

        # Write row
        writer.writerow([i, symbol, entry_price, exit_price, pnl, long_ema, short_ema, adx, atr, rsi, volume, side])

print(f"CSV file '{filename}' generated with 200 trade entries.")