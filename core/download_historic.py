import pandas as pd
import time
from binance.client import Client


class Binance:
    def __init__(self):
        self.api_key = ''
        self.api_secret = ''
        self.client = Client()

    def download_binance_futures_data(self, symbol, interval, start_date, end_date, filename):
        """
        Downloads historical futures data from Binance in chunks and saves it to a CSV file.

        :param symbol: Trading pair (e.g., 'BTCUSDT')
        :param interval: Kline interval (e.g., '1m', '5m', '15m', '1h', '1d')
        :param start_date: Start date in 'YYYY-MM-DD' format
        :param end_date: End date in 'YYYY-MM-DD' format
        :param filename: Output CSV filename
        """
        all_klines = []
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        while start_ts < end_ts:
            klines = self.client.futures_historical_klines(symbol, interval, start_str=start_ts, end_str=end_ts,
                                                           limit=1000)

            if not klines:
                break  # Stop if no more data

            all_klines.extend(klines)

            # Update the start timestamp to continue from the last received data point
            start_ts = klines[-1][0] + 1

            print(f"Fetched {len(klines)} records. Continuing from {pd.to_datetime(start_ts, unit='ms')}")
            time.sleep(1)  # To avoid rate limits

        # Convert to DataFrame
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(all_klines, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


# Example Usage
if __name__ == '__main__':
    mybinance = Binance()
    mybinance.download_binance_futures_data(symbol="BTCUSDT", interval="15m", start_date="2018-09-01",
                                            end_date="2025-01-30",
                                            filename="binance_futures_data.csv")