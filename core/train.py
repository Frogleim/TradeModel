import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib  # To save models
import os

# Load dataset
df = pd.read_csv("trades.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
df["side"] = label_encoder.fit_transform(df["side"])

# Features and target columns
features = ["entry_price", "exit_price", "long_ema", "short_ema", "adx", "atr", "rsi", "volume", "side"]
target = ["long_ema", "short_ema", "adx", "atr", "rsi"]  # Indicators to predict

# Train a model for each symbol separately
models = {}

for symbol in df["symbol"].unique():
    print(f"\n🔵 Training model for {symbol}...")

    # Filter data for this symbol
    symbol_df = df[df["symbol"] == symbol]

    X = symbol_df[features]
    y = symbol_df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    print(f"📉 Mean Absolute Error for {symbol}: {mae}")

    # Save model and scaler
    joblib.dump(model, f"./models/model_{symbol}.pkl")
    joblib.dump(scaler, f"./models/scaler_{symbol}.pkl")

    # Store model in dictionary
    models[symbol] = model

print("\n✅ All models trained and saved!")


def predict_corrections(symbol, data):
    """Predicts corrected indicators for a given symbol's trade data."""
    model_file = f"model_{symbol}.pkl"
    scaler_file = f"scaler_{symbol}.pkl"

    if not (os.path.exists(model_file) and os.path.exists(scaler_file)):
        print(f"❌ No trained model found for {symbol}. Skipping...")
        return None

    # Load model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Scale input data
    data_scaled = scaler.transform(data)

    # Predict corrected values
    y_pred = model.predict(data_scaled)

    return pd.DataFrame(y_pred, columns=target)


# Example Usage
symbol_to_predict = "BTCUSDT"  # Change as needed
sample_data = df[df["symbol"] == symbol_to_predict].sample(5)[features]
corrections = predict_corrections(symbol_to_predict, sample_data)

print("\n📌 Suggested Corrections:")
print(corrections)