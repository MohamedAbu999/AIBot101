import websocket
import json
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from threading import Lock

# Replace with your API token and APP ID
API_TOKEN = "your_api_token"
APP_ID = "your_app_id"

# WebSocket URL
DERIV_WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

# Deriv-supported markets
SUPPORTED_MARKETS = ["R_10", "R_25", "R_50"]

# Initialize variables
websocket_connection = None
market_data = {}
candles = {}
selected_market = None
last_trade_time = None
rf_model = None
lock = Lock()  # Thread-safe lock for shared variables

# Cooldown and recovery parameters
COOLDOWN_PERIOD = timedelta(seconds=30)
RECOVERY_PAUSE = timedelta(minutes=3)
last_loss_time = None
consecutive_losses = 0

# Trade frequency control
TRADE_LIMIT_PER_MINUTE = 3
trade_timestamps = []

# Profit lock parameters
PROFIT_LOCK_THRESHOLD = 2.0
is_profit_locked = False

# Risk and compounding parameters
initial_balance = 200
stake = initial_balance * 0.01  # 1% risk
max_stake = 10
stake_growth = 0.15
daily_loss_limit = initial_balance * 0.05
daily_profit_target = initial_balance * 0.2
current_balance = initial_balance
current_profit = 0
HISTORICAL_CANDLES = 50  # Number of candles to keep for analysis
MODEL_TRAIN_INTERVAL = timedelta(minutes=5)
last_model_train_time = None

# Trend filter (RSI)
RSI_PERIOD = 14
RSI_THRESHOLD = 70

# Debug mode for detailed logs
DEBUG_MODE = True

# Trade log
trade_log = []

# Retry logic for WebSocket
def reconnect_with_backoff():
    global websocket_connection
    backoff = 1  # Start with 1 second
    max_backoff = 30  # Limit backoff to 30 seconds

    while True:
        try:
            print(f"Reconnecting in {backoff} seconds...")
            time.sleep(backoff)
            websocket_connection = websocket.WebSocketApp(
                DERIV_WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )
            websocket_connection.run_forever()
            break
        except Exception as e:
            print(f"Reconnection failed: {e}")
            backoff = min(backoff * 2, max_backoff)  # Exponential backoff

# Data preprocessing
def scale_data(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

# Calculate RSI
def calculate_rsi(prices):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

# Machine Learning with Scikit-learn
def train_random_forest(X, y):
    X = scale_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)}")
    return model

# Log trade details
def log_trade(profit, result, reason):
    global trade_log, current_balance, consecutive_losses

    time_of_trade = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trade_log.append({
        "time": time_of_trade,
        "profit": profit,
        "balance": current_balance,
        "result": result,
        "reason": reason,
        "win_streak": consecutive_losses if result == "Win" else 0,
        "loss_streak": consecutive_losses if result == "Loss" else 0
    })

    print(f"Trade Log: {trade_log[-1]}")

# Handle tick data and perform trade analysis
def perform_analysis_and_trade():
    global candles, selected_market, last_trade_time, current_balance, stake, rf_model, last_model_train_time, consecutive_losses, last_loss_time, is_profit_locked, trade_timestamps

    if not selected_market or selected_market not in candles:
        print("No market selected or data unavailable. Skipping analysis.")
        return

    print(f"Analyzing candles for market: {selected_market}...")

    with lock:
        df = pd.DataFrame(candles[selected_market])

    # Ensure we have enough candles
    if len(df) < HISTORICAL_CANDLES:
        print(f"Insufficient candle data for market: {selected_market}. Waiting for more data...")
        return

    # Check profit lock
    if is_profit_locked:
        print("Profit lock active. Skipping trades.")
        return

    # Recovery pause
    if last_loss_time and datetime.now() - last_loss_time < RECOVERY_PAUSE:
        print("Recovery pause active. Skipping trades.")
        return

    # Add hybrid confirmation logic
    if len(df) >= 2:
        prev_candle = df.iloc[-2]
        current_candle = df.iloc[-1]
        if not (prev_candle["close"] > prev_candle["open"] or current_candle["close"] > current_candle["open"] * 1.02):
            if DEBUG_MODE:
                print(f"[DEBUG] Hybrid Confirmation - Previous Candle: {prev_candle}, Current Candle: {current_candle}")
            print("Hybrid confirmation failed. Skipping trade.")
            return

    # Check for momentum/volatility
    recent_volatility = df["high"].iloc[-5:].max() - df["low"].iloc[-5:].min()
    momentum_threshold = 0.02  # Adjusted for strong bullish trends
    if recent_volatility < momentum_threshold:
        if DEBUG_MODE:
            print(f"[DEBUG] Momentum Check - Recent Volatility: {recent_volatility}, Threshold: {momentum_threshold}")
            print(f"[DEBUG] Live Candle Data:\n{df.tail()}")
        print("Momentum/volatility check failed. Skipping trade.")
        return

    # Check RSI trend filter
    df["RSI"] = calculate_rsi(df["close"])
    if df["RSI"].iloc[-1] > RSI_THRESHOLD:
        print(f"RSI exceeds threshold ({RSI_THRESHOLD}). Skipping trade.")
        return

    # Check trade frequency control
    current_time = datetime.now()
    trade_timestamps = [ts for ts in trade_timestamps if current_time - ts < timedelta(minutes=1)]
    if len(trade_timestamps) >= TRADE_LIMIT_PER_MINUTE:
        print("Trade frequency limit reached. Skipping trade.")
        return

    # Prepare features and labels for ML models
    df["price_diff"] = df["close"].diff()
    df["high_low_diff"] = df["high"] - df["low"]
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)  # 1 if next close > current close, else 0
    df.dropna(inplace=True)

    X = df[["price_diff", "high_low_diff"]]
    y = df["target"]

    # Train the model if it's time
    if rf_model is None or (last_model_train_time is None or datetime.now() - last_model_train_time >= MODEL_TRAIN_INTERVAL):
        print("Training Random Forest model...")
        rf_model = train_random_forest(X, y)
        print("Random Forest model training complete.")
        last_model_train_time = datetime.now()

    # Make predictions
    latest_features = X.iloc[-1:].copy()
    print(f"Making prediction based on latest candle data for {selected_market}...")
    rf_prediction = rf_model.predict(latest_features)
    print(f"Model prediction: {'BUY signal' if rf_prediction[0] == 1 else 'NO trade signal'}.")

    # Decision logic: trade if the model predicts a "buy" signal
    if rf_prediction[0] == 1:
        if last_trade_time is None or datetime.now() - last_trade_time > COOLDOWN_PERIOD:
            if current_profit >= daily_profit_target:
                print("Daily profit target reached. Halting trades.")
                return
            if current_balance <= initial_balance - daily_loss_limit:
                print("Daily loss limit reached. Halting trades.")
                return
            print("Trade signal detected. Executing trade...")
            execute_trade(reason="Model prediction triggered")
            last_trade_time = datetime.now()
            trade_timestamps.append(last_trade_time)
        else:
            print("Cooldown active. Waiting before executing the next trade.")
    else:
        print("No trade signal detected. Bot remains idle.")

# Execute a trade
def execute_trade(reason):
    global stake, current_balance, consecutive_losses, last_loss_time, is_profit_locked, current_profit

    # Simulate trade outcome
    trade_outcome = np.random.choice(["Win", "Loss"], p=[0.6, 0.4])  # 60% win rate
    profit = stake * stake_growth if trade_outcome == "Win" else -stake

    # Update balance and profit
    current_balance += profit
    current_profit += profit

    if trade_outcome == "Win":
        consecutive_losses = 0
    else:
        consecutive_losses += 1
        last_loss_time = datetime.now()

    # Check profit lock
    if current_profit >= PROFIT_LOCK_THRESHOLD:
        is_profit_locked = True
        print("Profit lock triggered. Pausing trades.")

    log_trade(profit, trade_outcome, reason)

    print(f"Trade result: {trade_outcome}. Profit: {profit}. Current balance: {current_balance}.")

def handle_ticks(tick):
    global market_data, candles

    symbol = tick["symbol"]
    with lock:
        if symbol not in candles:
            candles[symbol] = []

        # Aggregate ticks into 1-minute candles
        current_time = datetime.utcfromtimestamp(tick["epoch"])
        current_minute = current_time.replace(second=0, microsecond=0)

        if len(candles[symbol]) == 0 or candles[symbol][-1]["time"] != current_minute:
            candles[symbol].append(
                {"time": current_minute, "open": tick["quote"], "high": tick["quote"], "low": tick["quote"], "close": tick["quote"]}
            )
        else:
            candle = candles[symbol][-1]
            candle["high"] = max(candle["high"], tick["quote"])
            candle["low"] = min(candle["low"], tick["quote"])
            candle["close"] = tick["quote"]

        # Limit candle history
        if len(candles[symbol]) > HISTORICAL_CANDLES:
            candles[symbol].pop(0)

def on_message(ws, message):
    global selected_market
    data = json.loads(message)

    # Handle authorization
    if "authorize" in data:
        print("Authorization successful.")
        fetch_markets(ws)

    # Handle tick data
    if "tick" in data:
        handle_ticks(data["tick"])
        perform_analysis_and_trade()

    # Handle active symbols
    if "active_symbols" in data:
        symbols = data["active_symbols"]
        selected_market = next(
            (symbol["symbol"] for symbol in symbols if symbol["symbol"] in SUPPORTED_MARKETS),
            None,
        )
        if selected_market:
            print(f"Selected Market: {selected_market}")
            subscribe_to_market_data(ws, selected_market)

def on_error(ws, error):
    print(f"WebSocket Error: {error}")
    reconnect_with_backoff()

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")
    reconnect_with_backoff()

def on_open(ws):
    print("WebSocket Connection Opened")
    authorize(ws)

def authorize(ws):
    auth_request = {"authorize": API_TOKEN}
    ws.send(json.dumps(auth_request))  # Fixed: Completed the function

def fetch_markets(ws):
    market_request = {"active_symbols": "brief", "product_type": "basic"}
    ws.send(json.dumps(market_request))
    print("Fetching Available Markets...")

def subscribe_to_market_data(ws, market):
    market_request = {"ticks": market, "subscribe": 1}
    ws.send(json.dumps(market_request))
    print(f"Subscribed to Market Data for {market}")

if __name__ == "__main__":
    websocket_connection = websocket.WebSocketApp(
        DERIV_WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
    )
    websocket_connection.run_forever()  # Added: WebSocket initialization