import os
import requests
import yfinance as yf
import pandas as pd

# -----------------------
# CONFIG
# -----------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

VDGR_TICKER = "VDGR.AX"
VIX_TICKER = "^VIX"

INVESTMENT_MAP = {
    "LOW": 0,
    "MEDIUM": 200,
    "HIGH": 400,
    "NONE": 0,
}

# -----------------------
# TELEGRAM
# -----------------------

def send_telegram(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("BOT_TOKEN and CHAT_ID must be set.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.get(
        url,
        params={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=30,
    )

    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)

    response.raise_for_status()

# -----------------------
# HELPERS
# -----------------------

def investment(signal):
    return INVESTMENT_MAP.get(str(signal).upper(), 0)


def build_reason(signal, rsi, vix):
    if signal == "HIGH":
        return f"RSI ({rsi:.1f}) < 35 and VIX ({vix:.1f}) > 25"
    elif signal == "MEDIUM":
        return f"RSI ({rsi:.1f}) < 45 and VIX ({vix:.1f}) > 20"
    elif signal == "LOW":
        return f"RSI ({rsi:.1f}) < 50 and VIX ({vix:.1f}) > 18"
    else:
        return f"No thresholds met (RSI {rsi:.1f}, VIX {vix:.1f})"


def count_extreme_signals(data, years):
    df = data.copy()

    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    df = df[df.index >= start_date].copy()

    df["EXTREME"] = (
        (df["RSI"] < 30) &
        (df["VIX"] > 30) &
        (df["DrawdownPct"] < -10)
    )

    return int(df["EXTREME"].sum())

# -----------------------
# LOAD DATA
# -----------------------

print("Downloading market data...")

# Need at least 3 years if you want 1y, 2y and 3y counts
vdgr = yf.download(VDGR_TICKER, period="3y", auto_adjust=False, progress=False)
vix = yf.download(VIX_TICKER, period="3y", auto_adjust=False, progress=False)

if vdgr.empty:
    raise ValueError("No VDGR data downloaded.")
if vix.empty:
    raise ValueError("No VIX data downloaded.")

if isinstance(vdgr.columns, pd.MultiIndex):
    vdgr.columns = vdgr.columns.get_level_values(0)

if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

vdgr = vdgr[["Close"]].copy()
vix = vix[["Close"]].copy()
vix.rename(columns={"Close": "VIX"}, inplace=True)

data = vdgr.join(vix, how="inner")

if data.empty:
    raise ValueError("Joined VDGR/VIX dataset is empty.")

# -----------------------
# RSI
# -----------------------

delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# -----------------------
# DRAWDOWN (CONTEXT ONLY)
# -----------------------

data["6M_High"] = data["Close"].rolling(126).max()
data["DrawdownPct"] = ((data["Close"] - data["6M_High"]) / data["6M_High"]) * 100

# -----------------------
# SIGNAL LOGIC (RSI + VIX ONLY)
# -----------------------

data["Signal"] = "NONE"
data.loc[(data["RSI"] < 50) & (data["VIX"] > 18), "Signal"] = "LOW"
data.loc[(data["RSI"] < 45) & (data["VIX"] > 20), "Signal"] = "MEDIUM"
data.loc[(data["RSI"] < 35) & (data["VIX"] > 25), "Signal"] = "HIGH"

data["Investment"] = data["Signal"].apply(investment)
data = data.dropna(subset=["Close", "RSI", "VIX", "DrawdownPct"]).copy()

if data.empty:
    raise ValueError("No usable rows after indicator calculations.")

# -----------------------
# EXTREME COUNTS
# -----------------------

print("\n--- EXTREME SIGNAL COUNTS ---")
for y in [1, 2, 3]:
    print(f"{y} year EXTREME signals: {count_extreme_signals(data, y)}")

# -----------------------
# LATEST VALUES
# -----------------------

latest = data.iloc[-1]

price = float(latest["Close"])
rsi = float(latest["RSI"])
vix_val = float(latest["VIX"])
drawdown = float(latest["DrawdownPct"])
signal = latest["Signal"]
investment_amt = investment(signal)
date_str = data.index[-1].strftime("%d-%m-%Y")

reason = build_reason(signal, rsi, vix_val)

# -----------------------
# MESSAGE
# -----------------------

message = (
    f"VDGR Daily Update\n\n"
    f"Date: {date_str}\n"
    f"Signal: {signal}\n"
    f"Price: ${price:.2f}\n"
    f"RSI: {rsi:.2f}\n"
    f"VIX: {vix_val:.2f}\n"
    f"Drawdown: {drawdown:.2f}%\n"
    f"Suggested Investment: ${investment_amt}\n\n"
    f"Why: {reason}\n\n"
    f"Signal amounts:\n"
    f"- LOW = ${INVESTMENT_MAP['LOW']}\n"
    f"- MEDIUM = ${INVESTMENT_MAP['MEDIUM']}\n"
    f"- HIGH = ${INVESTMENT_MAP['HIGH']}\n\n"
    f"Note: Drawdown is context only and not used in signal logic."
)

# -----------------------
# SEND ALERT
# -----------------------

# Uncomment this only when you want Telegram enabled
# send_telegram(message)

print("Done.")