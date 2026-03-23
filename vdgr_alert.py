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

# -----------------------
# HELPERS
# -----------------------

def send_telegram(message: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("BOT_TOKEN and CHAT_ID must be set as environment variables or GitHub Secrets.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.get(url, params={"chat_id": CHAT_ID, "text": message})

    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)

    response.raise_for_status()


def suggested_investment(signal: str) -> int:
    if signal == "LOW":
        return 400
    if signal == "MEDIUM":
        return 800
    if signal == "HIGH":
        return 1600
    return 0


def build_reason(signal: str, rsi: float, vix: float, drawdown: float) -> str:
    reasons = []

    if signal == "HIGH":
        reasons.append(f"RSI {rsi:.1f} < 35")
        reasons.append(f"VIX {vix:.1f} > 25")
    elif signal == "MEDIUM":
        reasons.append(f"RSI {rsi:.1f} < 45")
        reasons.append(f"VIX {vix:.1f} > 20")
    elif signal == "LOW":
        reasons.append(f"RSI {rsi:.1f} < 50")
        reasons.append(f"VIX {vix:.1f} > 18")
    else:
        reasons.append(f"No signal thresholds met (RSI={rsi:.1f}, VIX={vix:.1f})")

    reasons.append(f"Drawdown context: {drawdown:.1f}%")

    return " | ".join(reasons)


# -----------------------
# LOAD DATA
# -----------------------

print("Downloading market data...")

vdgr = yf.download(VDGR_TICKER, period="12mo", auto_adjust=False)
vix = yf.download(VIX_TICKER, period="12mo", auto_adjust=False)

# Flatten MultiIndex columns if returned by yfinance
if isinstance(vdgr.columns, pd.MultiIndex):
    vdgr.columns = vdgr.columns.get_level_values(0)

if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

vdgr = vdgr[['Close']].copy()
vix = vix[['Close']].copy()
vix.rename(columns={"Close": "VIX"}, inplace=True)

data = vdgr.join(vix, how="inner")

# -----------------------
# CALCULATE RSI
# -----------------------

delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# -----------------------
# CALCULATE DRAWDOWN (CONTEXT ONLY)
# -----------------------

data['6M_High'] = data['Close'].rolling(126).max()
data['DrawdownPct'] = ((data['Close'] - data['6M_High']) / data['6M_High']) * 100

# -----------------------
# SIGNAL LOGIC
# RSI + VIX ONLY
# -----------------------

data['Signal'] = "NONE"
data.loc[(data['RSI'] < 50) & (data['VIX'] > 18), 'Signal'] = "LOW"
data.loc[(data['RSI'] < 45) & (data['VIX'] > 20), 'Signal'] = "MEDIUM"
data.loc[(data['RSI'] < 35) & (data['VIX'] > 25), 'Signal'] = "HIGH"

data = data.dropna(subset=['Close', 'RSI', 'VIX']).copy()

# -----------------------
# LATEST VALUES
# -----------------------

latest = data.iloc[-1]

price = float(latest['Close'])
rsi = float(latest['RSI'])
vix_val = float(latest['VIX'])
drawdown = float(latest['DrawdownPct']) if pd.notna(latest['DrawdownPct']) else 0.0
signal = str(latest['Signal'])
investment = suggested_investment(signal)
reason = build_reason(signal, rsi, vix_val, drawdown)
date_str = data.index[-1].strftime("%Y-%m-%d")

# -----------------------
# TELEGRAM MESSAGE
# -----------------------

message = (
    f"VDGR Daily Update\n\n"
    f"Date: {date_str}\n"
    f"Signal: {signal}\n"
    f"Price: ${price:.2f}\n"
    f"RSI: {rsi:.2f}\n"
    f"VIX: {vix_val:.2f}\n"
    f"Drawdown: {drawdown:.2f}%\n"
    f"Suggested Investment: ${investment}\n\n"
    f"Why: {reason}"
)

send_telegram(message)

print("Done.")