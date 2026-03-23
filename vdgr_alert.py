import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# -----------------------
# SETTINGS
# -----------------------

BOT_TOKEN = "bot8659012048:AAEmT5pTsl6C1qLJnDUWYTFH-HtD7XnVAf8"
CHAT_ID = "7433669578"

VDGR_TICKER = "VDGR.AX"
VIX_TICKER = "^VIX"

# -----------------------
# TELEGRAM
# -----------------------

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": message})

# -----------------------
# DOWNLOAD DATA
# -----------------------

print("Downloading market data...")

vdgr = yf.download(VDGR_TICKER, period="2y", auto_adjust=False)
vix = yf.download(VIX_TICKER, period="2y", auto_adjust=False)

# Flatten columns in case yfinance returns MultiIndex
if isinstance(vdgr.columns, pd.MultiIndex):
    vdgr.columns = vdgr.columns.get_level_values(0)

if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# Keep only needed columns
vdgr = vdgr[['Close']].copy()
vix = vix[['Close']].copy()

# Rename VIX close column
vix.rename(columns={"Close": "VIX"}, inplace=True)

# Join datasets
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
# ALERT LOGIC
# -----------------------

data['Alert'] = None
data.loc[(data['RSI'] < 50) & (data['VIX'] > 18), 'Alert'] = "LOW"
data.loc[(data['RSI'] < 45) & (data['VIX'] > 20), 'Alert'] = "MEDIUM"
data.loc[(data['RSI'] < 35) & (data['VIX'] > 25), 'Alert'] = "HIGH"

# -----------------------
# GET LATEST VALUES
# -----------------------

latest = data.iloc[-1]

price = latest.at['Close']
rsi_val = latest.at['RSI']
vix_val = latest.at['VIX']
alert = latest.at['Alert']

# Force to plain Python numbers
price = float(price) if pd.notna(price) else None
rsi_val = float(rsi_val) if pd.notna(rsi_val) else None
vix_val = float(vix_val) if pd.notna(vix_val) else None

# -----------------------
# SAVE TO CSV
# -----------------------

new_row = pd.DataFrame([{
    "Date": datetime.today().strftime("%Y-%m-%d"),
    "Price": price,
    "RSI": rsi_val,
    "VIX": vix_val,
    "Alert": alert if pd.notna(alert) else ""
}])

try:
    existing = pd.read_csv("data.csv")
    updated = pd.concat([existing, new_row], ignore_index=True)
except Exception:
    updated = new_row

updated.to_csv("data.csv", index=False)

# -----------------------
# TELEGRAM ALERT
# -----------------------

if pd.notna(alert) and alert != "":
    message = (
        f"VDGR Alert: {alert}\n"
        f"Price: {price:.2f}\n"
        f"RSI: {rsi_val:.2f}\n"
        f"VIX: {vix_val:.2f}"
    )
    send_telegram(message)

print("Done.")