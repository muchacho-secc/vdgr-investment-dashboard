import os
import json

import pandas as pd
import requests
import yfinance as yf

from signals import (
    append_live_trade_if_needed,
    build_alert_message,
    prepare_signal_data,
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


def send_telegram(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("BOT_TOKEN and CHAT_ID must be set.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.get(
        url,
        params={"chat_id": CHAT_ID, "text": message},
        timeout=30,
    )
    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)
    response.raise_for_status()


def load_data():
    vdgr = yf.download("VDGR.AX", period="3y", auto_adjust=False, progress=False)
    vix = yf.download("^VIX", period="3y", auto_adjust=False, progress=False)

    if vdgr.empty:
        raise ValueError("No VDGR data downloaded.")
    if vix.empty:
        raise ValueError("No VIX data downloaded.")

    if isinstance(vdgr.columns, pd.MultiIndex):
        vdgr.columns = vdgr.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vdgr = vdgr[["Close"]].copy()
    vix = vix[["Close"]].copy().rename(columns={"Close": "VIX"})

    data = vdgr.join(vix, how="inner")
    if data.empty:
        raise ValueError("Joined VDGR/VIX dataset is empty.")

    return prepare_signal_data(data)


def main():
    print("Downloading market data...")
    data = load_data()

    latest = data.iloc[-1]
    market_date = pd.to_datetime(data.index[-1])

    recorded, record_message = append_live_trade_if_needed(data)
    print(record_message)

    message = build_alert_message(latest, market_date)
    if message is None:
        print("No Telegram alert today because the signal is not MEDIUM, HIGH, or EXTREME.")
        return

    send_telegram(message)
    print("Done.")


if __name__ == "__main__":
    main()