import pandas as pd
from pathlib import Path
from datetime import datetime

LAUNCH_DATE = pd.Timestamp("2026-03-27")
LEDGER_FILE = "live_signal_ledger.csv"

THRESHOLDS = {
    "LOW": {"rsi": 50, "vix": 18},
    "MEDIUM": {"rsi": 45, "vix": 20},
    "HIGH": {"rsi": 35, "vix": 25},
    "EXTREME": {"rsi": 30, "vix": 30, "drawdown": -10},
}

INVESTMENT_MAP = {
    "NONE": 0,
    "LOW": 0,
    "MEDIUM": 200,
    "HIGH": 400,
    "EXTREME": 800,
}

DISPLAY_LABEL_MAP = {
    "NONE": "NONE",
    "LOW": "WATCH",
    "MEDIUM": "MEDIUM",
    "HIGH": "HIGH",
    "EXTREME": "EXTREME",
}

CONFIDENCE_MAP = {
    "NONE": "No signal",
    "LOW": "Early weakness",
    "MEDIUM": "Moderate opportunity",
    "HIGH": "Strong opportunity",
    "EXTREME": "Rare opportunity",
}

ALERT_SIGNALS = {"MEDIUM", "HIGH", "EXTREME"}

def filter_months(data: pd.DataFrame, months: int) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    end_date = pd.to_datetime(data.index.max())
    start_date = end_date - pd.DateOffset(months=months)
    return data[data.index >= start_date].copy()

def format_display_signal(signal: str) -> str:
    return DISPLAY_LABEL_MAP.get(str(signal).upper(), str(signal).upper())

def investment(signal: str) -> int:
    return int(INVESTMENT_MAP.get(str(signal).upper(), 0))

def confidence(signal: str) -> str:
    return CONFIDENCE_MAP.get(str(signal).upper(), "No signal")

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["6M_High"] = df["Close"].rolling(126).max()
    df["DrawdownPct"] = ((df["Close"] - df["6M_High"]) / df["6M_High"]) * 100
    return df

def classify_signal(row: pd.Series) -> str:
    rsi = float(row["RSI"])
    vix = float(row["VIX"])
    drawdown = float(row["DrawdownPct"])
    signal = "NONE"
    if (rsi < THRESHOLDS["LOW"]["rsi"]) and (vix > THRESHOLDS["LOW"]["vix"]):
        signal = "LOW"
    if (rsi < THRESHOLDS["MEDIUM"]["rsi"]) and (vix > THRESHOLDS["MEDIUM"]["vix"]):
        signal = "MEDIUM"
    if (rsi < THRESHOLDS["HIGH"]["rsi"]) and (vix > THRESHOLDS["HIGH"]["vix"]):
        signal = "HIGH"
    if (
        (rsi < THRESHOLDS["EXTREME"]["rsi"])
        and (vix > THRESHOLDS["EXTREME"]["vix"])
        and (drawdown < THRESHOLDS["EXTREME"]["drawdown"])
    ):
        signal = "EXTREME"
    return signal

def apply_signals(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Signal"] = df.apply(classify_signal, axis=1)
    df["Investment"] = df["Signal"].apply(investment)
    df["Confidence"] = df["Signal"].apply(confidence)
    df["DisplaySignal"] = df["Signal"].apply(format_display_signal)
    return df

def prepare_signal_data(data: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(data)
    df = df.dropna(subset=["Close", "RSI", "VIX", "DrawdownPct"]).copy()
    df = apply_signals(df)
    df.index = pd.to_datetime(df.index)
    return df

def add_forward_returns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["FwdReturn_5d"] = (df["Close"].shift(-5) / df["Close"] - 1) * 100
    df["FwdReturn_20d"] = (df["Close"].shift(-20) / df["Close"] - 1) * 100
    df["FwdReturn_60d"] = (df["Close"].shift(-60) / df["Close"] - 1) * 100
    return df

def forward_return_summary(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for signal in ["MEDIUM", "HIGH", "EXTREME"]:
        subset = data[data["Signal"] == signal].copy()
        if subset.empty:
            continue
        row = {"Signal": format_display_signal(signal), "Count": int(len(subset))}
        for horizon, col in [(5, "FwdReturn_5d"), (20, "FwdReturn_20d"), (60, "FwdReturn_60d")]:
            valid = subset.dropna(subset=[col])
            row[f"Avg {horizon}d %"] = valid[col].mean() if not valid.empty else None
            row[f"Median {horizon}d %"] = valid[col].median() if not valid.empty else None
            row[f"Win Rate {horizon}d %"] = (valid[col] > 0).mean() * 100 if not valid.empty else None
        records.append(row)
    return pd.DataFrame(records)

def _next_tier_comment(signal: str, rsi: float, vix: float, drawdown: float) -> str:
    if signal == "MEDIUM":
        return (
            f"HIGH has not triggered yet because RSI is still {max(rsi - 35, 0):.1f} points above 35 "
            f"or VIX is {max(25 - vix, 0):.1f} points below 25."
        )
    if signal == "HIGH":
        return (
            f"EXTREME has not triggered because RSI is {max(rsi - 30, 0):.1f} points above 30, "
            f"or VIX is {max(30 - vix, 0):.1f} points below 30, "
            f"or drawdown is {max(-10 - drawdown, 0):.1f}% short of -10%."
        )
    return ""

def detailed_reason(row: pd.Series) -> str:
    signal = str(row["Signal"]).upper()
    rsi = float(row["RSI"])
    vix = float(row["VIX"])
    drawdown = float(row["DrawdownPct"])
    if signal == "LOW":
        return (
            f"VDGR is showing early weakness rather than a buy signal. RSI is {rsi:.1f}, "
            f"which is below 50, and VIX is {vix:.1f}, which is above 18."
        )
    if signal == "MEDIUM":
        return (
            f"VDGR momentum is weak enough to become actionable. RSI is {rsi:.1f} and VIX is {vix:.1f}, "
            f"so this qualifies as a moderate opportunity. {_next_tier_comment(signal, rsi, vix, drawdown)}"
        )
    if signal == "HIGH":
        return (
            f"Market stress is elevated and VDGR momentum is materially weaker. RSI is {rsi:.1f} and VIX is {vix:.1f}, "
            f"so this qualifies as a strong opportunity. {_next_tier_comment(signal, rsi, vix, drawdown)}"
        )
    if signal == "EXTREME":
        return (
            f"This is a rare opportunity. RSI is {rsi:.1f}, VIX is {vix:.1f}, and drawdown is {drawdown:.1f}%, "
            f"which means both fear and asset-level weakness are severe enough to trigger the top tier."
        )
    return f"No buy signal today. RSI is {rsi:.1f} and VIX is {vix:.1f}."

def build_alert_message(row: pd.Series, market_date: pd.Timestamp) -> str | None:
    signal = str(row["Signal"]).upper()
    if signal not in ALERT_SIGNALS:
        return None
    return (
        f"ACTION: Buy ${investment(signal)} — {confidence(signal)}\n\n"
        f"Latest market date: {pd.to_datetime(market_date).strftime('%d-%m-%Y')}\n"
        f"Signal: {format_display_signal(signal)}\n"
        f"VDGR price: ${float(row['Close']):.2f}\n"
        f"RSI: {float(row['RSI']):.2f}\n"
        f"VIX: {float(row['VIX']):.2f}\n"
        f"Drawdown: {float(row['DrawdownPct']):.2f}%\n"
        f"Confidence: {confidence(signal)}\n\n"
        f"Why this fired:\n{detailed_reason(row)}"
    )

LEDGER_COLUMNS = [
    "TradeDate",
    "Signal",
    "DisplaySignal",
    "Confidence",
    "BuyPrice",
    "AmountInvested",
    "UnitsBought",
    "LatestMarketDate",
    "RecordedAt",
]

def ensure_ledger_exists(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    if not path.exists():
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(path, index=False)
    return path

def load_live_ledger(csv_path: str | Path) -> pd.DataFrame:
    path = ensure_ledger_exists(csv_path)
    ledger = pd.read_csv(path)
    if ledger.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    for col in ["TradeDate", "LatestMarketDate", "RecordedAt"]:
        if col in ledger.columns:
            ledger[col] = pd.to_datetime(ledger[col], errors="coerce")
    return ledger

def append_live_trade_if_needed(data: pd.DataFrame, csv_path: str | Path, launch_date: pd.Timestamp = LAUNCH_DATE):
    if data.empty:
        return False, "No data available."
    ledger = load_live_ledger(csv_path)
    latest = data.iloc[-1]
    market_date = pd.to_datetime(data.index[-1]).normalize()
    signal = str(latest["Signal"]).upper()

    if market_date < pd.to_datetime(launch_date):
        return False, f"Launch date not reached. Latest market date is {market_date.strftime('%d-%m-%Y')}."
    if signal not in ALERT_SIGNALS:
        return False, f"No live trade recorded because signal is {format_display_signal(signal)}."
    if not ledger.empty and market_date in set(ledger["TradeDate"].dropna().dt.normalize()):
        return False, f"Trade for {market_date.strftime('%d-%m-%Y')} already exists."

    buy_price = float(latest["Close"])
    amount_invested = float(investment(signal))
    units_bought = amount_invested / buy_price if buy_price > 0 else 0.0

    new_row = pd.DataFrame([{
        "TradeDate": market_date.strftime("%Y-%m-%d"),
        "Signal": signal,
        "DisplaySignal": format_display_signal(signal),
        "Confidence": confidence(signal),
        "BuyPrice": round(buy_price, 6),
        "AmountInvested": round(amount_invested, 2),
        "UnitsBought": round(units_bought, 8),
        "LatestMarketDate": market_date.strftime("%Y-%m-%d"),
        "RecordedAt": datetime.utcnow().isoformat(),
    }])

    path = ensure_ledger_exists(csv_path)
    new_row.to_csv(path, mode="a", index=False, header=path.stat().st_size == 0)
    return True, f"Recorded {format_display_signal(signal)} trade for {market_date.strftime('%d-%m-%Y')}."

def live_performance_ledger(data: pd.DataFrame, csv_path: str | Path) -> pd.DataFrame:
    ledger = load_live_ledger(csv_path)
    if ledger.empty or data.empty:
        return pd.DataFrame()
    current_price = float(data["Close"].iloc[-1])
    latest_market_date = pd.to_datetime(data.index[-1])

    out = ledger.copy()
    out["CurrentPrice"] = current_price
    out["CurrentValue"] = out["UnitsBought"] * current_price
    out["ProfitLoss"] = out["CurrentValue"] - out["AmountInvested"]
    out["ReturnPct"] = ((out["CurrentValue"] / out["AmountInvested"]) - 1) * 100
    out["CurrentMarketDate"] = latest_market_date
    return out.sort_values("TradeDate", ascending=False).reset_index(drop=True)

def live_summary(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {
            "total_invested": 0.0,
            "current_value": 0.0,
            "profit_loss": 0.0,
            "return_pct": 0.0,
            "total_units": 0.0,
            "num_buys": 0,
        }
    total_invested = float(ledger["AmountInvested"].sum())
    current_value = float(ledger["CurrentValue"].sum())
    profit_loss = current_value - total_invested
    total_units = float(ledger["UnitsBought"].sum())
    return_pct = ((current_value / total_invested) - 1) * 100 if total_invested > 0 else 0.0
    return {
        "total_invested": total_invested,
        "current_value": current_value,
        "profit_loss": profit_loss,
        "return_pct": return_pct,
        "total_units": total_units,
        "num_buys": int(len(ledger)),
    }

def live_breakdown_by_signal(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    grouped = (
        ledger.groupby("DisplaySignal", as_index=False)
        .agg(
            Buys=("DisplaySignal", "count"),
            AmountInvested=("AmountInvested", "sum"),
            UnitsBought=("UnitsBought", "sum"),
            CurrentValue=("CurrentValue", "sum"),
            ProfitLoss=("ProfitLoss", "sum"),
        )
    )
    grouped["ReturnPct"] = ((grouped["CurrentValue"] / grouped["AmountInvested"]) - 1) * 100
    return grouped.sort_values("AmountInvested", ascending=False).reset_index(drop=True)
