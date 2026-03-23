import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta

# -----------------------
# PAGE SETUP
# -----------------------

st.set_page_config(page_title="VDGR Investment Dashboard", layout="wide")
st.title("VDGR Investment Dashboard")

# -----------------------
# SETTINGS
# -----------------------

FORTNIGHTLY_BASE = 800

COLORS = {
    "price": "#1f77b4",
    "low": "#4dabf7",       # blue
    "medium": "#f59f00",    # orange
    "high": "#e03131",      # red
    "none": "#adb5bd",      # grey
    "rsi": "#2b8a3e",       # green
    "vix": "#6f42c1",       # purple
    "drawdown": "#495057",  # dark grey
}

# -----------------------
# DATA LOADING
# -----------------------

@st.cache_data(ttl=3600)
def load_data():
    vdgr = yf.download("VDGR.AX", period="12mo", auto_adjust=False)
    vix = yf.download("^VIX", period="12mo", auto_adjust=False)

    if isinstance(vdgr.columns, pd.MultiIndex):
        vdgr.columns = vdgr.columns.get_level_values(0)

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vdgr = vdgr[['Close']].copy()
    vix = vix[['Close']].copy()
    vix.rename(columns={"Close": "VIX"}, inplace=True)

    data = vdgr.join(vix, how="inner")

    # RSI
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # 6-month drawdown
    data['6M_High'] = data['Close'].rolling(126).max()
    data['DrawdownPct'] = ((data['Close'] - data['6M_High']) / data['6M_High']) * 100

    # Signal logic
    data['Signal'] = "NONE"

    low_condition = (data['DrawdownPct'] <= -5) & (data['RSI'] <= 45)
    med_condition = (data['DrawdownPct'] <= -8) & (data['RSI'] <= 40) & (data['VIX'] >= 22)
    high_condition = (data['DrawdownPct'] <= -12) & (data['RSI'] <= 35) & (data['VIX'] >= 28)

    data.loc[low_condition, 'Signal'] = "LOW"
    data.loc[med_condition, 'Signal'] = "MEDIUM"
    data.loc[high_condition, 'Signal'] = "HIGH"

    def suggested_investment(signal):
        if signal == "LOW":
            return 400
        elif signal == "MEDIUM":
            return 800
        elif signal == "HIGH":
            return 1600
        return 0

    data['Suggested Investment'] = data['Signal'].apply(suggested_investment)

    data = data.dropna(subset=['RSI', 'VIX', 'DrawdownPct']).copy()
    return data


def ensure_date_column(df):
    df = df.copy().reset_index()
    if 'Date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    return df


def build_reason(row):
    reasons = []

    if row['DrawdownPct'] <= -12:
        reasons.append(f"Drawdown {row['DrawdownPct']:.1f}% ≤ -12%")
    elif row['DrawdownPct'] <= -8:
        reasons.append(f"Drawdown {row['DrawdownPct']:.1f}% ≤ -8%")
    elif row['DrawdownPct'] <= -5:
        reasons.append(f"Drawdown {row['DrawdownPct']:.1f}% ≤ -5%")

    if row['RSI'] <= 35:
        reasons.append(f"RSI {row['RSI']:.1f} ≤ 35")
    elif row['RSI'] <= 40:
        reasons.append(f"RSI {row['RSI']:.1f} ≤ 40")
    elif row['RSI'] <= 45:
        reasons.append(f"RSI {row['RSI']:.1f} ≤ 45")

    if row['VIX'] >= 28:
        reasons.append(f"VIX {row['VIX']:.1f} ≥ 28")
    elif row['VIX'] >= 22:
        reasons.append(f"VIX {row['VIX']:.1f} ≥ 22")

    if row['Signal'] == "NONE":
        return "No alert today: the LOW / MEDIUM / HIGH thresholds were not all met."

    return " | ".join(reasons)


data = load_data()

if data.empty:
    st.error("No data available.")
    st.stop()

latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest

latest_price = float(latest['Close'])
latest_rsi = float(latest['RSI'])
latest_vix = float(latest['VIX'])
latest_drawdown = float(latest['DrawdownPct'])
latest_signal = latest['Signal']
latest_investment = int(latest['Suggested Investment'])
latest_reason = build_reason(latest)

signal_days = data[data['Signal'] != 'NONE'].copy()

high_days = data[data['Signal'] == 'HIGH']
days_since_high = None if high_days.empty else (data.index[-1] - high_days.index[-1]).days

med_high_days = data[data['Signal'].isin(['MEDIUM', 'HIGH'])]
days_since_med_or_high = None if med_high_days.empty else (data.index[-1] - med_high_days.index[-1]).days

# -----------------------
# TOP CARDS
# -----------------------

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Current Signal", latest_signal)
col2.metric("VDGR Price", f"${latest_price:.2f}")
col3.metric("RSI", f"{latest_rsi:.2f}")
col4.metric("VIX", f"{latest_vix:.2f}")
col5.metric("Drawdown", f"{latest_drawdown:.2f}%")
col6.metric("Suggested Investment", f"${latest_investment}")

# -----------------------
# SUMMARY
# -----------------------

st.subheader("Signal Summary")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Previous Signal", previous['Signal'])
s2.metric("Days Since HIGH", "N/A" if days_since_high is None else str(days_since_high))
s3.metric("Days Since MEDIUM/HIGH", "N/A" if days_since_med_or_high is None else str(days_since_med_or_high))
s4.metric("Last Updated", data.index[-1].strftime("%Y-%m-%d"))

# -----------------------
# WHY THIS SIGNAL
# -----------------------

st.subheader("Why This Signal?")
st.info(latest_reason)

# -----------------------
# RSI / VIX EXPLANATION
# -----------------------

st.subheader("How to Read RSI and VIX")

e1, e2 = st.columns(2)

with e1:
    st.markdown("""
**RSI (Relative Strength Index)** measures price momentum.

- **Above 70** → often considered high / overbought
- **50 to 70** → stronger momentum
- **30 to 50** → softer / weaker momentum
- **Below 30** → often considered low / oversold

This dashboard uses:
- **45 or below** for LOW
- **40 or below** for MEDIUM
- **35 or below** for HIGH
""")

with e2:
    st.markdown("""
**VIX** is a market fear / volatility gauge.

- **Below 20** → calm market
- **20 to 25** → mild fear
- **25 to 30** → elevated fear
- **Above 30** → strong stress / panic

This dashboard uses:
- **22 or above** for MEDIUM
- **28 or above** for HIGH
""")

# -----------------------
# HISTORICAL SIGNAL VISUAL (12 MONTHS)
# -----------------------

st.subheader("Historical Signals Across the Previous 12 Months")

signal_vis = ensure_date_column(data)
signal_vis['Date'] = pd.to_datetime(signal_vis['Date'])

fig0, ax0 = plt.subplots(figsize=(14, 2.8))

signal_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
signal_vis['SignalNum'] = signal_vis['Signal'].map(signal_map)

for signal_name, color in [
    ("NONE", COLORS["none"]),
    ("LOW", COLORS["low"]),
    ("MEDIUM", COLORS["medium"]),
    ("HIGH", COLORS["high"]),
]:
    subset = signal_vis[signal_vis['Signal'] == signal_name]
    ax0.scatter(subset['Date'], subset['SignalNum'], color=color, s=22, label=signal_name)

ax0.set_yticks([0, 1, 2, 3])
ax0.set_yticklabels(["NONE", "LOW", "MEDIUM", "HIGH"])
ax0.set_title("Daily Signal for Each Trading Day")
ax0.set_xlabel("Date")
ax0.set_ylabel("Signal")
ax0.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig0)

# -----------------------
# PRICE CHART WITH SIGNAL MARKERS
# -----------------------

st.subheader("VDGR Price with Signal Markers")

fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(data.index, data['Close'], color=COLORS["price"], linewidth=2, label="VDGR Close")

low_points = data[data['Signal'] == 'LOW']
med_points = data[data['Signal'] == 'MEDIUM']
high_points = data[data['Signal'] == 'HIGH']

ax1.scatter(low_points.index, low_points['Close'], color=COLORS["low"], label="LOW", s=35)
ax1.scatter(med_points.index, med_points['Close'], color=COLORS["medium"], label="MEDIUM", s=50)
ax1.scatter(high_points.index, high_points['Close'], color=COLORS["high"], label="HIGH", s=70)

ax1.set_title("VDGR Price Over the Past 12 Months")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig1)

# -----------------------
# RSI CHART
# -----------------------

st.subheader("RSI Over the Past 12 Months")

fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(data.index, data['RSI'], color=COLORS["rsi"], linewidth=2, label="RSI")
ax2.axhline(70, linestyle="--", color="#868e96", label="Overbought (70)")
ax2.axhline(45, linestyle="--", color=COLORS["low"], label="LOW threshold (45)")
ax2.axhline(40, linestyle="--", color=COLORS["medium"], label="MEDIUM threshold (40)")
ax2.axhline(35, linestyle="--", color=COLORS["high"], label="HIGH threshold (35)")
ax2.axhline(30, linestyle="--", color="#868e96", label="Oversold (30)")
ax2.set_title("RSI with Thresholds")
ax2.set_xlabel("Date")
ax2.set_ylabel("RSI")
ax2.legend()
ax2.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig2)

# -----------------------
# VIX CHART
# -----------------------

st.subheader("VIX Over the Past 12 Months")

fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(data.index, data['VIX'], color=COLORS["vix"], linewidth=2, label="VIX")
ax3.axhline(20, linestyle="--", color="#868e96", label="Calm/Fear split (20)")
ax3.axhline(22, linestyle="--", color=COLORS["medium"], label="MEDIUM threshold (22)")
ax3.axhline(28, linestyle="--", color=COLORS["high"], label="HIGH threshold (28)")
ax3.axhline(30, linestyle="--", color="#495057", label="Stress zone (30)")
ax3.set_title("VIX with Thresholds")
ax3.set_xlabel("Date")
ax3.set_ylabel("VIX")
ax3.legend()
ax3.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig3)

# -----------------------
# DRAWDOWN CHART
# -----------------------

st.subheader("Drawdown from 6-Month High")

fig4, ax4 = plt.subplots(figsize=(14, 4))
ax4.plot(data.index, data['DrawdownPct'], color=COLORS["drawdown"], linewidth=2, label="Drawdown %")
ax4.axhline(-5, linestyle="--", color=COLORS["low"], label="LOW threshold (-5%)")
ax4.axhline(-8, linestyle="--", color=COLORS["medium"], label="MEDIUM threshold (-8%)")
ax4.axhline(-12, linestyle="--", color=COLORS["high"], label="HIGH threshold (-12%)")
ax4.set_title("VDGR Drawdown with Thresholds")
ax4.set_xlabel("Date")
ax4.set_ylabel("Drawdown %")
ax4.legend()
ax4.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig4)

# -----------------------
# SIGNAL COUNTS
# -----------------------

st.subheader("Signal Counts (Past 12 Months)")
signal_counts = data['Signal'].value_counts().reset_index()
signal_counts.columns = ['Signal', 'Count']
st.dataframe(signal_counts, use_container_width=True)

# -----------------------
# SIGNAL-ONLY TABLE
# -----------------------

st.subheader("Signal Days Only")

signal_table = ensure_date_column(data[data['Signal'] != 'NONE'])
signal_table.rename(columns={"Close": "Price"}, inplace=True)

signal_table['Price'] = signal_table['Price'].round(2)
signal_table['RSI'] = signal_table['RSI'].round(2)
signal_table['VIX'] = signal_table['VIX'].round(2)
signal_table['DrawdownPct'] = signal_table['DrawdownPct'].round(2)

st.dataframe(
    signal_table[['Date', 'Price', 'RSI', 'VIX', 'DrawdownPct', 'Signal', 'Suggested Investment']],
    use_container_width=True
)

# -----------------------
# FULL DAILY HISTORICAL SIGNAL TABLE
# -----------------------

st.subheader("Historical Daily Signals (Previous 12 Months)")

full_table = ensure_date_column(data)
full_table.rename(columns={"Close": "Price"}, inplace=True)

full_table['Price'] = full_table['Price'].round(2)
full_table['RSI'] = full_table['RSI'].round(2)
full_table['VIX'] = full_table['VIX'].round(2)
full_table['DrawdownPct'] = full_table['DrawdownPct'].round(2)

st.dataframe(
    full_table[['Date', 'Price', 'RSI', 'VIX', 'DrawdownPct', 'Signal', 'Suggested Investment']],
    use_container_width=True
)

# -----------------------
# SIMPLE BACKTEST
# -----------------------

st.subheader("Simple Backtest: $800 Fortnightly Base vs Signal-Based")

backtest = data.copy()
backtest['Base Invest'] = 0
backtest['Signal Invest'] = 0

start_date = backtest.index[0]
fortnight_dates = []

current_date = start_date
while current_date <= backtest.index[-1]:
    fortnight_dates.append(current_date)
    current_date += timedelta(days=14)

for d in fortnight_dates:
    nearest_idx = backtest.index.get_indexer([d], method='nearest')[0]
    backtest.iloc[nearest_idx, backtest.columns.get_loc('Base Invest')] = FORTNIGHTLY_BASE
    backtest.iloc[nearest_idx, backtest.columns.get_loc('Signal Invest')] = backtest.iloc[nearest_idx]['Suggested Investment']

backtest['Base Units'] = backtest['Base Invest'] / backtest['Close']
backtest['Signal Units'] = backtest['Signal Invest'] / backtest['Close']

base_total_invested = backtest['Base Invest'].sum()
signal_total_invested = backtest['Signal Invest'].sum()

base_total_units = backtest['Base Units'].sum()
signal_total_units = backtest['Signal Units'].sum()

ending_price = backtest['Close'].iloc[-1]

base_ending_value = base_total_units * ending_price
signal_ending_value = signal_total_units * ending_price

base_avg_price = base_total_invested / base_total_units if base_total_units > 0 else 0
signal_avg_price = signal_total_invested / signal_total_units if signal_total_units > 0 else 0

b1, b2, b3, b4 = st.columns(4)
b1.metric("Base Invested", f"${base_total_invested:,.0f}")
b2.metric("Signal Invested", f"${signal_total_invested:,.0f}")
b3.metric("Base Ending Value", f"${base_ending_value:,.0f}")
b4.metric("Signal Ending Value", f"${signal_ending_value:,.0f}")

b5, b6 = st.columns(2)
b5.metric("Base Avg Buy Price", f"${base_avg_price:.2f}")
b6.metric("Signal Avg Buy Price", f"${signal_avg_price:.2f}")