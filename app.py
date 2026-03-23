import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------
# PAGE SETUP
# -----------------------

st.set_page_config(page_title="VDGR Investment Dashboard", layout="wide")
st.title("VDGR Investment Dashboard")

COLORS = {
    "price": "#1f77b4",
    "low": "#4dabf7",
    "medium": "#f59f00",
    "high": "#e03131",
    "none": "#dee2e6",
    "rsi": "#2b8a3e",
    "vix": "#6f42c1",
    "drawdown": "#495057",
}

# -----------------------
# LOAD DATA
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

    # Drawdown (context only)
    data['6M_High'] = data['Close'].rolling(126).max()
    data['DrawdownPct'] = ((data['Close'] - data['6M_High']) / data['6M_High']) * 100

    # SIGNAL LOGIC (RSI + VIX ONLY)
    data['Signal'] = "NONE"
    data.loc[(data['RSI'] < 50) & (data['VIX'] > 18), 'Signal'] = "LOW"
    data.loc[(data['RSI'] < 45) & (data['VIX'] > 20), 'Signal'] = "MEDIUM"
    data.loc[(data['RSI'] < 35) & (data['VIX'] > 25), 'Signal'] = "HIGH"

    data = data.dropna(subset=['RSI', 'VIX']).copy()

    return data


def ensure_date_column(df):
    df = df.copy().reset_index()
    if 'Date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    return df


def build_reason(row):
    rsi = row['RSI']
    vix = row['VIX']
    drawdown = row['DrawdownPct']

    if row['Signal'] == "HIGH":
        return f"RSI {rsi:.1f} < 35 and VIX {vix:.1f} > 25 | Drawdown {drawdown:.1f}%"
    elif row['Signal'] == "MEDIUM":
        return f"RSI {rsi:.1f} < 45 and VIX {vix:.1f} > 20 | Drawdown {drawdown:.1f}%"
    elif row['Signal'] == "LOW":
        return f"RSI {rsi:.1f} < 50 and VIX {vix:.1f} > 18 | Drawdown {drawdown:.1f}%"
    else:
        return f"No signal | RSI={rsi:.1f}, VIX={vix:.1f}, Drawdown={drawdown:.1f}%"


# -----------------------
# LOAD + PREP
# -----------------------

data = load_data()

latest = data.iloc[-1]
previous = data.iloc[-2]

latest_signal = latest['Signal']
latest_price = float(latest['Close'])
latest_rsi = float(latest['RSI'])
latest_vix = float(latest['VIX'])
latest_drawdown = float(latest['DrawdownPct'])
latest_reason = build_reason(latest)

def suggested_investment(signal):
    if signal == "LOW":
        return 400
    if signal == "MEDIUM":
        return 800
    if signal == "HIGH":
        return 1600
    return 0

latest_investment = suggested_investment(latest_signal)

# -----------------------
# TODAY PANEL
# -----------------------

st.header("Today")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Signal", latest_signal)
col2.metric("Price", f"${latest_price:.2f}")
col3.metric("RSI", f"{latest_rsi:.2f}")
col4.metric("VIX", f"{latest_vix:.2f}")
col5.metric("Suggested Investment", f"${latest_investment}")

st.info(latest_reason)

# -----------------------
# HEATMAP
# -----------------------

st.header("12-Month Signal Heatmap")

heat = ensure_date_column(data)
heat['Date'] = pd.to_datetime(heat['Date'])
heat['Month'] = heat['Date'].dt.strftime('%Y-%m')
heat['Day'] = heat['Date'].dt.day

signal_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
heat['SignalNum'] = heat['Signal'].map(signal_map)

pivot = heat.pivot_table(index='Month', columns='Day', values='SignalNum', aggfunc='first')

fig, ax = plt.subplots(figsize=(16, max(4, len(pivot) * 0.5)))

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        if pd.isna(val):
            color = "white"
        elif val == 0:
            color = COLORS["none"]
        elif val == 1:
            color = COLORS["low"]
        elif val == 2:
            color = COLORS["medium"]
        else:
            color = COLORS["high"]

        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))

ax.set_xlim(0, pivot.shape[1])
ax.set_ylim(0, pivot.shape[0])
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_title("Daily Signal by Month")

plt.tight_layout()
st.pyplot(fig)

# -----------------------
# PRICE + SIGNAL MARKERS
# -----------------------

st.header("Price + Signals")

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(data.index, data['Close'], color=COLORS["price"], label="Price")

for signal, color in [("LOW", COLORS["low"]), ("MEDIUM", COLORS["medium"]), ("HIGH", COLORS["high"])]:
    subset = data[data['Signal'] == signal]
    ax2.scatter(subset.index, subset['Close'], color=color, label=signal, s=40)

ax2.legend()
ax2.grid(alpha=0.2)
plt.xticks(rotation=45)

st.pyplot(fig2)

# -----------------------
# RSI + VIX
# -----------------------

st.header("Indicators")

fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(data.index, data['RSI'], color=COLORS["rsi"])
ax3.axhline(50, linestyle="--")
ax3.axhline(45, linestyle="--")
ax3.axhline(35, linestyle="--")
ax3.set_title("RSI")

st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(14, 4))
ax4.plot(data.index, data['VIX'], color=COLORS["vix"])
ax4.axhline(18, linestyle="--")
ax4.axhline(20, linestyle="--")
ax4.axhline(25, linestyle="--")
ax4.set_title("VIX")

st.pyplot(fig4)

# -----------------------
# TABLE
# -----------------------

st.header("Signal History")

table = ensure_date_column(data)
table.rename(columns={"Close": "Price"}, inplace=True)

st.dataframe(
    table[['Date', 'Price', 'RSI', 'VIX', 'DrawdownPct', 'Signal']],
    use_container_width=True
)