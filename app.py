import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import calendar

st.set_page_config(page_title="VDGR Investment Dashboard", layout="wide")

st.title("VDGR Investment Dashboard")

# -----------------------
# COLOURS
# -----------------------

COLORS = {
    "LOW": "#4dabf7",
    "MEDIUM": "#f59f00",
    "HIGH": "#e03131",
    "NONE": "#dee2e6",
    "price": "#1f77b4",
    "rsi": "#2b8a3e",
    "vix": "#6f42c1",
    "drawdown": "#495057",
}

# -----------------------
# DATA
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
    vix.columns = ["VIX"]

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

    def investment(signal):
        return {"LOW": 400, "MEDIUM": 800, "HIGH": 1600}.get(signal, 0)

    data['Investment'] = data['Signal'].apply(investment)

    return data.dropna()


def format_full_date(dt_value):
    return pd.to_datetime(dt_value).strftime("%d-%m-%Y")


def signal_summary_text(row):
    signal = row["Signal"]
    rsi = row["RSI"]
    vix = row["VIX"]

    if signal == "HIGH":
        return f"HIGH signal because RSI is low ({rsi:.1f}) and VIX is elevated ({vix:.1f})."
    elif signal == "MEDIUM":
        return f"MEDIUM signal because RSI is soft ({rsi:.1f}) and VIX shows fear ({vix:.1f})."
    elif signal == "LOW":
        return f"LOW signal because RSI is below the low threshold ({rsi:.1f}) and VIX is above calm levels ({vix:.1f})."
    else:
        return f"No signal today because RSI ({rsi:.1f}) and VIX ({vix:.1f}) did not meet the alert thresholds together."


def detailed_explanation(row):
    signal = row["Signal"]
    rsi = row["RSI"]
    vix = row["VIX"]
    dd = row["DrawdownPct"]

    return f"""
**Signal logic used**
- LOW: RSI < 50 and VIX > 18  
- MEDIUM: RSI < 45 and VIX > 20  
- HIGH: RSI < 35 and VIX > 25  

**Today's values**
- RSI: {rsi:.2f}
- VIX: {vix:.2f}
- Drawdown: {dd:.2f}%

**Important**
- Drawdown is shown as **additional context only**
- Drawdown is **not included** in the signal logic
- A buy signal is triggered only by the RSI + VIX rules above

**How to read the indicators**
- **RSI** measures momentum:
  - Above 70 = often overbought / stretched high
  - 50 to 70 = stronger momentum
  - 30 to 50 = softer / weaker momentum
  - Below 30 = often oversold
- **VIX** measures market fear / volatility:
  - Below 20 = calm market
  - 20 to 25 = mild fear
  - 25 to 30 = elevated fear
  - Above 30 = strong stress / panic
"""


def build_calendar_heatmap(df):
    df = df.copy().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')

    month_periods = sorted(df['YearMonth'].unique())

    for period in month_periods:
        month_df = df[df['YearMonth'] == period].copy()
        year = period.year
        month = period.month

        st.markdown(f"#### {calendar.month_name[month]} {year}")

        cal = calendar.Calendar(firstweekday=0)  # Monday
        month_days = cal.monthdatescalendar(year, month)

        fig, ax = plt.subplots(figsize=(10, 2.8))

        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for week_idx, week in enumerate(month_days):
            for day_idx, day in enumerate(week):
                if day.month != month:
                    facecolor = "#ffffff"
                    label = ""
                else:
                    row = month_df[month_df['Date'].dt.date == day]
                    if row.empty:
                        facecolor = COLORS["NONE"]
                    else:
                        signal = row.iloc[0]["Signal"]
                        facecolor = COLORS.get(signal, COLORS["NONE"])
                    label = str(day.day)

                rect = plt.Rectangle((day_idx, len(month_days)-1-week_idx), 1, 1,
                                     facecolor=facecolor, edgecolor="white", linewidth=1.5)
                ax.add_patch(rect)
                ax.text(day_idx + 0.5, len(month_days)-1-week_idx + 0.5, label,
                        ha='center', va='center', fontsize=9, color="black")

        ax.set_xlim(0, 7)
        ax.set_ylim(0, len(month_days))
        ax.set_xticks([i + 0.5 for i in range(7)])
        ax.set_xticklabels(weekday_labels)
        ax.set_yticks([])
        ax.set_title(f"Signal Calendar - {calendar.month_name[month]} {year}", fontsize=11)
        ax.tick_params(axis='x', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        legend_handles = [
            mpatches.Patch(color=COLORS["NONE"], label="NONE"),
            mpatches.Patch(color=COLORS["LOW"], label="LOW"),
            mpatches.Patch(color=COLORS["MEDIUM"], label="MEDIUM"),
            mpatches.Patch(color=COLORS["HIGH"], label="HIGH"),
        ]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False)

        plt.tight_layout()
        st.pyplot(fig)


data = load_data()
latest = data.iloc[-1]

# -----------------------
# ACTION PANEL
# -----------------------

st.subheader("Today's Signal")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Signal", latest["Signal"])
col2.metric("Suggested Investment", f"${int(latest['Investment'])}")
col3.metric("RSI", f"{latest['RSI']:.2f}")
col4.metric("VIX", f"{latest['VIX']:.2f}")
col5.metric("Date", format_full_date(data.index[-1]))

st.success(signal_summary_text(latest))
st.caption(f"Drawdown context: {latest['DrawdownPct']:.2f}% (shown for context only, not used in signal logic)")

with st.expander("Detailed explanation of today's signal and indicators"):
    st.markdown(detailed_explanation(latest))

# -----------------------
# SIGNAL SUMMARY
# -----------------------

st.subheader("Signal Summary")

signal_counts = data['Signal'].value_counts()
c1, c2, c3 = st.columns(3)
c1.metric("LOW Signals", int(signal_counts.get("LOW", 0)))
c2.metric("MEDIUM Signals", int(signal_counts.get("MEDIUM", 0)))
c3.metric("HIGH Signals", int(signal_counts.get("HIGH", 0)))

# -----------------------
# CALENDAR HEATMAP
# -----------------------

st.subheader("12-Month Signal Calendar")
st.caption("Calendar-style view of historical daily signals across the previous 12 months.")
build_calendar_heatmap(data)

# -----------------------
# PRICE CHART
# -----------------------

st.subheader("VDGR Price with Signal Markers")

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(data.index, data['Close'], color=COLORS["price"], linewidth=2, label="VDGR Price")

for signal, color in [("LOW", COLORS["LOW"]), ("MEDIUM", COLORS["MEDIUM"]), ("HIGH", COLORS["HIGH"])]:
    subset = data[data["Signal"] == signal]
    ax2.scatter(subset.index, subset['Close'], color=color, label=signal, s=35)

ax2.set_title("VDGR Price Over the Past 12 Months")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
ax2.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# -----------------------
# RSI
# -----------------------

st.subheader("RSI")

fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(data.index, data['RSI'], color=COLORS["rsi"], linewidth=2, label="RSI")
ax3.axhline(50, color=COLORS["LOW"], linestyle="--", label="LOW threshold (50)")
ax3.axhline(45, color=COLORS["MEDIUM"], linestyle="--", label="MEDIUM threshold (45)")
ax3.axhline(35, color=COLORS["HIGH"], linestyle="--", label="HIGH threshold (35)")
ax3.axhline(70, color="#868e96", linestyle=":", label="Overbought reference (70)")
ax3.axhline(30, color="#868e96", linestyle=":", label="Oversold reference (30)")
ax3.set_title("RSI with Thresholds")
ax3.set_xlabel("Date")
ax3.set_ylabel("RSI")
ax3.legend()
ax3.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

# -----------------------
# VIX
# -----------------------

st.subheader("VIX")

fig4, ax4 = plt.subplots(figsize=(14, 4))
ax4.plot(data.index, data['VIX'], color=COLORS["vix"], linewidth=2, label="VIX")
ax4.axhline(18, color=COLORS["LOW"], linestyle="--", label="LOW threshold (18)")
ax4.axhline(20, color=COLORS["MEDIUM"], linestyle="--", label="MEDIUM threshold (20)")
ax4.axhline(25, color=COLORS["HIGH"], linestyle="--", label="HIGH threshold (25)")
ax4.axhline(30, color="#495057", linestyle=":", label="Stress reference (30)")
ax4.set_title("VIX with Thresholds")
ax4.set_xlabel("Date")
ax4.set_ylabel("VIX")
ax4.legend()
ax4.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig4)

# -----------------------
# DRAWDOWN
# -----------------------

st.subheader("Drawdown (Context Only)")

fig5, ax5 = plt.subplots(figsize=(14, 4))
ax5.plot(data.index, data['DrawdownPct'], color=COLORS["drawdown"], linewidth=2, label="Drawdown %")
ax5.axhline(-5, color=COLORS["LOW"], linestyle="--", label="-5% reference")
ax5.axhline(-10, color=COLORS["MEDIUM"], linestyle="--", label="-10% reference")
ax5.axhline(-15, color=COLORS["HIGH"], linestyle="--", label="-15% reference")
ax5.set_title("Drawdown from 6-Month High")
ax5.set_xlabel("Date")
ax5.set_ylabel("Drawdown %")
ax5.legend()
ax5.grid(alpha=0.2)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig5)

st.caption("Drawdown is displayed as extra context. It is not part of the alert rules for LOW / MEDIUM / HIGH signals.")

# -----------------------
# TABLES
# -----------------------

st.subheader("Recent Signals")

table = data.reset_index().copy()
table['Date'] = pd.to_datetime(table['Date']).dt.strftime("%d-%m-%Y")
table = table.rename(columns={"Close": "Price", "Investment": "Suggested Investment"})
table = table[['Date', 'Price', 'RSI', 'VIX', 'DrawdownPct', 'Signal', 'Suggested Investment']]
table = table.sort_values("Date", ascending=False)

st.dataframe(table, use_container_width=True)