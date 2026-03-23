import streamlit as st
import pandas as pd
import yfinance as yf
import calendar
import plotly.graph_objects as go

st.set_page_config(page_title="VDGR Investment Dashboard", layout="wide")

st.title("VDGR Investment Dashboard")

# -----------------------
# SETTINGS
# -----------------------

COLORS = {
    "LOW": "#ffd43b",       # yellow
    "MEDIUM": "#f59f00",    # orange
    "HIGH": "#e03131",      # red
    "NONE": "#f1f3f5",      # very light grey
    "price": "#1f77b4",     # blue
    "rsi": "#2b8a3e",       # green
    "vix": "#6f42c1",       # purple
    "drawdown": "#495057",  # dark grey
}

TIME_OPTIONS = {
    "3 months": 3,
    "6 months": 6,
    "12 months": 12,
    "24 months": 24,
    "36 months": 36,
}

# -----------------------
# HELPERS
# -----------------------

def format_full_date(dt_value):
    return pd.to_datetime(dt_value).strftime("%d-%m-%Y")


def filter_months(data, months):
    if data.empty:
        return data
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(months=months)
    return data[data.index >= start_date].copy()


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


@st.cache_data(ttl=3600)
def load_data():
    vdgr = yf.download("VDGR.AX", period="36mo", auto_adjust=False)
    vix = yf.download("^VIX", period="36mo", auto_adjust=False)

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

    # Signal logic: RSI + VIX ONLY
    data['Signal'] = "NONE"
    data.loc[(data['RSI'] < 50) & (data['VIX'] > 18), 'Signal'] = "LOW"
    data.loc[(data['RSI'] < 45) & (data['VIX'] > 20), 'Signal'] = "MEDIUM"
    data.loc[(data['RSI'] < 35) & (data['VIX'] > 25), 'Signal'] = "HIGH"

    def investment(signal):
        return {"LOW": 400, "MEDIUM": 800, "HIGH": 1600}.get(signal, 0)

    data['Investment'] = data['Signal'].apply(investment)
    data = data.dropna().copy()
    data.index = pd.to_datetime(data.index)

    return data


def build_recent_signal_strip(data, days=30):
    recent = data.tail(days).copy()
    signal_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    z = [recent["Signal"].map(signal_map).tolist()]
    x = [format_full_date(d) for d in recent.index]

    colorscale = [
        [0.00, COLORS["NONE"]],
        [0.24, COLORS["NONE"]],
        [0.25, COLORS["LOW"]],
        [0.49, COLORS["LOW"]],
        [0.50, COLORS["MEDIUM"]],
        [0.74, COLORS["MEDIUM"]],
        [0.75, COLORS["HIGH"]],
        [1.00, COLORS["HIGH"]],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=["Signal"],
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            showscale=False,
            hovertemplate="Date: %{x}<extra></extra>",
        )
    )

    fig.update_layout(
        height=130,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Recent Trading Days",
        yaxis_showticklabels=False,
    )
    return fig


def build_price_chart(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="VDGR Price",
            line=dict(color=COLORS["price"], width=2),
            hovertemplate="Date: %{x|%d-%m-%Y}<br>Price: %{y:.2f}<extra></extra>",
        )
    )

    for signal in ["LOW", "MEDIUM", "HIGH"]:
        subset = data[data["Signal"] == signal]
        fig.add_trace(
            go.Scatter(
                x=subset.index,
                y=subset["Close"],
                mode="markers",
                name=signal,
                marker=dict(size=9, color=COLORS[signal]),
                hovertemplate=(
                    "Date: %{x|%d-%m-%Y}<br>"
                    "Price: %{y:.2f}<br>"
                    f"Signal: {signal}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=500,
        legend_title_text="Series",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    return fig


def build_rsi_chart(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color=COLORS["rsi"], width=2),
            hovertemplate="Date: %{x|%d-%m-%Y}<br>RSI: %{y:.2f}<extra></extra>",
        )
    )

    for y_val, name, color, dash in [
        (50, "LOW threshold (50)", COLORS["LOW"], "dash"),
        (45, "MEDIUM threshold (45)", COLORS["MEDIUM"], "dash"),
        (35, "HIGH threshold (35)", COLORS["HIGH"], "dash"),
        (70, "Overbought reference (70)", "#868e96", "dot"),
        (30, "Oversold reference (30)", "#868e96", "dot"),
    ]:
        fig.add_hline(y=y_val, line_color=color, line_dash=dash, annotation_text=name, annotation_position="top left")

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="RSI",
        showlegend=True,
        hovermode="x unified",
    )
    return fig


def build_vix_chart(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["VIX"],
            mode="lines",
            name="VIX",
            line=dict(color=COLORS["vix"], width=2),
            hovertemplate="Date: %{x|%d-%m-%Y}<br>VIX: %{y:.2f}<extra></extra>",
        )
    )

    for y_val, name, color, dash in [
        (18, "LOW threshold (18)", COLORS["LOW"], "dash"),
        (20, "MEDIUM threshold (20)", COLORS["MEDIUM"], "dash"),
        (25, "HIGH threshold (25)", COLORS["HIGH"], "dash"),
        (30, "Stress reference (30)", "#495057", "dot"),
    ]:
        fig.add_hline(y=y_val, line_color=color, line_dash=dash, annotation_text=name, annotation_position="top left")

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="VIX",
        showlegend=True,
        hovermode="x unified",
    )
    return fig


def build_drawdown_chart(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["DrawdownPct"],
            mode="lines",
            name="Drawdown %",
            line=dict(color=COLORS["drawdown"], width=2),
            hovertemplate="Date: %{x|%d-%m-%Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    for y_val, name, color, dash in [
        (-5, "-5% reference", COLORS["LOW"], "dash"),
        (-10, "-10% reference", COLORS["MEDIUM"], "dash"),
        (-15, "-15% reference", COLORS["HIGH"], "dash"),
    ]:
        fig.add_hline(y=y_val, line_color=color, line_dash=dash, annotation_text=name, annotation_position="bottom left")

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        showlegend=True,
        hovermode="x unified",
    )
    return fig


def build_month_calendar_plot(month_df, year, month):
    cal = calendar.Calendar(firstweekday=0)
    month_days = cal.monthdatescalendar(year, month)

    z = []
    text = []

    signal_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    color_scale = [
        [0.00, COLORS["NONE"]],
        [0.24, COLORS["NONE"]],
        [0.25, COLORS["LOW"]],
        [0.49, COLORS["LOW"]],
        [0.50, COLORS["MEDIUM"]],
        [0.74, COLORS["MEDIUM"]],
        [0.75, COLORS["HIGH"]],
        [1.00, COLORS["HIGH"]],
    ]

    for week in month_days:
        z_row = []
        text_row = []
        for day in week:
            if day.month != month:
                z_row.append(None)
                text_row.append("")
            else:
                row = month_df[month_df["Date"].dt.date == day]
                if row.empty:
                    z_row.append(0)
                    text_row.append(str(day.day))
                else:
                    signal = row.iloc[0]["Signal"]
                    z_row.append(signal_map.get(signal, 0))
                    text_row.append(str(day.day))
        z.append(z_row)
        text.append(text_row)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z[::-1],
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=[f"Week {i+1}" for i in range(len(z))][::-1],
            text=text[::-1],
            texttemplate="%{text}",
            textfont={"size": 15, "color": "black"},
            colorscale=color_scale,
            zmin=0,
            zmax=3,
            showscale=False,
            hoverongaps=False,
            hovertemplate="Day: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        height=210,
        margin=dict(l=10, r=10, t=0, b=0),
        xaxis=dict(side="bottom"),
        yaxis=dict(showticklabels=False),
    )
    return fig


# -----------------------
# LOAD BASE DATA
# -----------------------

data = load_data()

if data.empty:
    st.error("No data available.")
    st.stop()

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
# RECENT SIGNAL STRIP
# -----------------------

st.subheader("Recent Signal Timeline")
st.plotly_chart(build_recent_signal_strip(data, 30), use_container_width=True)

# -----------------------
# SIGNAL SUMMARY
# -----------------------

st.subheader("Signal Summary")
signal_counts_12m = filter_months(data, 12)['Signal'].value_counts()

c1, c2, c3 = st.columns(3)
c1.metric("LOW Signals (12m)", int(signal_counts_12m.get("LOW", 0)))
c2.metric("MEDIUM Signals (12m)", int(signal_counts_12m.get("MEDIUM", 0)))
c3.metric("HIGH Signals (12m)", int(signal_counts_12m.get("HIGH", 0)))

# -----------------------
# PRICE CHART
# -----------------------

st.subheader("VDGR Price with Signal Markers")
price_range = st.radio("Price chart range", list(TIME_OPTIONS.keys()), index=2, horizontal=True, key="price_range")
price_data = filter_months(data, TIME_OPTIONS[price_range])
st.plotly_chart(build_price_chart(price_data), use_container_width=True)

# -----------------------
# INDICATORS
# -----------------------

st.subheader("Indicators")

tab_rsi, tab_vix, tab_dd = st.tabs(["RSI", "VIX", "Drawdown"])

with tab_rsi:
    rsi_range = st.radio("RSI range", ["3 months", "6 months", "12 months"], index=1, horizontal=True, key="rsi_range")
    rsi_data = filter_months(data, TIME_OPTIONS[rsi_range])
    st.plotly_chart(build_rsi_chart(rsi_data), use_container_width=True)

with tab_vix:
    vix_range = st.radio("VIX range", ["3 months", "6 months", "12 months"], index=1, horizontal=True, key="vix_range")
    vix_data = filter_months(data, TIME_OPTIONS[vix_range])
    st.plotly_chart(build_vix_chart(vix_data), use_container_width=True)

with tab_dd:
    dd_range = st.radio("Drawdown range", ["6 months", "12 months", "24 months", "36 months"], index=1, horizontal=True, key="dd_range")
    dd_data = filter_months(data, TIME_OPTIONS[dd_range])
    st.plotly_chart(build_drawdown_chart(dd_data), use_container_width=True)
    st.caption("Drawdown is displayed as extra context. It is not part of the alert rules for LOW / MEDIUM / HIGH signals.")

# -----------------------
# CALENDAR
# -----------------------

with st.expander("Signal Calendar", expanded=False):
    st.caption("Compact calendar-style view of historical daily signals. Latest month shown first.")

    calendar_range = st.radio("Calendar range", ["3 months", "6 months", "12 months"], index=0, horizontal=True, key="calendar_range")
    cal_data = filter_months(data, TIME_OPTIONS[calendar_range]).copy()

    legend_cols = st.columns(4)
    legend_cols[0].markdown(f"<div style='background:{COLORS['NONE']};padding:8px;border-radius:6px;text-align:center;'>NONE</div>", unsafe_allow_html=True)
    legend_cols[1].markdown(f"<div style='background:{COLORS['LOW']};padding:8px;border-radius:6px;text-align:center;'>LOW</div>", unsafe_allow_html=True)
    legend_cols[2].markdown(f"<div style='background:{COLORS['MEDIUM']};padding:8px;border-radius:6px;text-align:center;'>MEDIUM</div>", unsafe_allow_html=True)
    legend_cols[3].markdown(f"<div style='background:{COLORS['HIGH']};padding:8px;border-radius:6px;text-align:center;color:white;'>HIGH</div>", unsafe_allow_html=True)

    calendar_df = cal_data.reset_index().rename(columns={"index": "Date"})
    calendar_df["Date"] = pd.to_datetime(calendar_df["Date"])
    calendar_df["YearMonth"] = calendar_df["Date"].dt.to_period("M")
    month_periods = sorted(calendar_df["YearMonth"].unique(), reverse=True)

    for i in range(0, len(month_periods), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(month_periods):
                period_month = month_periods[i + j]
                year = period_month.year
                month = period_month.month
                month_df = calendar_df[calendar_df["YearMonth"] == period_month].copy()

                with cols[j]:
                    st.markdown(f"**{calendar.month_name[month]} {year}**")
                    st.plotly_chart(build_month_calendar_plot(month_df, year, month), use_container_width=True)

# -----------------------
# TABLES
# -----------------------

st.subheader("History")

history_tab1, history_tab2 = st.tabs(["Signal Days Only", "Full History"])

table_base = data.reset_index().copy().rename(columns={"index": "Date"})
table_base = table_base.sort_values("Date", ascending=False)

with history_tab1:
    signal_table = table_base[table_base["Signal"] != "NONE"].copy()
    signal_table["Date"] = pd.to_datetime(signal_table["Date"]).dt.strftime("%d-%m-%Y")
    signal_table = signal_table.rename(columns={"Close": "Price", "Investment": "Suggested Investment"})
    signal_table = signal_table[['Date', 'Price', 'RSI', 'VIX', 'DrawdownPct', 'Signal', 'Suggested Investment']]
    st.dataframe(signal_table, use_container_width=True)

with history_tab2:
    full_table = table_base.copy()
    full_table["Date"] = pd.to_datetime(full_table["Date"]).dt.strftime("%d-%m-%Y")
    full_table = full_table.rename(columns={"Close": "Price", "Investment": "Suggested Investment"})
    full_table = full_table[['Date', 'Price', 'RSI', 'VIX', 'DrawdownPct', 'Signal', 'Suggested Investment']]
    st.dataframe(full_table, use_container_width=True)