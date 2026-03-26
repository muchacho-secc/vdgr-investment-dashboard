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
    "NONE": "#f1f3f5",
    "LOW": "#ffd43b",
    "MEDIUM": "#f59f00",
    "HIGH": "#e03131",
    "EXTREME": "#7b2cbf",
    "price": "#1f77b4",
    "rsi": "#2b8a3e",
    "vix": "#6f42c1",
    "drawdown": "#495057",
}

TIME_OPTIONS = {
    "3 months": 3,
    "6 months": 6,
    "12 months": 12,
    "24 months": 24,
    "36 months": 36,
}

SIGNAL_COUNT_OPTIONS = {
    "3 months": 3,
    "6 months": 6,
    "1 year": 12,
    "2 years": 24,
    "3 years": 36,
}

INVESTMENT_MAP = {
    "NONE": 0,
    "LOW": 0,
    "MEDIUM": 200,
    "HIGH": 400,
    "EXTREME": 600,
}

# -----------------------
# HELPERS
# -----------------------

def format_full_date(dt_value):
    return pd.to_datetime(dt_value).strftime("%d-%m-%Y")


def filter_months(data, months):
    if data.empty:
        return data.copy()
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(months=months)
    return data[data.index >= start_date].copy()


def investment(signal):
    return INVESTMENT_MAP.get(str(signal).upper(), 0)


def build_reason(signal, rsi, vix, drawdown):
    if signal == "EXTREME":
        return f"RSI ({rsi:.1f}) < 30, VIX ({vix:.1f}) > 30, and drawdown ({drawdown:.1f}%) < -10%"
    if signal == "HIGH":
        return f"RSI ({rsi:.1f}) < 35 and VIX ({vix:.1f}) > 25"
    if signal == "MEDIUM":
        return f"RSI ({rsi:.1f}) < 45 and VIX ({vix:.1f}) > 20"
    if signal == "LOW":
        return f"RSI ({rsi:.1f}) < 50 and VIX ({vix:.1f}) > 18"
    return f"No thresholds met (RSI {rsi:.1f}, VIX {vix:.1f})"


def signal_summary_text(row):
    signal = row["Signal"]
    rsi = row["RSI"]
    vix = row["VIX"]
    drawdown = row["DrawdownPct"]
    suggested = investment(signal)

    if signal == "EXTREME":
        return (
            f"EXTREME signal because RSI is very weak ({rsi:.1f}), VIX is elevated ({vix:.1f}), "
            f"and drawdown is deep ({drawdown:.1f}%). Suggested purchase: ${suggested}."
        )
    if signal == "HIGH":
        return f"HIGH signal because RSI is low ({rsi:.1f}) and VIX is elevated ({vix:.1f}). Suggested purchase: ${suggested}."
    if signal == "MEDIUM":
        return f"MEDIUM signal because RSI is soft ({rsi:.1f}) and VIX shows fear ({vix:.1f}). Suggested purchase: ${suggested}."
    if signal == "LOW":
        return f"LOW signal because RSI is below the low threshold ({rsi:.1f}) and VIX is above calm levels ({vix:.1f}). Suggested purchase: ${suggested}."
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
- EXTREME: RSI < 30 and VIX > 30 and Drawdown < -10%

**Suggested purchases**
- LOW: ${INVESTMENT_MAP['LOW']}
- MEDIUM: ${INVESTMENT_MAP['MEDIUM']}
- HIGH: ${INVESTMENT_MAP['HIGH']}
- EXTREME: ${INVESTMENT_MAP['EXTREME']}

**Today's values**
- RSI: {rsi:.2f}
- VIX: {vix:.2f}
- Drawdown: {dd:.2f}%

**Important**
- Drawdown is **not used** for LOW, MEDIUM, or HIGH
- Drawdown is only used to confirm **EXTREME**
- EXTREME overrides HIGH when its stricter rule is met

**How to read the indicators**
- **RSI** measures momentum:
  - Above 70 = often overbought
  - 50 to 70 = stronger momentum
  - 30 to 50 = softer momentum
  - Below 30 = often oversold
- **VIX** measures market fear:
  - Below 20 = calm market
  - 20 to 25 = mild fear
  - 25 to 30 = elevated fear
  - Above 30 = strong stress
- **Drawdown** measures how far VDGR is below its 6-month high
"""


@st.cache_data(ttl=3600)
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
    vix = vix[["Close"]].copy()
    vix.columns = ["VIX"]

    data = vdgr.join(vix, how="inner")

    if data.empty:
        raise ValueError("Joined VDGR/VIX dataset is empty.")

    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Drawdown from rolling 6-month high
    data["6M_High"] = data["Close"].rolling(126).max()
    data["DrawdownPct"] = ((data["Close"] - data["6M_High"]) / data["6M_High"]) * 100

    # Signal logic
    data["Signal"] = "NONE"
    data.loc[(data["RSI"] < 50) & (data["VIX"] > 18), "Signal"] = "LOW"
    data.loc[(data["RSI"] < 45) & (data["VIX"] > 20), "Signal"] = "MEDIUM"
    data.loc[(data["RSI"] < 35) & (data["VIX"] > 25), "Signal"] = "HIGH"
    data.loc[
        (data["RSI"] < 30) & (data["VIX"] > 30) & (data["DrawdownPct"] < -10),
        "Signal"
    ] = "EXTREME"

    data["Investment"] = data["Signal"].apply(investment)
    data = data.dropna(subset=["Close", "RSI", "VIX", "DrawdownPct"]).copy()
    data.index = pd.to_datetime(data.index)

    return data


def build_recent_signal_timeline(data, days=30):
    recent = data.tail(days).copy()

    if recent.empty:
        return go.Figure()

    recent["PlotX"] = list(range(len(recent)))
    recent["Color"] = recent["Signal"].map(COLORS).fillna(COLORS["NONE"])
    recent["BuyAmount"] = recent["Signal"].apply(investment)

    hover_data = list(
        zip(
            recent.index.strftime("%d-%m-%Y"),
            recent["Signal"],
            recent["BuyAmount"],
            recent["RSI"].round(2),
            recent["VIX"].round(2),
            recent["DrawdownPct"].round(2),
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=recent["PlotX"],
            y=[1] * len(recent),
            marker=dict(
                color=recent["Color"],
                line=dict(color="rgba(255,255,255,0.95)", width=1),
            ),
            width=1.0,
            customdata=hover_data,
            hovertemplate=(
                "Date: %{customdata[0]}<br>"
                "Signal: %{customdata[1]}<br>"
                "Suggested Investment: $%{customdata[2]}<br>"
                "RSI: %{customdata[3]}<br>"
                "VIX: %{customdata[4]}<br>"
                "Drawdown: %{customdata[5]}%<extra></extra>"
            ),
            showlegend=False,
        )
    )

    top_tickvals = []
    top_ticktext = []
    prev_month_label = None

    for plot_x, dt in zip(recent["PlotX"], recent.index):
        month_label = dt.strftime("%b %Y")
        if month_label != prev_month_label:
            top_tickvals.append(plot_x)
            top_ticktext.append(month_label)
            prev_month_label = month_label

    fig.update_xaxes(
        tickmode="array",
        tickvals=recent["PlotX"].tolist(),
        ticktext=recent.index.strftime("%d").tolist(),
        tickfont=dict(size=10),
        tickangle=0,
        showgrid=False,
        zeroline=False,
        showline=False,
        range=[-0.5, len(recent) - 0.5],
        fixedrange=True,
    )

    fig.update_layout(
        height=170,
        margin=dict(l=10, r=10, t=35, b=45),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0,
        xaxis=dict(
            title="",
            side="bottom",
            tickmode="array",
            tickvals=recent["PlotX"].tolist(),
            ticktext=recent.index.strftime("%d").tolist(),
            tickfont=dict(size=10),
            tickangle=0,
            showgrid=False,
            zeroline=False,
            showline=False,
            fixedrange=True,
        ),
        xaxis2=dict(
            title="",
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=top_tickvals,
            ticktext=top_ticktext,
            tickfont=dict(size=11),
            showgrid=False,
            zeroline=False,
            showline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            visible=False,
            range=[0, 1],
            fixedrange=True,
        ),
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

    for signal in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
        subset = data[data["Signal"] == signal]
        fig.add_trace(
            go.Scatter(
                x=subset.index,
                y=subset["Close"],
                mode="markers",
                name=signal,
                marker=dict(size=9, color=COLORS[signal]),
                hovertemplate="Date: %{x|%d-%m-%Y}<br>Price: %{y:.2f}<br>Signal: " + signal + "<extra></extra>",
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
        (30, "EXTREME threshold (30)", COLORS["EXTREME"], "dash"),
        (70, "Overbought reference (70)", "#868e96", "dot"),
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
        (30, "EXTREME threshold (30)", COLORS["EXTREME"], "dash"),
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
        (-10, "EXTREME drawdown threshold (-10%)", COLORS["EXTREME"], "dash"),
        (-15, "-15% reference", COLORS["HIGH"], "dot"),
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

    signal_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "EXTREME": 4}
    color_scale = [
        [0.00, COLORS["NONE"]],
        [0.19, COLORS["NONE"]],
        [0.20, COLORS["LOW"]],
        [0.39, COLORS["LOW"]],
        [0.40, COLORS["MEDIUM"]],
        [0.59, COLORS["MEDIUM"]],
        [0.60, COLORS["HIGH"]],
        [0.79, COLORS["HIGH"]],
        [0.80, COLORS["EXTREME"]],
        [1.00, COLORS["EXTREME"]],
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
            textfont={"size": 14, "color": "black"},
            colorscale=color_scale,
            zmin=0,
            zmax=4,
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
# LOAD DATA
# -----------------------

try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

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
st.caption(f"Drawdown context: {latest['DrawdownPct']:.2f}%")

with st.expander("Detailed explanation of today's signal and indicators"):
    st.markdown(detailed_explanation(latest))

# -----------------------
# RECENT SIGNAL TIMELINE
# -----------------------

st.subheader("Recent Signal Timeline")
st.caption("Daily signal ribbon for the most recent 30 trading days.")

legend_cols = st.columns(5)
legend_cols[0].markdown(
    f"<div style='background:{COLORS['NONE']};padding:6px;border-radius:6px;text-align:center;'>NONE</div>",
    unsafe_allow_html=True
)
legend_cols[1].markdown(
    f"<div style='background:{COLORS['LOW']};padding:6px;border-radius:6px;text-align:center;'>LOW</div>",
    unsafe_allow_html=True
)
legend_cols[2].markdown(
    f"<div style='background:{COLORS['MEDIUM']};padding:6px;border-radius:6px;text-align:center;'>MEDIUM</div>",
    unsafe_allow_html=True
)
legend_cols[3].markdown(
    f"<div style='background:{COLORS['HIGH']};padding:6px;border-radius:6px;text-align:center;color:white;'>HIGH</div>",
    unsafe_allow_html=True
)
legend_cols[4].markdown(
    f"<div style='background:{COLORS['EXTREME']};padding:6px;border-radius:6px;text-align:center;color:white;'>EXTREME</div>",
    unsafe_allow_html=True
)

st.plotly_chart(build_recent_signal_timeline(data, 30), use_container_width=True)

# -----------------------
# SIGNAL SUMMARY
# -----------------------

st.subheader("Signal Summary")

signal_count_range = st.radio(
    "Signal count range",
    list(SIGNAL_COUNT_OPTIONS.keys()),
    index=2,
    horizontal=True,
    key="signal_count_range",
)

count_data = filter_months(data, SIGNAL_COUNT_OPTIONS[signal_count_range])
signal_counts = count_data["Signal"].value_counts()

c1, c2, c3, c4 = st.columns(4)
c1.metric("LOW Signals", int(signal_counts.get("LOW", 0)))
c2.metric("MEDIUM Signals", int(signal_counts.get("MEDIUM", 0)))
c3.metric("HIGH Signals", int(signal_counts.get("HIGH", 0)))
c4.metric("EXTREME Signals", int(signal_counts.get("EXTREME", 0)))

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

# -----------------------
# CALENDAR
# -----------------------

with st.expander("Signal Calendar", expanded=False):
    st.caption("Compact calendar-style view of historical daily signals. Latest month shown first.")

    calendar_range = st.radio("Calendar range", ["3 months", "6 months", "12 months"], index=0, horizontal=True, key="calendar_range")
    cal_data = filter_months(data, TIME_OPTIONS[calendar_range]).copy()

    legend_cols = st.columns(5)
    legend_cols[0].markdown(f"<div style='background:{COLORS['NONE']};padding:8px;border-radius:6px;text-align:center;'>NONE</div>", unsafe_allow_html=True)
    legend_cols[1].markdown(f"<div style='background:{COLORS['LOW']};padding:8px;border-radius:6px;text-align:center;'>LOW</div>", unsafe_allow_html=True)
    legend_cols[2].markdown(f"<div style='background:{COLORS['MEDIUM']};padding:8px;border-radius:6px;text-align:center;'>MEDIUM</div>", unsafe_allow_html=True)
    legend_cols[3].markdown(f"<div style='background:{COLORS['HIGH']};padding:8px;border-radius:6px;text-align:center;color:white;'>HIGH</div>", unsafe_allow_html=True)
    legend_cols[4].markdown(f"<div style='background:{COLORS['EXTREME']};padding:8px;border-radius:6px;text-align:center;color:white;'>EXTREME</div>", unsafe_allow_html=True)

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
# HISTORY TABLES
# -----------------------

st.subheader("History")

history_tab1, history_tab2 = st.tabs(["Signal Days Only", "Full History"])

table_base = data.reset_index().copy().rename(columns={"index": "Date"})
table_base = table_base.sort_values("Date", ascending=False)

with history_tab1:
    signal_table = table_base[table_base["Signal"] != "NONE"].copy()
    signal_table["Date"] = pd.to_datetime(signal_table["Date"]).dt.strftime("%d-%m-%Y")
    signal_table = signal_table.rename(columns={"Close": "Price", "Investment": "Suggested Investment"})
    signal_table = signal_table[["Date", "Price", "RSI", "VIX", "DrawdownPct", "Signal", "Suggested Investment"]]
    st.dataframe(signal_table, use_container_width=True)

with history_tab2:
    full_table = table_base.copy()
    full_table["Date"] = pd.to_datetime(full_table["Date"]).dt.strftime("%d-%m-%Y")
    full_table = full_table.rename(columns={"Close": "Price", "Investment": "Suggested Investment"})
    full_table = full_table[["Date", "Price", "RSI", "VIX", "DrawdownPct", "Signal", "Suggested Investment"]]
    st.dataframe(full_table, use_container_width=True)