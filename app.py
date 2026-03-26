import calendar
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from signals import (
    LAUNCH_DATE,
    LEDGER_FILE,
    add_forward_returns,
    detailed_reason,
    filter_months,
    format_display_signal,
    investment,
    live_breakdown_by_signal,
    live_performance_ledger,
    live_summary,
    prepare_signal_data,
    forward_return_summary,
)

st.set_page_config(page_title="VDGR Investment Dashboard", layout="wide")
st.title("VDGR Investment Dashboard")

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

TIME_OPTIONS = {"3 months": 3, "6 months": 6, "12 months": 12, "24 months": 24, "36 months": 36}
SIGNAL_COUNT_OPTIONS = {"3 months": 3, "6 months": 6, "1 year": 12, "2 years": 24, "3 years": 36}

def format_full_date(dt_value):
    return pd.to_datetime(dt_value).strftime("%d-%m-%Y")

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
    vix = vix[["Close"]].copy().rename(columns={"Close": "VIX"})
    data = vdgr.join(vix, how="inner")
    if data.empty:
        raise ValueError("Joined VDGR/VIX dataset is empty.")

    data = prepare_signal_data(data)
    data = add_forward_returns(data)
    return data

def signal_summary_text(row):
    suggested = investment(row["Signal"])
    reason = detailed_reason(row)
    if row["Signal"] == "NONE":
        return reason
    return f"{row['DisplaySignal']}: {reason} Suggested purchase: ${suggested}."

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
            recent["DisplaySignal"],
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
            marker=dict(color=recent["Color"], line=dict(color="rgba(255,255,255,0.95)", width=1)),
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

    top_tickvals, top_ticktext, prev_month_label = [], [], None
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
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
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
                name=format_display_signal(signal),
                marker=dict(size=9, color=COLORS[signal]),
                hovertemplate="Date: %{x|%d-%m-%Y}<br>Price: %{y:.2f}<br>Signal: " + format_display_signal(signal) + "<extra></extra>",
            )
        )
    fig.update_layout(height=500, legend_title_text="Series", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    return fig

def build_indicator_chart(data, column, yaxis_title, lines):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            mode="lines",
            name=yaxis_title,
            line=dict(width=2),
            hovertemplate=f"Date: %{{x|%d-%m-%Y}}<br>{yaxis_title}: %{{y:.2f}}<extra></extra>",
        )
    )
    for y_val, name, color, dash, pos in lines:
        fig.add_hline(y=y_val, line_color=color, line_dash=dash, annotation_text=name, annotation_position=pos)
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title=yaxis_title, showlegend=True, hovermode="x unified")
    return fig

def build_month_calendar_plot(month_df, year, month):
    cal = calendar.Calendar(firstweekday=0)
    month_days = cal.monthdatescalendar(year, month)
    z, text = [], []
    signal_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "EXTREME": 4}
    color_scale = [
        [0.00, COLORS["NONE"]], [0.19, COLORS["NONE"]],
        [0.20, COLORS["LOW"]], [0.39, COLORS["LOW"]],
        [0.40, COLORS["MEDIUM"]], [0.59, COLORS["MEDIUM"]],
        [0.60, COLORS["HIGH"]], [0.79, COLORS["HIGH"]],
        [0.80, COLORS["EXTREME"]], [1.00, COLORS["EXTREME"]],
    ]
    for week in month_days:
        z_row, text_row = [], []
        for day in week:
            if day.month != month:
                z_row.append(None); text_row.append("")
            else:
                row = month_df[month_df["Date"].dt.date == day]
                if row.empty:
                    z_row.append(0); text_row.append(str(day.day))
                else:
                    z_row.append(signal_map.get(row.iloc[0]["Signal"], 0)); text_row.append(str(day.day))
        z.append(z_row); text.append(text_row)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z[::-1], x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=[f"Week {i+1}" for i in range(len(z))][::-1],
            text=text[::-1], texttemplate="%{text}", textfont={"size": 14, "color": "black"},
            colorscale=color_scale, zmin=0, zmax=4, showscale=False, hoverongaps=False,
            hovertemplate="Day: %{text}<extra></extra>",
        )
    )
    fig.update_layout(height=210, margin=dict(l=10, r=10, t=0, b=0), xaxis=dict(side="bottom"), yaxis=dict(showticklabels=False))
    return fig

data = load_data()
latest = data.iloc[-1]
latest_market_date = data.index[-1]

st.subheader("Today's Signal")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Signal", latest["DisplaySignal"])
col2.metric("Suggested Investment", f"${int(latest['Investment'])}")
col3.metric("RSI", f"{latest['RSI']:.2f}")
col4.metric("VIX", f"{latest['VIX']:.2f}")
col5.metric("Latest Market Date", format_full_date(latest_market_date))
st.success(signal_summary_text(latest))
st.caption(f"Launch date for live method tracking: {LAUNCH_DATE.strftime('%d-%m-%Y')}")

st.subheader("Recent Signal Timeline")
st.caption("Daily signal ribbon for the most recent 30 trading days.")
legend_cols = st.columns(5)
legend_labels = [("NONE", "NONE"), ("LOW", "WATCH"), ("MEDIUM", "MEDIUM"), ("HIGH", "HIGH"), ("EXTREME", "EXTREME")]
for i, (key, label) in enumerate(legend_labels):
    text_color = "white" if key in {"HIGH", "EXTREME"} else "black"
    legend_cols[i].markdown(
        f"<div style='background:{COLORS[key]};padding:6px;border-radius:6px;text-align:center;color:{text_color};'>{label}</div>",
        unsafe_allow_html=True,
    )
st.plotly_chart(build_recent_signal_timeline(data, 30), use_container_width=True)

st.subheader("Signal Summary")
signal_count_range = st.radio("Signal count range", list(SIGNAL_COUNT_OPTIONS.keys()), index=2, horizontal=True, key="signal_count_range")
count_data = filter_months(data, SIGNAL_COUNT_OPTIONS[signal_count_range])
signal_counts = count_data["Signal"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("WATCH Signals", int(signal_counts.get("LOW", 0)))
c2.metric("MEDIUM Signals", int(signal_counts.get("MEDIUM", 0)))
c3.metric("HIGH Signals", int(signal_counts.get("HIGH", 0)))
c4.metric("EXTREME Signals", int(signal_counts.get("EXTREME", 0)))

st.subheader("Live Method Performance (from 27 Mar 2026)")
live_ledger = live_performance_ledger(data, LEDGER_FILE)
live_stats = live_summary(live_ledger)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Invested", f"${live_stats['total_invested']:,.2f}")
c2.metric("Current Value", f"${live_stats['current_value']:,.2f}")
c3.metric("Profit / Loss", f"${live_stats['profit_loss']:,.2f}")
c4.metric("Return %", f"{live_stats['return_pct']:.2f}%")
c5.metric("Total Units", f"{live_stats['total_units']:.4f}")
c6.metric("Live Buys", f"{live_stats['num_buys']}")

if live_ledger.empty:
    st.info("No live trades have been recorded yet. The live ledger will start recording from 27-03-2026 when a MEDIUM, HIGH, or EXTREME signal occurs.")
else:
    breakdown = live_breakdown_by_signal(live_ledger)
    st.caption(f"Current valuation uses latest available VDGR close from {format_full_date(latest_market_date)}.")
    st.dataframe(breakdown.round(2), use_container_width=True)
    display_ledger = live_ledger.copy()
    display_ledger["TradeDate"] = pd.to_datetime(display_ledger["TradeDate"]).dt.strftime("%d-%m-%Y")
    display_ledger["CurrentMarketDate"] = pd.to_datetime(display_ledger["CurrentMarketDate"]).dt.strftime("%d-%m-%Y")
    display_ledger = display_ledger[["TradeDate", "DisplaySignal", "Confidence", "BuyPrice", "AmountInvested", "UnitsBought", "CurrentPrice", "CurrentValue", "ProfitLoss", "ReturnPct", "CurrentMarketDate"]]
    st.dataframe(display_ledger.round(4), use_container_width=True)

st.subheader("Forward Return Analysis")
st.caption("Historical research view: average forward returns after signals over the selected lookback window.")
summary_data = filter_months(data, SIGNAL_COUNT_OPTIONS[signal_count_range])
summary = forward_return_summary(summary_data)
if summary.empty:
    st.info("Not enough data yet to calculate forward returns.")
else:
    st.dataframe(summary.round(2), use_container_width=True)

st.subheader("VDGR Price with Signal Markers")
price_range = st.radio("Price chart range", list(TIME_OPTIONS.keys()), index=2, horizontal=True, key="price_range")
price_data = filter_months(data, TIME_OPTIONS[price_range])
st.plotly_chart(build_price_chart(price_data), use_container_width=True)

st.subheader("Indicators")
tab_rsi, tab_vix, tab_dd = st.tabs(["RSI", "VIX", "Drawdown"])
with tab_rsi:
    rsi_range = st.radio("RSI range", ["3 months", "6 months", "12 months"], index=1, horizontal=True, key="rsi_range")
    rsi_data = filter_months(data, TIME_OPTIONS[rsi_range])
    st.plotly_chart(build_indicator_chart(rsi_data, "RSI", "RSI", [
        (50, "WATCH threshold (50)", COLORS["LOW"], "dash", "top left"),
        (45, "MEDIUM threshold (45)", COLORS["MEDIUM"], "dash", "top left"),
        (35, "HIGH threshold (35)", COLORS["HIGH"], "dash", "top left"),
        (30, "EXTREME threshold (30)", COLORS["EXTREME"], "dash", "top left"),
        (70, "Overbought reference (70)", "#868e96", "dot", "top left"),
    ]), use_container_width=True)
with tab_vix:
    vix_range = st.radio("VIX range", ["3 months", "6 months", "12 months"], index=1, horizontal=True, key="vix_range")
    vix_data = filter_months(data, TIME_OPTIONS[vix_range])
    st.plotly_chart(build_indicator_chart(vix_data, "VIX", "VIX", [
        (18, "WATCH threshold (18)", COLORS["LOW"], "dash", "top left"),
        (20, "MEDIUM threshold (20)", COLORS["MEDIUM"], "dash", "top left"),
        (25, "HIGH threshold (25)", COLORS["HIGH"], "dash", "top left"),
        (30, "EXTREME threshold (30)", COLORS["EXTREME"], "dash", "top left"),
    ]), use_container_width=True)
with tab_dd:
    dd_range = st.radio("Drawdown range", ["6 months", "12 months", "24 months", "36 months"], index=1, horizontal=True, key="dd_range")
    dd_data = filter_months(data, TIME_OPTIONS[dd_range])
    st.plotly_chart(build_indicator_chart(dd_data, "DrawdownPct", "Drawdown %", [
        (-5, "-5% reference", COLORS["LOW"], "dash", "bottom left"),
        (-10, "EXTREME drawdown threshold (-10%)", COLORS["EXTREME"], "dash", "bottom left"),
        (-15, "-15% reference", COLORS["HIGH"], "dot", "bottom left"),
    ]), use_container_width=True)

with st.expander("Signal Calendar", expanded=False):
    st.caption("Compact calendar-style view of historical daily signals. Latest month shown first.")
    calendar_range = st.radio("Calendar range", ["3 months", "6 months", "12 months"], index=0, horizontal=True, key="calendar_range")
    cal_data = filter_months(data, TIME_OPTIONS[calendar_range]).copy()
    legend_cols = st.columns(5)
    for i, (key, label) in enumerate(legend_labels):
        text_color = "white" if key in {"HIGH", "EXTREME"} else "black"
        legend_cols[i].markdown(
            f"<div style='background:{COLORS[key]};padding:8px;border-radius:6px;text-align:center;color:{text_color};'>{label}</div>",
            unsafe_allow_html=True,
        )
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

st.subheader("History")
history_tab1, history_tab2 = st.tabs(["Signal Days Only", "Full History"])
table_base = data.reset_index().copy().rename(columns={"index": "Date"})
table_base["DisplaySignal"] = table_base["Signal"].apply(format_display_signal)
table_base = table_base.sort_values("Date", ascending=False)
with history_tab1:
    signal_table = table_base[table_base["Signal"] != "NONE"].copy()
    signal_table["Date"] = pd.to_datetime(signal_table["Date"]).dt.strftime("%d-%m-%Y")
    signal_table = signal_table.rename(columns={"Close": "Price", "Investment": "Suggested Investment", "DisplaySignal": "Signal"})
    signal_table = signal_table[["Date", "Price", "RSI", "VIX", "DrawdownPct", "Signal", "Suggested Investment"]]
    st.dataframe(signal_table.round(2), use_container_width=True)
with history_tab2:
    full_table = table_base.copy()
    full_table["Date"] = pd.to_datetime(full_table["Date"]).dt.strftime("%d-%m-%Y")
    full_table = full_table.rename(columns={"Close": "Price", "Investment": "Suggested Investment", "DisplaySignal": "Signal"})
    full_table = full_table[["Date", "Price", "RSI", "VIX", "DrawdownPct", "Signal", "Suggested Investment"]]
    st.dataframe(full_table.round(2), use_container_width=True)
