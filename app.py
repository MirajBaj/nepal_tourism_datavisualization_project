
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import build_project

st.set_page_config(page_title="Nepal Tourism Decision Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data(show_spinner=False)
def get_data():
    return build_project(DATA_DIR)


raw_data, tables, base_kpis = get_data()

master = tables["master_yearly"].copy()
monthly = tables["monthly_arrivals"].copy()
nationality_yearly = tables["nationality_yearly"].copy()
nationality_monthly = tables["nationality_monthly"].copy()
purpose_yearly = tables["purpose_yearly"].copy()
parks_yearly = tables["parks_yearly"].copy()
site_visitors = tables["site_visitors_monthly"].copy()

st.title("Nepal Tourism Decision Dashboard")
st.caption("SDG 8.9 focus: maximize sustainable tourism growth, yield, and resilience using 26 tourism datasets.")

with st.sidebar:
    st.header("Filters")

    year_min = int(master["year"].min())
    year_max = int(master["year"].max())
    selected_years = st.slider("Year Range", min_value=year_min, max_value=year_max, value=(2014, year_max))

    available_markets = sorted(nationality_yearly["nationality"].dropna().unique().tolist())
    default_markets = [m for m in ["India", "USA", "China", "United Kingdom", "Australia"] if m in available_markets][:5]
    selected_markets = st.multiselect(
        "Markets",
        options=available_markets,
        default=default_markets if default_markets else available_markets[:5],
    )

    available_purposes = sorted(purpose_yearly["visit_purpose"].dropna().unique().tolist())
    selected_purposes = st.multiselect(
        "Visit Purposes",
        options=available_purposes,
        default=available_purposes[:4],
    )

    available_parks = sorted(parks_yearly["park_name"].dropna().unique().tolist())
    selected_parks = st.multiselect(
        "Protected Areas",
        options=available_parks,
        default=available_parks[:5],
    )

master_f = master[(master["year"] >= selected_years[0]) & (master["year"] <= selected_years[1])].copy()
monthly_f = monthly[(monthly["year"] >= selected_years[0]) & (monthly["year"] <= selected_years[1])].copy()
nationality_y_f = nationality_yearly[
    (nationality_yearly["year"] >= selected_years[0]) &
    (nationality_yearly["year"] <= selected_years[1]) &
    (nationality_yearly["nationality"].isin(selected_markets))
].copy()
nationality_m_f = nationality_monthly[
    (nationality_monthly["year"] >= selected_years[0]) &
    (nationality_monthly["year"] <= selected_years[1]) &
    (nationality_monthly["nationality"].isin(selected_markets))
].copy()
purpose_f = purpose_yearly[
    (purpose_yearly["year"] >= selected_years[0]) &
    (purpose_yearly["year"] <= selected_years[1]) &
    (purpose_yearly["visit_purpose"].isin(selected_purposes))
].copy()
parks_f = parks_yearly[
    (parks_yearly["year"] >= selected_years[0]) &
    (parks_yearly["year"] <= selected_years[1]) &
    (parks_yearly["park_name"].isin(selected_parks))
].copy()
site_f = site_visitors[(site_visitors["year"] >= selected_years[0]) & (site_visitors["year"] <= selected_years[1])].copy()

latest = master_f.sort_values("year").iloc[-1]
prev = master_f.sort_values("year").iloc[-2] if len(master_f) > 1 else latest

st.markdown("## Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Arrivals", f"{latest['total_arrivals']:,.0f}", delta=f"{latest['total_arrivals'] - prev['total_arrivals']:,.0f}")
k2.metric("Avg Stay (days)", f"{latest['avg_stay_days']:.1f}" if pd.notna(latest["avg_stay_days"]) else "NA")
k3.metric("Air Share %", f"{latest['air_share_pct']:.1f}%" if pd.notna(latest["air_share_pct"]) else "NA")
k4.metric("Complaints / 100k", f"{latest['complaints_per_100k_arrivals']:.1f}" if pd.notna(latest["complaints_per_100k_arrivals"]) else "NA")

with st.container():
    c1, c2 = st.columns((1.5, 1))
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=master_f["year"], y=master_f["total_arrivals"], mode="lines+markers", name="Arrivals"))
        fig.add_trace(go.Bar(x=master_f["year"], y=master_f["fx_earnings_npr_million"], name="FX Earnings", opacity=0.35, yaxis="y2"))
        fig.add_trace(go.Scatter(x=master_f["year"], y=master_f["yield_per_arrival"], mode="lines+markers", name="Yield / Arrival", yaxis="y3"))
        fig.update_layout(
            title="Multi-layer Demand, Earnings, and Yield",
            template="plotly_white",
            height=500,
            yaxis=dict(title="Arrivals"),
            yaxis2=dict(title="FX Earnings (NPR mn)", overlaying="y", side="right"),
            yaxis3=dict(title="Yield", overlaying="y", side="right", position=0.94),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        scatter_df = master_f.dropna(subset=["air_connectivity_index", "total_arrivals"]).copy()
        bubble = px.scatter(
            scatter_df,
            x="air_connectivity_index",
            y="total_arrivals",
            size="bed_count",
            color="holiday_share_pct",
            hover_name="year",
            title="Air Connectivity vs Arrivals vs Capacity",
            labels={"air_connectivity_index": "Connectivity Index", "total_arrivals": "Total Arrivals"},
        )
        bubble.update_layout(template="plotly_white", height=500)
        st.plotly_chart(bubble, use_container_width=True)

st.markdown("## Trends")
t1, t2 = st.columns(2)

with t1:
    monthly_line = px.line(
        monthly_f.sort_values("date"),
        x="date",
        y=["total_monthly_arrivals", "foreign_monthly_arrivals", "rolling_3m_arrivals"],
        title="Monthly Arrivals Trend",
        labels={"value": "Arrivals", "date": "Month", "variable": "Series"},
    )
    monthly_line.update_layout(template="plotly_white", height=420)
    st.plotly_chart(monthly_line, use_container_width=True)

with t2:
    if not nationality_m_f.empty:
        facet = px.line(
            nationality_m_f.sort_values("date"),
            x="date",
            y="arrivals",
            facet_col="nationality",
            facet_col_wrap=2,
            title="Faceted Trend by Selected Markets",
        )
        facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        facet.update_yaxes(matches=None)
        facet.update_layout(template="plotly_white", height=420)
        st.plotly_chart(facet, use_container_width=True)

st.markdown("## Analysis")
a1, a2 = st.columns(2)

with a1:
    purpose_latest = purpose_f[purpose_f["year"] == purpose_f["year"].max()].sort_values("arrivals", ascending=False)
    purpose_chart = px.bar(
        purpose_latest,
        x="visit_purpose",
        y="arrivals",
        color="purpose_share_pct",
        title=f"Visit Purpose Mix ({int(purpose_latest['year'].max())})" if not purpose_latest.empty else "Visit Purpose Mix",
    )
    purpose_chart.update_layout(template="plotly_white", height=420, xaxis_title="")
    st.plotly_chart(purpose_chart, use_container_width=True)

with a2:
    park_latest = parks_f[parks_f["year"] == parks_f["year"].max()].sort_values("foreign_visitors", ascending=False)
    park_chart = px.bar(
        park_latest,
        x="park_name",
        y="foreign_visitors",
        color="park_share_pct",
        title=f"Protected Area Load ({int(park_latest['year'].max())})" if not park_latest.empty else "Protected Area Load",
    )
    park_chart.update_layout(template="plotly_white", height=420, xaxis_title="")
    st.plotly_chart(park_chart, use_container_width=True)

st.markdown("## Advanced Insights")
i1, i2 = st.columns(2)

with i1:
    heatmap_cols = [
        c for c in [
            "total_arrivals", "fx_earnings_npr_million", "air_connectivity_index",
            "bed_count", "park_visitors", "trekkers", "holiday_share_pct",
            "complaints_per_100k_arrivals", "yield_per_arrival"
        ] if c in master_f.columns
    ]
    corr = master_f[heatmap_cols].corr(numeric_only=True)
    heat = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", aspect="auto")
    heat.update_layout(template="plotly_white", height=420)
    st.plotly_chart(heat, use_container_width=True)

with i2:
    compare = nationality_y_f.sort_values(["nationality", "year"])
    trend = px.line(
        compare,
        x="year",
        y="arrivals",
        color="nationality",
        markers=True,
        title="Market Comparison",
    )
    trend.update_layout(template="plotly_white", height=420)
    st.plotly_chart(trend, use_container_width=True)

with st.container():
    st.markdown("### Religious Destination Demand")
    site_chart = px.line(
        site_f.sort_values("date"),
        x="date",
        y="visitors",
        color="site",
        title="Pashupatinath and Lumbini Visitor Trends",
    )
    site_chart.update_layout(template="plotly_white", height=420)
    st.plotly_chart(site_chart, use_container_width=True)

st.markdown("### Analytical use-case")
st.info(
    "Decision question: which source markets and tourism segments should Nepal prioritize to improve arrival growth, spending yield, and protected-area sustainability under SDG 8.9?"
)
