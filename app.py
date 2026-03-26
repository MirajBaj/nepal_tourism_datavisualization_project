
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    build_project,
    prepare_revenue_map_data,
    plot_revenue_contribution_choropleth,
    prepare_arrival_map_data,
    plot_arrival_choropleth,
)

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
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Latest Arrivals", f"{latest['total_arrivals']:,.0f}", delta=f"{latest['total_arrivals'] - prev['total_arrivals']:,.0f}")
k2_metric_text = "NA"
k2_delta = None
revenue_rows = master_f[master_f["fx_earnings_npr_million"].notna()].sort_values("year")
if not revenue_rows.empty:
    latest_rev = revenue_rows.iloc[-1]
    prev_rev = revenue_rows.iloc[-2] if len(revenue_rows) > 1 else latest_rev
    k2_metric_text = f"{latest_rev['fx_earnings_npr_million']:,.0f}"
    k2_delta = f"{latest_rev['fx_earnings_npr_million'] - prev_rev['fx_earnings_npr_million']:,.0f}"
k2.metric("Total Revenue (NPR mn)", k2_metric_text, delta=k2_delta)
k3_text = "NA"
k3_delta = None
avg_rows = master_f[master_f["avg_stay_days"].notna()].sort_values("year")
if not avg_rows.empty:
    latest_avg = avg_rows.iloc[-1]
    prev_avg = avg_rows.iloc[-2] if len(avg_rows) > 1 else latest_avg
    k3_text = f"{latest_avg['avg_stay_days']:.1f}"
    k3_delta = float(latest_avg["avg_stay_days"] - prev_avg["avg_stay_days"])
    k3_delta = round(k3_delta, 1)
    if abs(k3_delta) < 1e-9:
        k3_delta = 0.0
k3.metric("Avg Stay (days)", k3_text, delta=k3_delta)

k4_text = "NA"
k4_delta = None
air_rows = master_f[master_f["air_share_pct"].notna()].sort_values("year")
if not air_rows.empty:
    latest_air = air_rows.iloc[-1]
    prev_air = air_rows.iloc[-2] if len(air_rows) > 1 else latest_air
    k4_text = f"{latest_air['air_share_pct']:.1f}%"
    k4_delta = float(latest_air["air_share_pct"] - prev_air["air_share_pct"])
    k4_delta = round(k4_delta, 1)
    if abs(k4_delta) < 1e-9:
        k4_delta = 0.0
k4.metric("Air Share %", k4_text, delta=k4_delta)

k5_text = "NA"
k5_delta = None
compl_rows = master_f[master_f["complaints_per_100k_arrivals"].notna()].sort_values("year")
if not compl_rows.empty:
    latest_comp = compl_rows.iloc[-1]
    prev_comp = compl_rows.iloc[-2] if len(compl_rows) > 1 else latest_comp
    k5_text = f"{latest_comp['complaints_per_100k_arrivals']:.1f}"
    k5_delta = float(latest_comp["complaints_per_100k_arrivals"] - prev_comp["complaints_per_100k_arrivals"])
    k5_delta = round(k5_delta, 1)
    if abs(k5_delta) < 1e-9:
        k5_delta = 0.0
k5.metric("Complaints / 100k", k5_text, delta=k5_delta)
revenue_per_tourist_text = "NA"
revenue_per_tourist_delta = None
yield_rows = master_f[master_f["yield_per_arrival"].notna()].sort_values("year")
if not yield_rows.empty:
    latest_yield = yield_rows.iloc[-1]
    prev_yield = yield_rows.iloc[-2] if len(yield_rows) > 1 else latest_yield
    # `yield_per_arrival` is computed from `fx_earnings_npr_million`, so it's in NPR-millions per tourist.
    revenue_per_tourist_npr = float(latest_yield["yield_per_arrival"]) * 1_000_000
    revenue_per_tourist_prev_npr = float(prev_yield["yield_per_arrival"]) * 1_000_000
    revenue_per_tourist_text = f"{revenue_per_tourist_npr:,.0f}"
    revenue_per_tourist_delta = f"{revenue_per_tourist_npr - revenue_per_tourist_prev_npr:,.0f}"
k6.metric("Revenue per Tourist (NPR)", revenue_per_tourist_text, delta=revenue_per_tourist_delta)

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
    # For the pie chart we want the full purpose mix for the selected year range,
    # independent of the sidebar "Visit Purposes" multi-select (which is used by other charts).
    purpose_pie = purpose_yearly[
        (purpose_yearly["year"] >= selected_years[0])
        & (purpose_yearly["year"] <= selected_years[1])
    ].copy()
    used_year = int(purpose_pie["year"].max()) if not purpose_pie.empty else None
    purpose_latest = (
        purpose_pie[purpose_pie["year"] == used_year].copy() if used_year is not None else purpose_pie
    )

    if not purpose_latest.empty:
        # Merge "Conv./ Conf." into "Business" for a clearer composition.
        purpose_latest["visit_purpose"] = purpose_latest["visit_purpose"].astype(str).str.strip()
        purpose_latest["visit_purpose"] = np.where(
            purpose_latest["visit_purpose"] == "Conv./ Conf.",
            "Business",
            purpose_latest["visit_purpose"],
        )

        # Re-sum arrivals after merging.
        purpose_latest = (
            purpose_latest.groupby("visit_purpose", as_index=False)["arrivals"].sum()
        )
        purpose_latest = purpose_latest[purpose_latest["arrivals"] > 0].copy()

    # Top 5 purposes + Others, but ensure "Business" slice is always present (if available).
    TOP_N = 5
    if purpose_latest is not None and not purpose_latest.empty:
        sorted_df = purpose_latest.sort_values("arrivals", ascending=False).copy()
        business_row = sorted_df[sorted_df["visit_purpose"] == "Business"].copy()

        if not business_row.empty:
            top_others = sorted_df[sorted_df["visit_purpose"] != "Business"].head(TOP_N - 1).copy()
            top = pd.concat([business_row, top_others], ignore_index=True)
        else:
            top = sorted_df.head(TOP_N).copy()

        selected = set(top["visit_purpose"].astype(str).tolist())
        remaining = sorted_df[~sorted_df["visit_purpose"].astype(str).isin(selected)]
        others_sum = float(remaining["arrivals"].sum()) if not remaining.empty else 0.0

        if others_sum > 0:
            top = pd.concat(
                [
                    top,
                    pd.DataFrame({"visit_purpose": ["Others"], "arrivals": [others_sum]}),
                ],
                ignore_index=True,
            )

        pie = px.pie(
            top,
            values="arrivals",
            names="visit_purpose",
            title=f"Purpose of Visit Mix (Full Pie) - {used_year}",
        )
        pie.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Arrivals: %{value:,.0f}<br>Share: %{percent}",
        )
        pie.update_layout(template="plotly_white", height=420)
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("No purpose-of-visit data available for the current filters.")

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
    st.markdown("### Tourist Arrivals by Country (Nationality)")
    map_year = selected_years[1]
    map_data = prepare_arrival_map_data(nationality_yearly, map_year)
    if not map_data.empty:
        fig = plot_arrival_choropleth(map_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected year.")

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


