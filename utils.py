
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

TOP_MARKETS = [
    "India", "China", "USA", "United States", "United Kingdom",
    "Bangladesh", "Australia", "Germany", "France", "Sri Lanka", "Japan"
]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip().lower())).strip("_")
        for c in out.columns
    ]
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()
    return out


def parse_month(value) -> float:
    if pd.isna(value):
        return np.nan
    txt = str(value).strip().lower()
    return MONTH_MAP.get(txt[:3], np.nan)


def parse_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"-": np.nan, "nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )


def parse_bs_year_to_ad(year_value) -> float:
    if pd.isna(year_value):
        return np.nan
    txt = str(year_value).strip()
    match = re.search(r"(\d{4})", txt)
    if not match:
        return np.nan
    bs_year = int(match.group(1))
    return bs_year - 57  # assignment assumption


def parse_fiscal_year_start(value) -> float:
    if pd.isna(value):
        return np.nan
    txt = str(value).strip()
    match = re.search(r"(\d{4})", txt)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d{2})/(\d{2})", txt)
    if match:
        yy = int(match.group(1))
        return float(2000 + yy if yy <= 30 else 1900 + yy)
    return np.nan


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], k: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if s.notna().sum() < 8:
            out[col] = s
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            out[col] = s
            continue
        lower, upper = q1 - k * iqr, q3 + k * iqr
        out[col] = s.clip(lower=lower, upper=upper)
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(df)
    out = out.drop_duplicates().copy()

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})

    if "year" in out.columns:
        year_str = out["year"].astype(str).str.strip()
        out["ad_year"] = np.nan
        if year_str.str.contains("/").any():
            fy = year_str.str.contains(r"^\d{4}/\d{2,4}$", regex=True, na=False)
            bs = year_str.str.contains(r"^\d{4}(?:/\d{2,4})?$", regex=True, na=False) & (
                parse_numeric_series(out["year"]).fillna(0) > 2050
            )
            out.loc[fy, "ad_year"] = out.loc[fy, "year"].apply(parse_fiscal_year_start).astype(float).to_numpy()
            out.loc[bs, "ad_year"] = out.loc[bs, "year"].apply(parse_bs_year_to_ad).astype(float).to_numpy()
            remaining = out["ad_year"].isna()
            out.loc[remaining, "ad_year"] = parse_numeric_series(out.loc[remaining, "year"]).astype(float).to_numpy()
        else:
            yr = parse_numeric_series(out["year"]).astype(float)
            out["ad_year"] = np.where(yr > 2050, yr - 57, yr)

    if "metric" in out.columns:
        out["metric"] = out["metric"].astype(str).str.strip().str.lower()
    if "type" in out.columns:
        out["type"] = out["type"].astype(str).str.strip().str.lower()
    if "category" in out.columns:
        out["category"] = out["category"].astype(str).str.strip()
    if "nationality" in out.columns:
        out["nationality"] = out["nationality"].astype(str).str.strip()
    if "month" in out.columns:
        out["month_num"] = out["month"].apply(parse_month)
        if "ad_year" in out.columns:
            out["date"] = pd.to_datetime(
                dict(year=out["ad_year"].fillna(2000).astype(int), month=out["month_num"].fillna(1).astype(int), day=1),
                errors="coerce",
            )

    for col in out.columns:
        col_lower = col.lower()
        if col_lower in {"value", "visitors", "count", "hotel_count", "bed_count", "total", "percent"}:
            out[col] = parse_numeric_series(out[col])

    numeric_candidates = []
    for col in out.columns:
        if out[col].dtype == "object" and col not in {"month", "metric", "type", "category", "region", "nationality", "details", "entry_point", "hotel_type", "types_of_course", "conservation_area"}:
            converted = parse_numeric_series(out[col])
            if converted.notna().mean() >= 0.8:
                out[col] = converted
                numeric_candidates.append(col)
        elif pd.api.types.is_numeric_dtype(out[col]):
            numeric_candidates.append(col)

    # Keep raw values intact; outlier treatment is handled only in analysis-specific views.
    return out


def load_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return {f.stem: clean_data(read_csv_flexible(f)) for f in csv_files}


def prepare_master_tables(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    arrivals_core = data["1.tourist-arrivals-and-average-length-of-stay-in-nepal-19642024"].copy()
    metric_pivot = arrivals_core.pivot_table(index="year", columns="metric", values="value", aggfunc="first").reset_index()
    type_split = (
        arrivals_core[arrivals_core["metric"].isin(["by_air", "by_land"])]
        .pivot_table(index=["year", "metric"], columns="type", values="value", aggfunc="first")
        .reset_index()
    )
    transport_pivot = type_split.pivot(index="year", columns="metric", values="number").reset_index()
    transport_pct = type_split.pivot(index="year", columns="metric", values="percent").reset_index()

    annual_total = data["5.tourist-arrivals-by-year-19962024"].copy()
    annual_total["third_country"] = parse_numeric_series(annual_total["third_country"])
    annual_total["total_arrivals"] = annual_total["third_country"] + annual_total["indian"]

    annual = annual_total.merge(metric_pivot, on="year", how="left", suffixes=("", "_metric"))
    annual = annual.merge(transport_pivot.rename(columns={"by_air": "arrivals_by_air", "by_land": "arrivals_by_land"}), on="year", how="left")
    annual = annual.merge(transport_pct.rename(columns={"by_air": "air_share_pct", "by_land": "land_share_pct"}), on="year", how="left")
    annual["avg_stay_days"] = annual["avg_stay"]
    annual["annual_growth_pct"] = annual["annual_growth"]
    annual["air_dominance_ratio"] = annual["arrivals_by_air"] / annual["total_arrivals"]
    tables["annual_arrivals"] = annual.sort_values("year").reset_index(drop=True)

    monthly_total = data["2.tourist-arrivals-by-month-19952024"].copy()
    monthly_foreign = data["3.tourist-arrivals-by-month-excluding-indian-citizens-19952024"].copy()
    monthly_indian_air = data["4.indian-tourist-arrivals-by-month-by-air-19952024"].copy()

    monthly = monthly_total[["year", "month", "month_num", "date", "value"]].rename(columns={"value": "total_monthly_arrivals"})
    monthly = monthly.merge(
        monthly_foreign[["year", "month", "value"]].rename(columns={"value": "foreign_monthly_arrivals"}),
        on=["year", "month"],
        how="left",
    )
    monthly = monthly.merge(
        monthly_indian_air[["year", "month", "value"]].rename(columns={"value": "indian_air_monthly_arrivals"}),
        on=["year", "month"],
        how="left",
    )
    monthly["estimated_indian_arrivals"] = monthly["total_monthly_arrivals"] - monthly["foreign_monthly_arrivals"]
    monthly["foreign_share_pct"] = monthly["foreign_monthly_arrivals"] / monthly["total_monthly_arrivals"] * 100
    monthly["season"] = pd.cut(
        monthly["month_num"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["Winter", "Spring", "Monsoon", "Autumn", "Winter"],
        include_lowest=True,
        ordered=False,
    ).astype(str)
    monthly["rolling_3m_arrivals"] = monthly.sort_values("date").groupby("year")["total_monthly_arrivals"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    tables["monthly_arrivals"] = monthly.sort_values("date").reset_index(drop=True)

    nationality = data["8.tourist-arrivals-by-major-nationalities-20142024"].copy()
    nationality = nationality.rename(columns={"value": "arrivals", "percent": "arrival_pct_share"})
    fx = data["18foreign-exchange-earnings-from-tourism-fy-2001-022024-25"].copy()
    fx_total = (
        fx[fx["metric"].str.contains("total_foreign_exchange_earnings", case=False, na=False)]
        .groupby("ad_year", as_index=False)["value"].sum()
        .rename(columns={"ad_year": "year", "value": "fx_earnings_npr_million"})
    )
    nationality = nationality.merge(
        fx_total[["year", "fx_earnings_npr_million"]], left_on="year", right_on="year", how="left"
    )
    year_totals = tables["annual_arrivals"][["year", "total_arrivals"]].rename(columns={"total_arrivals": "annual_total_arrivals"})
    nationality = nationality.merge(year_totals, on="year", how="left")
    nationality["estimated_revenue_share_npr_million"] = nationality["fx_earnings_npr_million"] * nationality["arrival_pct_share"] / 100
    nationality["estimated_spend_index"] = nationality["estimated_revenue_share_npr_million"] / nationality["arrivals"].replace(0, np.nan)
    nationality["is_high_value_market"] = np.where(nationality["arrival_pct_share"] < 8, "Niche/High-value", "Mass Market")
    tables["nationality_yearly"] = nationality.sort_values(["year", "arrivals"], ascending=[True, False]).reset_index(drop=True)

    nationality_monthly = data["9.tourist-arrival-by-major-nationalities-month-20202024"].copy()
    nationality_monthly = nationality_monthly.rename(columns={"value": "arrivals", "total": "annual_nationality_total"})
    nationality_monthly["year"] = nationality_monthly["year"].astype("Int64")
    nationality_monthly["date"] = pd.to_datetime(
        dict(
            year=nationality_monthly["year"].fillna(2000).astype(int),
            month=nationality_monthly["month_num"].fillna(1).astype(int),
            day=1,
        ),
        errors="coerce",
    )
    nationality_monthly["monthly_share_pct"] = nationality_monthly["arrivals"] / nationality_monthly.groupby(["year", "month"])["arrivals"].transform("sum") * 100
    tables["nationality_monthly"] = nationality_monthly.sort_values("date").reset_index(drop=True)

    purpose = data["11.tourist-arrival-by-purpose-of-visit-19932024"].copy()
    purpose = purpose.rename(columns={"value": "arrivals", "category": "visit_purpose"})
    if "ad_year" in purpose.columns:
        purpose["year"] = purpose["ad_year"].fillna(purpose["year"])
    purpose["purpose_share_pct"] = purpose["arrivals"] / purpose.groupby("year")["arrivals"].transform("sum") * 100
    tables["purpose_yearly"] = purpose.sort_values(["year", "arrivals"], ascending=[True, False]).reset_index(drop=True)

    trekkers = data["12number-of-trekkers-in-different-trekking-areas-20012024"].copy()
    trekkers = trekkers.rename(columns={"value": "trekkers", "region": "trek_region"})
    if "ad_year" in trekkers.columns:
        trekkers["year"] = trekkers["ad_year"].fillna(trekkers["year"])
    trekkers["region_share_pct"] = trekkers["trekkers"] / trekkers.groupby("year")["trekkers"].transform("sum") * 100
    tables["trekkers_yearly"] = trekkers.sort_values(["year", "trekkers"], ascending=[True, False]).reset_index(drop=True)

    parks = data["13number-of-foreign-visitors-to-national-parks-and-conservation-areascleaned"].copy()
    parks = parks.rename(columns={"visitors": "foreign_visitors", "conservation_area": "park_name"})
    if "ad_year" in parks.columns:
        parks["year"] = parks["ad_year"].fillna(parks["year"])
    parks["park_share_pct"] = parks["foreign_visitors"] / parks.groupby("year")["foreign_visitors"].transform("sum") * 100
    tables["parks_yearly"] = parks.sort_values(["year", "foreign_visitors"], ascending=[True, False]).reset_index(drop=True)

    pashu = data["14.monthly-visitors-excluding-indian-citizens-to-pashupatinath-20102024"].copy()
    pashu["site"] = "Pashupatinath (Foreign)"
    pashu = pashu.rename(columns={"value": "visitors"})
    if "ad_year" in pashu.columns:
        pashu["year"] = pashu["ad_year"].fillna(pashu["year"])

    lumbini_india = data["15.indian-visitors-to-lumbini-20142024"].copy()
    lumbini_india["site"] = "Lumbini (Indian)"
    lumbini_india = lumbini_india.rename(columns={"value": "visitors"})
    if "ad_year" in lumbini_india.columns:
        lumbini_india["year"] = lumbini_india["ad_year"].fillna(lumbini_india["year"])

    lumbini_other = data["16.third-country-visitors-to-lumbini-excluding-indian-citizens-20142024"].copy()
    lumbini_other["site"] = "Lumbini (Third-country)"
    lumbini_other = lumbini_other.rename(columns={"value": "visitors"})
    if "ad_year" in lumbini_other.columns:
        lumbini_other["year"] = lumbini_other["ad_year"].fillna(lumbini_other["year"])

    site_visitors = pd.concat([pashu, lumbini_india, lumbini_other], ignore_index=True, sort=False)
    site_visitors["date"] = pd.to_datetime(
        dict(year=site_visitors["year"].fillna(2000).astype(int), month=site_visitors["month_num"].fillna(1).astype(int), day=1),
        errors="coerce",
    )
    tables["site_visitors_monthly"] = site_visitors[["site", "year", "month", "month_num", "date", "visitors"]].sort_values("date")

    flights_int = data["20.international-flight-and-passenger-movement-20132024"].copy()
    flights_dom = data["21.domestic-flight-passenger-movement-tia-20132024"].copy()

    int_pivot = flights_int.pivot_table(index="year", columns=["metric", "type"], values="value", aggfunc="first")
    int_pivot.columns = [f"intl_{a.lower()}_{b.lower()}" for a, b in int_pivot.columns]
    int_pivot = int_pivot.reset_index()

    dom_pivot = flights_dom.pivot_table(index="year", columns=["metric", "type"], values="value", aggfunc="first")
    dom_pivot.columns = [f"dom_{a.lower()}_{b.lower()}" for a, b in dom_pivot.columns]
    dom_pivot = dom_pivot.reset_index()

    flights = int_pivot.merge(dom_pivot, on="year", how="outer")
    flights["air_connectivity_index"] = (
        flights.get("intl_flight_movement_total", 0).fillna(0)
        + flights.get("dom_flight_movement_dep", 0).fillna(0)
        + flights.get("dom_flight_movement_arr", 0).fillna(0)
    )
    tables["flights_yearly"] = flights.sort_values("year").reset_index(drop=True)

    hotels = data["23.tourist-standard-hotel-registered-in-nepal-2061-2081"].copy()
    hotels["year"] = hotels["ad_year"]
    hotels_agg = hotels.groupby("year", as_index=False)[["hotel_count", "bed_count"]].sum()
    hotels_agg["beds_per_hotel"] = hotels_agg["bed_count"] / hotels_agg["hotel_count"]
    tables["hotels_yearly"] = hotels_agg

    industries = data["24.-tourist-industries-and-guide-in-nepal"].copy()
    industries["year"] = industries["ad_year"]
    industries["value"] = parse_numeric_series(industries["value"])
    industry_pivot = industries.pivot_table(index="year", columns="category", values="value", aggfunc="sum").reset_index()
    industry_pivot.columns = [str(c).lower().replace(" ", "_").replace("/", "_") for c in industry_pivot.columns]
    tables["industry_yearly"] = industry_pivot

    education = data["25.-tourism-and-hotel-management-related-education-and-trainings"].copy()
    education["year"] = education["ad_year"]
    education = education.rename(columns={"types_of_course": "course_type", "value": "graduates"})
    education["graduates"] = parse_numeric_series(education["graduates"])
    tables["education_yearly"] = education.sort_values(["year", "graduates"], ascending=[True, False]).reset_index(drop=True)

    complaints = data["26.complaints-registered-in-tourist-police-office-2020-2024"].copy()
    complaints = complaints.rename(columns={"value": "complaints", "details": "complaint_type"})
    if "ad_year" in complaints.columns:
        complaints["year"] = complaints["ad_year"].fillna(complaints["year"])
    complaint_totals = complaints.groupby("year", as_index=False)["complaints"].sum()
    complaint_totals["complaints_per_100k_arrivals"] = np.nan
    tables["complaints_yearly"] = complaints.sort_values(["year", "complaints"], ascending=[True, False]).reset_index(drop=True)

    parks_totals = parks.groupby("year", as_index=False)["foreign_visitors"].sum().rename(columns={"foreign_visitors": "park_visitors"})
    trekkers_totals = trekkers.groupby("year", as_index=False)["trekkers"].sum()
    purpose_holiday = (
        purpose[purpose["visit_purpose"].str.contains("holiday|pleasure|recreation", case=False, na=False)]
        .groupby("year", as_index=False)["arrivals"].sum()
        .rename(columns={"arrivals": "holiday_arrivals"})
    )

    master_yearly = (
        tables["annual_arrivals"]
        .merge(fx_total.rename(columns={"value": "fx_earnings_npr_million"}), left_on="year", right_on="year", how="left")
        .merge(flights[["year", "air_connectivity_index"]], on="year", how="left")
        .merge(hotels_agg[["year", "hotel_count", "bed_count"]], on="year", how="left")
        .merge(parks_totals, on="year", how="left")
        .merge(trekkers_totals, on="year", how="left")
        .merge(purpose_holiday, on="year", how="left")
        .merge(complaint_totals[["year", "complaints"]], on="year", how="left")
    )
    master_yearly["yield_per_arrival"] = master_yearly["fx_earnings_npr_million"] / master_yearly["total_arrivals"].replace(0, np.nan)
    master_yearly["park_pressure_index"] = master_yearly["park_visitors"] / master_yearly["bed_count"].replace(0, np.nan)
    master_yearly["trekking_share_pct"] = master_yearly["trekkers"] / master_yearly["total_arrivals"].replace(0, np.nan) * 100
    master_yearly["holiday_share_pct"] = master_yearly["holiday_arrivals"] / master_yearly["total_arrivals"].replace(0, np.nan) * 100
    master_yearly["complaints_per_100k_arrivals"] = master_yearly["complaints"] / master_yearly["total_arrivals"].replace(0, np.nan) * 100000
    master_yearly["bed_capacity_per_1000_arrivals"] = master_yearly["bed_count"] / master_yearly["total_arrivals"].replace(0, np.nan) * 1000
    tables["master_yearly"] = master_yearly.sort_values("year").reset_index(drop=True)

    return tables


def calculate_kpis(master_yearly: pd.DataFrame) -> Dict[str, float]:
    df = master_yearly.dropna(subset=["year"]).sort_values("year").copy()
    latest = df.iloc[-1]
    first_valid = df.dropna(subset=["total_arrivals"]).iloc[0]
    periods = max(1, int(latest["year"] - first_valid["year"]))
    cagr = ((latest["total_arrivals"] / first_valid["total_arrivals"]) ** (1 / periods) - 1) * 100 if first_valid["total_arrivals"] > 0 else np.nan
    return {
        "latest_year": int(latest["year"]),
        "latest_total_arrivals": float(latest["total_arrivals"]),
        "latest_fx_earnings_npr_million": float(latest["fx_earnings_npr_million"]) if pd.notna(latest["fx_earnings_npr_million"]) else np.nan,
        "latest_avg_stay_days": float(latest["avg_stay_days"]) if pd.notna(latest["avg_stay_days"]) else np.nan,
        "latest_yield_per_arrival": float(latest["yield_per_arrival"]) if pd.notna(latest["yield_per_arrival"]) else np.nan,
        "arrival_cagr_pct": float(cagr),
        "peak_arrival_year": int(df.loc[df["total_arrivals"].idxmax(), "year"]),
        "peak_arrival_value": float(df["total_arrivals"].max()),
        "mean_air_share_pct": float(df["air_share_pct"].mean()),
        "mean_holiday_share_pct": float(df["holiday_share_pct"].mean()),
    }


# =========================
# Matplotlib / Seaborn EDA
# =========================
def plot_line_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str = "", xlabel: str = ""):
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df[x], df[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel or x.replace("_", " ").title())
    ax.set_ylabel(ylabel or y.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    # Format y-axis to show values in thousands with 'K' suffix
    def thousands_formatter(x, pos):
        return f'{x/1e3:.1f}K'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    # Hide the default offset text
    ax.yaxis.offsetText.set_visible(False)
    plt.tight_layout()
    return fig


def plot_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str = "", xlabel: str = "", top_n: int | None = None):
    from matplotlib.ticker import FuncFormatter
    plot_df = df.copy()
    if top_n:
        plot_df = plot_df.nlargest(top_n, y)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_df, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x.replace("_", " ").title())
    ax.set_ylabel(ylabel or y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    # Format y-axis to show values in thousands with 'K' suffix
    def thousands_formatter(x, pos):
        return f'{x/1e3:.1f}K'
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    # Hide the default offset text
    ax.yaxis.offsetText.set_visible(False)
    plt.tight_layout()
    return fig


def plot_histogram(df: pd.DataFrame, column: str, title: str, bins: int = 25):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    plt.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x=x, y=y, ax=ax, palette="tab10")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame, columns: List[str], title: str):
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# =========================
# Plotly advanced visuals
# =========================
def plot_multilayer_arrivals_fx(master_yearly: pd.DataFrame) -> go.Figure:
    df = master_yearly[master_yearly["year"] >= 2000].sort_values("year").copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["total_arrivals"], mode="lines+markers", name="Total Arrivals", yaxis="y1",
        line=dict(width=2.5), marker=dict(size=7)
    ))
    fig.add_trace(go.Bar(
        x=df["year"], y=df["fx_earnings_npr_million"], name="FX Earnings (NPR mn)", opacity=0.45, yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["yield_per_arrival"], mode="lines+markers", name="Yield per Arrival", yaxis="y3",
        line=dict(width=2.5, dash="dot"), marker=dict(size=7)
    ))
    fig.update_layout(
        title=dict(text="Tourism Demand, Earnings, and Yield", x=0.5, xanchor="center", y=0.95, yanchor="top", font=dict(size=20)),
        xaxis=dict(title=dict(text="Year", font=dict(size=14)), range=[1999, df["year"].max() + 0.5], tickfont=dict(size=12)),
        yaxis=dict(title=dict(text="Arrivals", font=dict(size=14)), tickfont=dict(size=12)),
        yaxis2=dict(
            title=dict(text="FX Earnings (NPR mn)", font=dict(color="#ff7f0e", size=14)),
            tickfont=dict(color="#ff7f0e", size=12),
            overlaying="y",
            side="right",
            position=0.92,
        ),
        yaxis3=dict(
            title=dict(text="Yield per Arrival", font=dict(color="#2ca02c", size=14)),
            tickfont=dict(color="#2ca02c", size=12),
            overlaying="y",
            side="right",
            anchor="free",
            position=0.98,
        ),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=12)),
        margin=dict(l=70, r=110, t=100, b=70),
        template="plotly_white",
        height=620,
        width=1080,
        hovermode="x unified",
    )
    return fig


def plot_faceted_top_markets(nationality_monthly: pd.DataFrame, top_n: int = 6) -> go.Figure:
    latest_year = int(nationality_monthly["year"].dropna().max())
    top_markets = (
        nationality_monthly[nationality_monthly["year"] == latest_year]
        .groupby("nationality", as_index=False)["arrivals"].sum()
        .nlargest(top_n, "arrivals")["nationality"]
        .tolist()
    )
    df = nationality_monthly[nationality_monthly["nationality"].isin(top_markets)].copy()
    fig = px.line(
        df.sort_values("date"),
        x="date",
        y="arrivals",
        facet_col="nationality",
        facet_col_wrap=3,
        title=f"Faceted Monthly Arrivals for Top {top_n} Markets",
        labels={"arrivals": "Monthly Arrivals", "date": "Month"},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None)
    fig.update_layout(template="plotly_white", height=650)
    return fig


def plot_interactive_market_comparison(nationality_yearly: pd.DataFrame) -> go.Figure:
    df = nationality_yearly.copy()
    markets = df.groupby("nationality")["arrivals"].sum().sort_values(ascending=False).head(8).index.tolist()
    df = df[df["nationality"].isin(markets)].sort_values(["nationality", "year"])

    fig = go.Figure()
    for i, market in enumerate(markets):
        sub = df[df["nationality"] == market]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["arrivals"], mode="lines+markers", name=market, visible=(i == 0)
        ))

    buttons = []
    for i, market in enumerate(markets):
        visible = [False] * len(markets)
        visible[i] = True
        buttons.append(
            dict(
                label=market,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Yearly Arrivals Trend - {market}"}
                ],
            )
        )

    buttons.insert(
        0,
        dict(
            label="All Top Markets",
            method="update",
            args=[{"visible": [True] * len(markets)}, {"title": "Yearly Arrivals Trend - All Top Markets"}],
        )
    )

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=1.02, y=1, xanchor="left", yanchor="top")],
        title="Yearly Arrivals Trend - All Top Markets",
        xaxis_title="Year",
        yaxis_title="Arrivals",
        template="plotly_white",
        height=500,
    )
    return fig


def filter_by_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    year_col = "year" if "year" in df.columns else "ad_year"
    return df[(df[year_col] >= start_year) & (df[year_col] <= end_year)].copy()


def build_project(data_dir: str | Path = DEFAULT_DATA_DIR) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, float]]:
    raw = load_data(data_dir)
    tables = prepare_master_tables(raw)
    kpis = calculate_kpis(tables["master_yearly"])
    return raw, tables, kpis
