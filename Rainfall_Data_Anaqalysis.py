from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    KeepInFrame,
)


DATA_FILE = Path("daily-rainfall-data-district-level.csv")
OUTPUT_DIR = Path("outputs")
CHART_DIR = OUTPUT_DIR / "charts"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"
ASSET_DIR = Path("assets")
STATE_GEOJSON = ASSET_DIR / "india_state.geojson"
DISTRICT_GEOJSON = ASSET_DIR / "india_district_modern.geojson"
STATE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
)
DISTRICT_GEOJSON_URL = (
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson"
)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "blue_light": "#A3BEFA",
    "gold": "#B8A037",
    "gold_light": "#FFE15B",
    "orange": "#CC6F47",
    "orange_light": "#F0986E",
    "olive": "#71B436",
    "olive_light": "#A3D576",
    "pink": "#BD569B",
    "pink_light": "#F390CA",
}

SEASONS = {
    1: "Winter",
    2: "Winter",
    3: "Pre-monsoon",
    4: "Pre-monsoon",
    5: "Pre-monsoon",
    6: "Pre-monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Monsoon",
    11: "Post-monsoon",
    12: "Post-monsoon",
}

SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


@dataclass
class ChartSpec:
    path: Path
    title: str
    subtitle: str
    insight: str
    section: str


def ensure_dirs() -> None:
    for folder in [CHART_DIR, TABLE_DIR, REPORT_DIR, ASSET_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    for folder, pattern in [(CHART_DIR, "*.png"), (TABLE_DIR, "*.csv"), (REPORT_DIR, "*.pdf")]:
        for old_file in folder.glob(pattern):
            try:
                old_file.unlink()
            except PermissionError:
                pass


def writable_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    try:
        with path.open("ab"):
            return path
    except PermissionError:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def slugify(value: str, limit: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return value[:limit] or "chart"


def setup_plot_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "savefig.facecolor": TOKENS["surface"],
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )


def add_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, 76, break_long_words=False)
    subtitle = textwrap.fill(subtitle, 118, break_long_words=False)
    ax.set_title("")
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    top = max(0.56, 0.75 - 0.035 * (title_lines - 1) - 0.024 * (subtitle_lines - 1))
    fig.subplots_adjust(top=top, left=0.16, right=0.96, bottom=0.16)
    left = ax.get_position().x0
    fig.text(left, 0.97, title, ha="left", va="top", fontsize=14, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.905, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_chart(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def pct(value: float) -> str:
    if pd.isna(value) or math.isinf(value):
        return "n/a"
    return f"{value:.1f}%"


def num(value: float, digits: int = 1) -> str:
    if pd.isna(value) or math.isinf(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def weighted_corr(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    valid = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return float("nan")
    if valid[x_col].std(ddof=0) == 0 or valid[y_col].std(ddof=0) == 0:
        return float("nan")
    return float(valid[x_col].corr(valid[y_col]))


def build_six_month_forecasts(national_monthly: pd.DataFrame, state_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = pd.to_datetime(national_monthly["date"].max())
    future_dates = pd.date_range(max_date + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    national_rows = []
    for target_date in future_dates:
        month = int(target_date.month)
        history = national_monthly[national_monthly["month"] == month]
        national_rows.append(
            {
                "date": target_date,
                "year": int(target_date.year),
                "month": month,
                "season": SEASONS[month],
                "forecast_actual_mm": float(history["actual_mm"].median()),
                "forecast_normal_mm": float(history["normal_mm"].median()),
                "lower_actual_mm": float(history["actual_mm"].quantile(0.10)),
                "upper_actual_mm": float(history["actual_mm"].quantile(0.90)),
                "history_points": int(history["actual_mm"].count()),
                "model": "seasonal_calendar_median",
            }
        )
    national_forecast = pd.DataFrame(national_rows)
    national_forecast["forecast_departure_pct"] = np.where(
        national_forecast["forecast_normal_mm"] > 0,
        (national_forecast["forecast_actual_mm"] - national_forecast["forecast_normal_mm"]) / national_forecast["forecast_normal_mm"] * 100,
        np.nan,
    )
    national_forecast["risk_band"] = np.select(
        [
            national_forecast["forecast_departure_pct"] <= -20,
            national_forecast["forecast_departure_pct"] >= 20,
        ],
        ["dry_watch", "wet_watch"],
        default="normal_range",
    )
    state_rows = []
    for state, part in state_monthly.groupby("state_name"):
        for target_date in future_dates:
            month = int(target_date.month)
            history = part[part["month"] == month]
            if history.empty:
                continue
            state_rows.append(
                {
                    "state_name": state,
                    "date": target_date,
                    "year": int(target_date.year),
                    "month": month,
                    "season": SEASONS[month],
                    "forecast_actual_mm": float(history["actual_mm"].median()),
                    "forecast_normal_mm": float(history["normal_mm"].median()),
                    "lower_actual_mm": float(history["actual_mm"].quantile(0.10)),
                    "upper_actual_mm": float(history["actual_mm"].quantile(0.90)),
                    "history_points": int(history["actual_mm"].count()),
                    "model": "state_calendar_median",
                }
            )
    state_forecast = pd.DataFrame(state_rows)
    state_forecast["forecast_departure_pct"] = np.where(
        state_forecast["forecast_normal_mm"] > 0,
        (state_forecast["forecast_actual_mm"] - state_forecast["forecast_normal_mm"]) / state_forecast["forecast_normal_mm"] * 100,
        np.nan,
    )
    state_forecast["risk_band"] = np.select(
        [
            state_forecast["forecast_departure_pct"] <= -20,
            state_forecast["forecast_departure_pct"] >= 20,
        ],
        ["dry_watch", "wet_watch"],
        default="normal_range",
    )
    return national_forecast, state_forecast


def load_and_aggregate(csv_path: Path, chunksize: int = 350_000) -> dict[str, pd.DataFrame]:
    usecols = [
        "date",
        "state_code",
        "state_name",
        "district_code",
        "district_name",
        "actual",
        "rfs",
        "normal",
        "deviation",
    ]
    annual_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    weekly_parts: list[pd.DataFrame] = []
    state_daily_parts: list[pd.DataFrame] = []
    district_daily_parts: list[pd.DataFrame] = []
    weekday_parts: list[pd.DataFrame] = []
    day_of_month_parts: list[pd.DataFrame] = []
    quality_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        for col in ["actual", "rfs", "normal", "deviation"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["year"] = chunk["date"].dt.year
        chunk["month"] = chunk["date"].dt.month
        chunk["day_of_month"] = chunk["date"].dt.day
        chunk["weekday"] = chunk["date"].dt.day_name()
        chunk["weekday_num"] = chunk["date"].dt.weekday
        iso_calendar = chunk["date"].dt.isocalendar()
        chunk["iso_year"] = iso_calendar.year.astype("Int64")
        chunk["week"] = iso_calendar.week.astype("Int64")
        chunk["season"] = chunk["month"].map(SEASONS)
        chunk["actual_filled"] = chunk["actual"].fillna(0)
        chunk["normal_filled"] = chunk["normal"].fillna(0)
        chunk["wet_day"] = (chunk["actual"].fillna(0) >= 2.5).astype(int)
        chunk["heavy_day"] = (chunk["actual"].fillna(0) >= 64.5).astype(int)
        chunk["very_heavy_day"] = (chunk["actual"].fillna(0) >= 115.6).astype(int)
        chunk["extreme_day"] = (chunk["actual"].fillna(0) >= 204.5).astype(int)

        group_cols = ["state_name", "district_name", "year"]
        annual_parts.append(
            chunk.groupby(group_cols, dropna=False)
            .agg(
                actual_mm=("actual_filled", "sum"),
                normal_mm=("normal_filled", "sum"),
                wet_days=("wet_day", "sum"),
                heavy_days=("heavy_day", "sum"),
                very_heavy_days=("very_heavy_day", "sum"),
                extreme_days=("extreme_day", "sum"),
                observations=("date", "count"),
                missing_actual=("actual", lambda s: int(s.isna().sum())),
                missing_normal=("normal", lambda s: int(s.isna().sum())),
            )
            .reset_index()
        )

        monthly_parts.append(
            chunk.groupby(["state_name", "district_name", "year", "month", "season"], dropna=False)
            .agg(
                actual_mm=("actual_filled", "sum"),
                normal_mm=("normal_filled", "sum"),
                wet_days=("wet_day", "sum"),
                heavy_days=("heavy_day", "sum"),
                observations=("date", "count"),
            )
            .reset_index()
        )

        weekly_parts.append(
            chunk.groupby(["state_name", "district_name", "iso_year", "week"], dropna=False)
            .agg(
                actual_mm=("actual_filled", "sum"),
                normal_mm=("normal_filled", "sum"),
                wet_days=("wet_day", "sum"),
                heavy_days=("heavy_day", "sum"),
                observations=("date", "count"),
            )
            .reset_index()
        )

        state_daily_parts.append(
            chunk.groupby(["date", "state_name"], dropna=False)
            .agg(actual_mm=("actual_filled", "sum"), normal_mm=("normal_filled", "sum"), districts=("district_name", "nunique"))
            .reset_index()
        )

        district_daily_parts.append(
            chunk.groupby(["date", "state_name", "district_name"], dropna=False)
            .agg(actual_mm=("actual_filled", "sum"), normal_mm=("normal_filled", "sum"))
            .reset_index()
        )

        weekday_parts.append(
            chunk.groupby(["state_name", "weekday", "weekday_num"], dropna=False)
            .agg(
                actual_mm=("actual_filled", "sum"),
                normal_mm=("normal_filled", "sum"),
                wet_days=("wet_day", "sum"),
                heavy_days=("heavy_day", "sum"),
                observations=("date", "count"),
            )
            .reset_index()
        )

        day_of_month_parts.append(
            chunk.groupby(["state_name", "day_of_month"], dropna=False)
            .agg(
                actual_mm=("actual_filled", "sum"),
                normal_mm=("normal_filled", "sum"),
                wet_days=("wet_day", "sum"),
                heavy_days=("heavy_day", "sum"),
                observations=("date", "count"),
            )
            .reset_index()
        )

        quality_parts.append(
            pd.DataFrame(
                {
                    "rows": [len(chunk)],
                    "missing_actual": [int(chunk["actual"].isna().sum())],
                    "missing_normal": [int(chunk["normal"].isna().sum())],
                    "missing_deviation": [int(chunk["deviation"].isna().sum())],
                    "start_date": [chunk["date"].min()],
                    "end_date": [chunk["date"].max()],
                    "states": [chunk["state_name"].nunique()],
                    "districts": [chunk[["state_name", "district_name"]].drop_duplicates().shape[0]],
                }
            )
        )

    annual = pd.concat(annual_parts, ignore_index=True)
    annual = (
        annual.groupby(["state_name", "district_name", "year"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    annual["departure_pct"] = np.where(annual["normal_mm"] > 0, (annual["actual_mm"] - annual["normal_mm"]) / annual["normal_mm"] * 100, np.nan)

    monthly = pd.concat(monthly_parts, ignore_index=True)
    monthly = (
        monthly.groupby(["state_name", "district_name", "year", "month", "season"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    monthly["departure_pct"] = np.where(monthly["normal_mm"] > 0, (monthly["actual_mm"] - monthly["normal_mm"]) / monthly["normal_mm"] * 100, np.nan)

    weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly = (
        weekly.groupby(["state_name", "district_name", "iso_year", "week"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    weekly["departure_pct"] = np.where(weekly["normal_mm"] > 0, (weekly["actual_mm"] - weekly["normal_mm"]) / weekly["normal_mm"] * 100, np.nan)

    state_daily = pd.concat(state_daily_parts, ignore_index=True)
    state_daily = (
        state_daily.groupby(["date", "state_name"], dropna=False)
        .agg(actual_mm=("actual_mm", "sum"), normal_mm=("normal_mm", "sum"), districts=("districts", "max"))
        .reset_index()
    )
    state_daily["year"] = state_daily["date"].dt.year
    state_daily["month"] = state_daily["date"].dt.month
    national_daily = (
        state_daily.groupby("date", dropna=False)
        .agg(actual_mm=("actual_mm", "sum"), normal_mm=("normal_mm", "sum"), districts=("districts", "sum"))
        .reset_index()
    )
    national_daily["year"] = national_daily["date"].dt.year
    national_daily["month"] = national_daily["date"].dt.month

    district_daily = pd.concat(district_daily_parts, ignore_index=True)
    district_daily = (
        district_daily.groupby(["date", "state_name", "district_name"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )

    weekday = pd.concat(weekday_parts, ignore_index=True)
    weekday = (
        weekday.groupby(["state_name", "weekday", "weekday_num"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    weekday["actual_per_observation"] = np.where(weekday["observations"] > 0, weekday["actual_mm"] / weekday["observations"], np.nan)
    weekday["normal_per_observation"] = np.where(weekday["observations"] > 0, weekday["normal_mm"] / weekday["observations"], np.nan)
    weekday["wet_day_rate"] = np.where(weekday["observations"] > 0, weekday["wet_days"] / weekday["observations"] * 100, np.nan)

    day_of_month = pd.concat(day_of_month_parts, ignore_index=True)
    day_of_month = (
        day_of_month.groupby(["state_name", "day_of_month"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    day_of_month["actual_per_observation"] = np.where(day_of_month["observations"] > 0, day_of_month["actual_mm"] / day_of_month["observations"], np.nan)
    day_of_month["normal_per_observation"] = np.where(day_of_month["observations"] > 0, day_of_month["normal_mm"] / day_of_month["observations"], np.nan)
    day_of_month["wet_day_rate"] = np.where(day_of_month["observations"] > 0, day_of_month["wet_days"] / day_of_month["observations"] * 100, np.nan)

    quality = pd.concat(quality_parts, ignore_index=True)
    quality_summary = pd.DataFrame(
        {
            "rows": [int(quality["rows"].sum())],
            "missing_actual": [int(quality["missing_actual"].sum())],
            "missing_normal": [int(quality["missing_normal"].sum())],
            "missing_deviation": [int(quality["missing_deviation"].sum())],
            "start_date": [quality["start_date"].min()],
            "end_date": [quality["end_date"].max()],
        }
    )

    state_annual = (
        annual.groupby(["state_name", "year"], dropna=False)
        .agg(
            actual_mm=("actual_mm", "sum"),
            normal_mm=("normal_mm", "sum"),
            wet_days=("wet_days", "sum"),
            heavy_days=("heavy_days", "sum"),
            very_heavy_days=("very_heavy_days", "sum"),
            extreme_days=("extreme_days", "sum"),
            observations=("observations", "sum"),
            missing_actual=("missing_actual", "sum"),
            missing_normal=("missing_normal", "sum"),
            districts=("district_name", "nunique"),
        )
        .reset_index()
    )
    state_annual["departure_pct"] = np.where(state_annual["normal_mm"] > 0, (state_annual["actual_mm"] - state_annual["normal_mm"]) / state_annual["normal_mm"] * 100, np.nan)
    state_annual["avg_district_actual_mm"] = np.where(state_annual["districts"] > 0, state_annual["actual_mm"] / state_annual["districts"], np.nan)
    state_annual["avg_district_normal_mm"] = np.where(state_annual["districts"] > 0, state_annual["normal_mm"] / state_annual["districts"], np.nan)

    state_monthly = (
        monthly.groupby(["state_name", "year", "month", "season"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    state_monthly["departure_pct"] = np.where(state_monthly["normal_mm"] > 0, (state_monthly["actual_mm"] - state_monthly["normal_mm"]) / state_monthly["normal_mm"] * 100, np.nan)

    state_weekly = (
        weekly.groupby(["state_name", "iso_year", "week"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    state_weekly["departure_pct"] = np.where(state_weekly["normal_mm"] > 0, (state_weekly["actual_mm"] - state_weekly["normal_mm"]) / state_weekly["normal_mm"] * 100, np.nan)
    state_weekly["date"] = pd.to_datetime(state_weekly["iso_year"].astype(str) + "-W" + state_weekly["week"].astype(str).str.zfill(2) + "-1", format="%G-W%V-%u", errors="coerce")

    national_annual = (
        state_annual.groupby("year", dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    national_annual["departure_pct"] = np.where(national_annual["normal_mm"] > 0, (national_annual["actual_mm"] - national_annual["normal_mm"]) / national_annual["normal_mm"] * 100, np.nan)

    national_monthly = (
        state_monthly.groupby(["year", "month", "season"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    national_monthly["date"] = pd.to_datetime(national_monthly["year"].astype(str) + "-" + national_monthly["month"].astype(str) + "-01")
    national_monthly["departure_pct"] = np.where(national_monthly["normal_mm"] > 0, (national_monthly["actual_mm"] - national_monthly["normal_mm"]) / national_monthly["normal_mm"] * 100, np.nan)

    national_weekly = (
        state_weekly.groupby(["iso_year", "week"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    national_weekly["departure_pct"] = np.where(national_weekly["normal_mm"] > 0, (national_weekly["actual_mm"] - national_weekly["normal_mm"]) / national_weekly["normal_mm"] * 100, np.nan)
    national_weekly["date"] = pd.to_datetime(national_weekly["iso_year"].astype(str) + "-W" + national_weekly["week"].astype(str).str.zfill(2) + "-1", format="%G-W%V-%u", errors="coerce")

    national_weekday = (
        weekday.groupby(["weekday", "weekday_num"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
        .sort_values("weekday_num")
    )
    national_weekday["actual_per_observation"] = np.where(national_weekday["observations"] > 0, national_weekday["actual_mm"] / national_weekday["observations"], np.nan)
    national_weekday["normal_per_observation"] = np.where(national_weekday["observations"] > 0, national_weekday["normal_mm"] / national_weekday["observations"], np.nan)
    national_weekday["wet_day_rate"] = np.where(national_weekday["observations"] > 0, national_weekday["wet_days"] / national_weekday["observations"] * 100, np.nan)

    national_day_of_month = (
        day_of_month.groupby("day_of_month", dropna=False)
        .sum(numeric_only=True)
        .reset_index()
        .sort_values("day_of_month")
    )
    national_day_of_month["actual_per_observation"] = np.where(national_day_of_month["observations"] > 0, national_day_of_month["actual_mm"] / national_day_of_month["observations"], np.nan)
    national_day_of_month["normal_per_observation"] = np.where(national_day_of_month["observations"] > 0, national_day_of_month["normal_mm"] / national_day_of_month["observations"], np.nan)
    national_day_of_month["wet_day_rate"] = np.where(national_day_of_month["observations"] > 0, national_day_of_month["wet_days"] / national_day_of_month["observations"] * 100, np.nan)

    national_season = (
        national_monthly.groupby("season", dropna=False)
        .agg(actual_mm=("actual_mm", "mean"), normal_mm=("normal_mm", "mean"), heavy_days=("heavy_days", "mean"))
        .reset_index()
    )
    national_season["season"] = pd.Categorical(national_season["season"], categories=SEASON_ORDER, ordered=True)
    national_season = national_season.sort_values("season")
    national_season["departure_pct"] = np.where(national_season["normal_mm"] > 0, (national_season["actual_mm"] - national_season["normal_mm"]) / national_season["normal_mm"] * 100, np.nan)

    national_seasonal_year = (
        state_monthly.groupby(["year", "season"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    national_seasonal_year["season"] = pd.Categorical(national_seasonal_year["season"], categories=SEASON_ORDER, ordered=True)
    national_seasonal_year["departure_pct"] = np.where(national_seasonal_year["normal_mm"] > 0, (national_seasonal_year["actual_mm"] - national_seasonal_year["normal_mm"]) / national_seasonal_year["normal_mm"] * 100, np.nan)

    state_seasonal_year = (
        state_monthly.groupby(["state_name", "year", "season"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    state_seasonal_year["season"] = pd.Categorical(state_seasonal_year["season"], categories=SEASON_ORDER, ordered=True)
    state_seasonal_year["departure_pct"] = np.where(state_seasonal_year["normal_mm"] > 0, (state_seasonal_year["actual_mm"] - state_seasonal_year["normal_mm"]) / state_seasonal_year["normal_mm"] * 100, np.nan)

    district_seasonal_year = (
        monthly.groupby(["state_name", "district_name", "year", "season"], dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )
    district_seasonal_year["season"] = pd.Categorical(district_seasonal_year["season"], categories=SEASON_ORDER, ordered=True)
    district_seasonal_year["departure_pct"] = np.where(district_seasonal_year["normal_mm"] > 0, (district_seasonal_year["actual_mm"] - district_seasonal_year["normal_mm"]) / district_seasonal_year["normal_mm"] * 100, np.nan)

    district_summary = (
        annual.groupby(["state_name", "district_name"], dropna=False)
        .agg(
            actual_mm=("actual_mm", "mean"),
            normal_mm=("normal_mm", "mean"),
            wet_days=("wet_days", "mean"),
            heavy_days=("heavy_days", "mean"),
            very_heavy_days=("very_heavy_days", "mean"),
            extreme_days=("extreme_days", "mean"),
            years=("year", "nunique"),
            observations=("observations", "sum"),
        )
        .reset_index()
    )
    district_summary["departure_pct"] = np.where(district_summary["normal_mm"] > 0, (district_summary["actual_mm"] - district_summary["normal_mm"]) / district_summary["normal_mm"] * 100, np.nan)

    state_summary = (
        state_annual.groupby("state_name", dropna=False)
        .agg(
            actual_mm=("actual_mm", "mean"),
            normal_mm=("normal_mm", "mean"),
            avg_district_actual_mm=("avg_district_actual_mm", "mean"),
            avg_district_normal_mm=("avg_district_normal_mm", "mean"),
            wet_days=("wet_days", "mean"),
            heavy_days=("heavy_days", "mean"),
            very_heavy_days=("very_heavy_days", "mean"),
            extreme_days=("extreme_days", "mean"),
            departure_pct=("departure_pct", "mean"),
            years=("year", "nunique"),
            districts=("districts", "max"),
        )
        .reset_index()
    )
    state_summary["rainfall_normal_corr"] = [
        weighted_corr(state_annual[state_annual["state_name"] == s], "avg_district_actual_mm", "avg_district_normal_mm")
        for s in state_summary["state_name"]
    ]

    national_forecast, state_forecast = build_six_month_forecasts(national_monthly, state_monthly)
    hourly_variation_status = pd.DataFrame(
        [
            {
                "requested_view": "daily_24_hour_variation",
                "status": "not_available_from_source",
                "reason": "The CSV contains date-level daily records but no timestamp, hour, or sub-daily rainfall field.",
                "needed_data": "Add hourly or sub-hourly rainfall observations with state and district identifiers.",
            }
        ]
    )

    tables = {
        "quality_summary": quality_summary,
        "annual_district": annual,
        "monthly_district": monthly,
        "weekly_district": weekly,
        "district_seasonal_year": district_seasonal_year,
        "state_daily": state_daily,
        "district_daily": district_daily,
        "state_annual": state_annual,
        "state_monthly": state_monthly,
        "state_weekly": state_weekly,
        "state_seasonal_year": state_seasonal_year,
        "national_daily": national_daily,
        "national_annual": national_annual,
        "national_monthly": national_monthly,
        "national_weekly": national_weekly,
        "national_seasonal_year": national_seasonal_year,
        "national_weekday": national_weekday,
        "national_day_of_month": national_day_of_month,
        "national_season": national_season,
        "national_forecast_next_six_months": national_forecast,
        "state_forecast_next_six_months": state_forecast,
        "hourly_variation_status": hourly_variation_status,
        "state_weekday": weekday,
        "state_day_of_month": day_of_month,
        "state_summary": state_summary,
        "district_summary": district_summary,
    }
    for name, frame in tables.items():
        if name != "district_daily":
            frame.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    return tables


def chart_line(df: pd.DataFrame, x: str, y_cols: list[str], title: str, subtitle: str, path: Path, labels: list[str] | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    colors_cycle = [TOKENS["blue"], TOKENS["orange"], TOKENS["olive"], TOKENS["pink"], TOKENS["gold"]]
    for i, col in enumerate(y_cols):
        ax.plot(df[x], df[col], color=colors_cycle[i % len(colors_cycle)], linewidth=1.55, marker="o", markersize=2.4, label=(labels or y_cols)[i])
    ax.grid(True, axis="y")
    if np.issubdtype(df[x].dtype, np.datetime64):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}"))
    add_header(fig, ax, title, subtitle)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=min(3, len(y_cols)), borderaxespad=0, fontsize=8.5, handlelength=2.6)
    save_chart(fig, path)
    return path


def chart_bar(df: pd.DataFrame, category: str, value: str, title: str, subtitle: str, path: Path, color: str = "orange", horizontal: bool = True) -> Path:
    plot_df = df.copy()
    fig_height = max(6.2, min(12, 0.35 * len(plot_df) + 2.5))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    fill = TOKENS.get(color, TOKENS["orange"])
    edge = TOKENS["ink"]
    if horizontal:
        ax.barh(plot_df[category], plot_df[value], color=fill, edgecolor=edge, linewidth=0.8)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}"))
    else:
        ax.bar(plot_df[category], plot_df[value], color=fill, edgecolor=edge, linewidth=0.8)
        ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="x" if horizontal else "y")
    add_header(fig, ax, title, subtitle)
    save_chart(fig, path)
    return path


def chart_heatmap(matrix: pd.DataFrame, title: str, subtitle: str, path: Path, label_mode: str = "raw") -> Path:
    fig, ax = plt.subplots(figsize=(12, max(6, min(13, matrix.shape[0] * 0.34 + 2.5))))
    cmap = mcolors.LinearSegmentedColormap.from_list("rainfall_blue", ["#FFFFFF", "#EAF1FE", TOKENS["blue_light"], TOKENS["blue"]])
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
    labels = []
    for c in matrix.columns:
        if label_mode == "month":
            try:
                labels.append(MONTH_NAMES.get(int(c), str(c)))
            except ValueError:
                labels.append(str(c))
        elif label_mode == "week":
            labels.append(f"W{int(c):02d}" if pd.notna(c) else "")
        else:
            labels.append(str(c))
    ax.set_xticks(np.arange(matrix.shape[1]), labels)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.tick_params(axis="both", length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Rainfall mm")
    add_header(fig, ax, title, subtitle)
    save_chart(fig, path)
    return path


def chart_scatter(df: pd.DataFrame, x: str, y: str, label: str, title: str, subtitle: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.scatter(df[x], df[y], s=42, color=TOKENS["blue_light"], edgecolor=TOKENS["blue"], linewidth=0.7, alpha=0.85)
    for _, row in df.nlargest(6, y).iterrows():
        ax.annotate(str(row[label])[:18], (row[x], row[y]), fontsize=7.5, color=TOKENS["muted"])
    ax.grid(True)
    add_header(fig, ax, title, subtitle)
    save_chart(fig, path)
    return path


def chart_forecast(history: pd.DataFrame, forecast: pd.DataFrame, title: str, subtitle: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    recent = history.sort_values("date").tail(24)
    ax.plot(recent["date"], recent["actual_mm"], color=TOKENS["blue"], linewidth=1.7, marker="o", markersize=3.4, label="Observed actual")
    ax.plot(forecast["date"], forecast["forecast_actual_mm"], color=TOKENS["orange"], linewidth=1.9, marker="o", markersize=4, label="Six-month forecast")
    ax.fill_between(
        forecast["date"],
        forecast["lower_actual_mm"],
        forecast["upper_actual_mm"],
        color=TOKENS["orange_light"],
        alpha=0.25,
        label="10th-90th percentile band",
    )
    ax.plot(forecast["date"], forecast["forecast_normal_mm"], color=TOKENS["muted"], linestyle=":", linewidth=1.4, label="Forecast normal")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    add_header(fig, ax, title, subtitle)
    save_chart(fig, path)
    return path


def chart_multiseries_line(df: pd.DataFrame, x: str, y: str, series: str, title: str, subtitle: str, path: Path, max_series: int = 6) -> Path:
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    colors_cycle = [TOKENS["blue"], TOKENS["orange"], TOKENS["olive"], TOKENS["pink"], TOKENS["gold"], TOKENS["blue_light"]]
    series_order = list(df[series].dropna().drop_duplicates())[:max_series]
    for i, value in enumerate(series_order):
        part = df[df[series] == value].sort_values(x)
        ax.plot(part[x], part[y], color=colors_cycle[i % len(colors_cycle)], linewidth=1.35, marker="o", markersize=2.3, label=str(value))
    ax.grid(True, axis="y")
    if np.issubdtype(df[x].dtype, np.datetime64):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}"))
    add_header(fig, ax, title, subtitle)
    legend_cols = min(3, max(1, len(series_order)))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=legend_cols, borderaxespad=0, fontsize=8.0, handlelength=2.5, columnspacing=1.4)
    save_chart(fig, path)
    return path


def create_national_charts(tables: dict[str, pd.DataFrame], charts: list[ChartSpec]) -> None:
    national_daily = tables["national_daily"].sort_values("date")
    national_weekly = tables["national_weekly"].sort_values("date")
    national_annual = tables["national_annual"].sort_values("year")
    national_monthly = tables["national_monthly"].sort_values("date")
    national_seasonal_year = tables["national_seasonal_year"].sort_values(["year", "season"])
    state_summary = tables["state_summary"].sort_values("avg_district_actual_mm", ascending=False)
    district_summary = tables["district_summary"].sort_values("actual_mm", ascending=False)

    path = chart_line(
        national_daily,
        "date",
        ["actual_mm", "normal_mm"],
        "India daily rainfall time series",
        "Daily national rainfall aggregation across reported districts.",
        CHART_DIR / "national_daily_time_series.png",
        ["Actual", "Normal"],
    )
    charts.append(ChartSpec(path, "India daily rainfall time series", "Daily actual vs normal rainfall", "The daily time series exposes short-run spikes, dry spells, and extreme-day clusters that monthly views can hide.", "National time series"))

    path = chart_line(
        national_weekly,
        "date",
        ["actual_mm", "normal_mm"],
        "India weekly rainfall time series",
        "ISO-week national rainfall aggregation across reported districts.",
        CHART_DIR / "national_weekly_time_series.png",
        ["Actual", "Normal"],
    )
    charts.append(ChartSpec(path, "India weekly rainfall time series", "Weekly actual vs normal rainfall", "Weekly time series smooth daily noise while preserving monsoon onset, peak, and withdrawal patterns.", "National time series"))

    path = chart_line(
        national_annual,
        "year",
        ["actual_mm", "normal_mm"],
        "India annual rainfall against normal",
        "District-level daily records aggregated to annual national totals; values are summed rainfall millimetres across reported districts.",
        CHART_DIR / "national_annual_actual_vs_normal.png",
        ["Actual", "Normal"],
    )
    charts.append(ChartSpec(path, "India annual rainfall against normal", "Annual actual vs normal rainfall", "The national annual curve shows whether observed rainfall is persistently above or below the climatological normal.", "National"))

    path = chart_line(
        national_monthly,
        "date",
        ["actual_mm", "normal_mm"],
        "Monthly rainfall time series",
        "Month-by-month national aggregation from 2009 through the latest available month.",
        CHART_DIR / "national_monthly_time_series.png",
        ["Actual", "Normal"],
    )
    charts.append(ChartSpec(path, "Monthly rainfall time series", "Actual vs normal by month", "Monthly aggregation exposes seasonality and outlier wet or dry months more clearly than annual totals.", "National"))

    path = chart_multiseries_line(
        national_seasonal_year,
        "year",
        "actual_mm",
        "season",
        "India seasonal rainfall time series",
        "Seasonal rainfall by year using the requested season definition.",
        CHART_DIR / "national_seasonal_time_series.png",
        max_series=4,
    )
    charts.append(ChartSpec(path, "India seasonal rainfall time series", "Seasonal rainfall by year", "The seasonal time series separates winter, pre-monsoon, monsoon, and post-monsoon movement across years.", "National time series"))

    monthly_profile = national_monthly.groupby("month").agg(actual_mm=("actual_mm", "mean"), normal_mm=("normal_mm", "mean")).reset_index()
    path = chart_line(
        monthly_profile,
        "month",
        ["actual_mm", "normal_mm"],
        "Average monthly rainfall profile",
        "Mean monthly actual and normal rainfall across all available years.",
        CHART_DIR / "national_average_monthly_profile.png",
        ["Actual", "Normal"],
    )
    charts.append(ChartSpec(path, "Average monthly rainfall profile", "Mean actual vs normal by calendar month", "The monsoon months dominate the national water budget, so deviations in June to September carry outsized annual impact.", "National"))

    for metric, title, color in [
        ("avg_district_actual_mm", "Wettest states by average district rainfall", "blue"),
        ("departure_pct", "States with highest average positive departure", "olive"),
        ("heavy_days", "States with most heavy-rainfall days", "orange"),
        ("rainfall_normal_corr", "Actual-to-normal correlation by state", "pink"),
    ]:
        top = state_summary.sort_values(metric, ascending=False).head(20)
        path = chart_bar(top, "state_name", metric, title, "Average across available years; district-level observations are aggregated to state level.", CHART_DIR / f"national_{slugify(title)}.png", color=color)
        charts.append(ChartSpec(path, title, metric, f"{title} identifies where the long-run signal is strongest and where monitoring should be most granular.", "National"))

    for metric, title, color in [
        ("actual_mm", "Wettest districts by average annual rainfall", "blue"),
        ("departure_pct", "Districts with highest positive rainfall departure", "olive"),
        ("heavy_days", "Districts with most heavy-rainfall days", "orange"),
        ("extreme_days", "Districts with most extreme-rainfall days", "pink"),
    ]:
        top = district_summary.dropna(subset=[metric]).sort_values(metric, ascending=False).head(25).copy()
        top["district_label"] = top["district_name"] + ", " + top["state_name"]
        path = chart_bar(top, "district_label", metric, title, "Average annual district metrics across available years.", CHART_DIR / f"district_{slugify(title)}.png", color=color)
        charts.append(ChartSpec(path, title, metric, f"{title} highlights the places where rainfall risk or surplus is most concentrated.", "District"))

    heat = tables["state_monthly"].pivot_table(index="state_name", columns="month", values="actual_mm", aggfunc="mean").fillna(0)
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
    path = chart_heatmap(heat, "State-month rainfall heatmap", "Average monthly actual rainfall by state; darker cells mark wetter months.", CHART_DIR / "state_month_rainfall_heatmap.png", label_mode="month")
    charts.append(ChartSpec(path, "State-month rainfall heatmap", "Average monthly rainfall by state", "The heatmap separates monsoon-dominant regions from areas with meaningful pre- or post-monsoon rainfall.", "State"))

    corr_df = state_summary.dropna(subset=["avg_district_actual_mm", "rainfall_normal_corr"])
    path = chart_scatter(corr_df, "avg_district_actual_mm", "rainfall_normal_corr", "state_name", "Rainfall depth versus actual-normal correlation", "Each point is a state; rainfall depth is annual rainfall divided by reporting district count.", CHART_DIR / "state_depth_vs_correlation.png")
    charts.append(ChartSpec(path, "Rainfall depth versus actual-normal correlation", "Average district rainfall vs actual-normal correlation", "States with high average district rainfall but weaker correlation deserve closer monitoring because normal rainfall is less predictive.", "Correlation"))


def create_state_charts(tables: dict[str, pd.DataFrame], charts: list[ChartSpec], max_states: int | None = None) -> None:
    state_daily = tables["state_daily"]
    state_weekly = tables["state_weekly"]
    state_annual = tables["state_annual"]
    state_monthly = tables["state_monthly"]
    state_seasonal_year = tables["state_seasonal_year"]
    district_summary = tables["district_summary"]
    states = sorted(state_annual["state_name"].dropna().unique())
    if max_states:
        states = states[:max_states]

    for state in states:
        state_slug = slugify(state)
        daily = state_daily[state_daily["state_name"] == state].sort_values("date")
        weekly = state_weekly[state_weekly["state_name"] == state].sort_values("date")
        annual = state_annual[state_annual["state_name"] == state].sort_values("year")
        if annual.empty:
            continue
        if not daily.empty:
            path = chart_line(
                daily,
                "date",
                ["actual_mm", "normal_mm"],
                f"{state}: daily rainfall time series",
                "Daily actual and normal rainfall aggregated across reporting districts in the state.",
                CHART_DIR / f"state_{state_slug}_daily_time_series.png",
                ["Actual", "Normal"],
            )
            charts.append(ChartSpec(path, f"{state}: daily rainfall time series", "State daily actual vs normal rainfall", f"{state}'s daily time series shows short-run rainfall spikes and dry spells.", "State time series"))

        if not weekly.empty:
            path = chart_line(
                weekly,
                "date",
                ["actual_mm", "normal_mm"],
                f"{state}: weekly rainfall time series",
                "ISO-week actual and normal rainfall aggregated across reporting districts in the state.",
                CHART_DIR / f"state_{state_slug}_weekly_time_series.png",
                ["Actual", "Normal"],
            )
            charts.append(ChartSpec(path, f"{state}: weekly rainfall time series", "State weekly actual vs normal rainfall", f"{state}'s weekly time series smooths daily volatility while preserving seasonal build-up.", "State time series"))

        path = chart_line(
            annual,
            "year",
            ["actual_mm", "normal_mm"],
            f"{state}: annual rainfall against normal",
            "Actual and normal rainfall aggregated from all reporting districts in the state.",
            CHART_DIR / f"state_{state_slug}_annual_actual_vs_normal.png",
            ["Actual", "Normal"],
        )
        dep = annual["departure_pct"].mean()
        charts.append(ChartSpec(path, f"{state}: annual rainfall against normal", "State annual actual vs normal rainfall", f"{state} averages {pct(dep)} annual departure from normal across the observed period.", "State"))

        monthly_series = state_monthly[state_monthly["state_name"] == state].copy()
        monthly_series["date"] = pd.to_datetime(monthly_series["year"].astype(str) + "-" + monthly_series["month"].astype(str) + "-01")
        path = chart_line(
            monthly_series.sort_values("date"),
            "date",
            ["actual_mm", "normal_mm"],
            f"{state}: monthly rainfall time series",
            "Month-by-month actual and normal rainfall aggregated across reporting districts in the state.",
            CHART_DIR / f"state_{state_slug}_monthly_time_series.png",
            ["Actual", "Normal"],
        )
        charts.append(ChartSpec(path, f"{state}: monthly rainfall time series", "State monthly actual vs normal rainfall", f"{state}'s monthly time series shows recurring seasonality and exceptional wet or dry months.", "State time series"))

        seasonal_series = state_seasonal_year[state_seasonal_year["state_name"] == state].sort_values(["year", "season"])
        if not seasonal_series.empty:
            path = chart_multiseries_line(
                seasonal_series,
                "year",
                "actual_mm",
                "season",
                f"{state}: seasonal rainfall time series",
                "Seasonal rainfall by year using the requested season definition.",
                CHART_DIR / f"state_{state_slug}_seasonal_time_series.png",
                max_series=4,
            )
            charts.append(ChartSpec(path, f"{state}: seasonal rainfall time series", "State seasonal rainfall by year", f"{state}'s seasonal time series separates winter, pre-monsoon, monsoon, and post-monsoon movement.", "State time series"))

        profile = monthly_series.groupby("month").agg(actual_mm=("actual_mm", "mean"), normal_mm=("normal_mm", "mean")).reset_index()
        path = chart_line(
            profile,
            "month",
            ["actual_mm", "normal_mm"],
            f"{state}: average monthly rainfall profile",
            "Mean month-of-year actual and normal rainfall across available years.",
            CHART_DIR / f"state_{state_slug}_monthly_profile.png",
            ["Actual", "Normal"],
        )
        peak_month = int(profile.loc[profile["actual_mm"].idxmax(), "month"]) if not profile.empty else 0
        charts.append(ChartSpec(path, f"{state}: average monthly rainfall profile", "State monthly seasonality", f"{state}'s peak observed rainfall month is {MONTH_NAMES.get(peak_month, 'n/a')}, anchoring the local seasonal risk window.", "State"))

        districts = district_summary[district_summary["state_name"] == state].dropna(subset=["actual_mm"]).sort_values("actual_mm", ascending=False).head(12).copy()
        if len(districts) >= 3:
            path = chart_bar(
                districts,
                "district_name",
                "actual_mm",
                f"{state}: wettest districts",
                "Average annual actual rainfall by district.",
                CHART_DIR / f"state_{state_slug}_wettest_districts.png",
                color="blue",
            )
            top_district = districts.iloc[0]["district_name"]
            charts.append(ChartSpec(path, f"{state}: wettest districts", "District ranking within state", f"{top_district} is the wettest district in {state} on average, so it drives a large share of state rainfall exposure.", "District"))


def create_india_scope_state_time_charts(tables: dict[str, pd.DataFrame], charts: list[ChartSpec]) -> None:
    state_monthly = tables["state_monthly"]
    state_weekly = tables["state_weekly"]
    state_month_matrix = state_monthly.pivot_table(index="state_name", columns="month", values="actual_mm", aggfunc="mean").fillna(0)
    state_month_matrix = state_month_matrix.loc[state_month_matrix.mean(axis=1).sort_values(ascending=False).index]
    path = chart_heatmap(
        state_month_matrix,
        "India scope: state month-wise rainfall",
        "Average monthly rainfall by state; columns are calendar months and rows are states.",
        CHART_DIR / "india_scope_state_month_wise_rainfall.png",
        label_mode="month",
    )
    charts.append(ChartSpec(path, "India scope: state month-wise rainfall", "State month-wise matrix", "India-level month-wise analysis shows how each state contributes to the annual rainfall cycle.", "India state month-wise"))

    state_season_matrix = state_monthly.groupby(["state_name", "season"]).agg(actual_mm=("actual_mm", "mean")).reset_index()
    state_season_matrix["season"] = pd.Categorical(state_season_matrix["season"], categories=SEASON_ORDER, ordered=True)
    state_season_matrix = state_season_matrix.pivot_table(index="state_name", columns="season", values="actual_mm", aggfunc="mean").reindex(columns=SEASON_ORDER).fillna(0)
    state_season_matrix = state_season_matrix.loc[state_season_matrix.mean(axis=1).sort_values(ascending=False).index]
    path = chart_heatmap(
        state_season_matrix,
        "India scope: state season-wise rainfall",
        "Average rainfall by requested season definition for every state.",
        CHART_DIR / "india_scope_state_season_wise_rainfall.png",
    )
    charts.append(ChartSpec(path, "India scope: state season-wise rainfall", "State season-wise matrix", "Season-wise state analysis separates winter, pre-monsoon, monsoon, and post-monsoon rainfall under the requested calendar.", "India state season-wise"))

    state_week_matrix = state_weekly.pivot_table(index="state_name", columns="week", values="actual_mm", aggfunc="mean").fillna(0)
    state_week_matrix = state_week_matrix.loc[state_week_matrix.mean(axis=1).sort_values(ascending=False).index]
    path = chart_heatmap(
        state_week_matrix,
        "India scope: state week-wise rainfall",
        "Average ISO-week rainfall by state; columns are week numbers.",
        CHART_DIR / "india_scope_state_week_wise_rainfall.png",
        label_mode="week",
    )
    charts.append(ChartSpec(path, "India scope: state week-wise rainfall", "State week-wise matrix", "Week-wise state analysis reveals the progression and concentration of rainfall through the year.", "India state week-wise"))


def create_state_scope_district_time_charts(tables: dict[str, pd.DataFrame], charts: list[ChartSpec]) -> None:
    district_daily = tables["district_daily"]
    annual = tables["annual_district"]
    monthly = tables["monthly_district"]
    weekly = tables["weekly_district"]
    seasonal = tables["district_seasonal_year"]
    district_summary = tables["district_summary"]
    for state_name in sorted(monthly["state_name"].dropna().unique()):
        state_slug = slugify(state_name)
        top_districts = (
            district_summary[district_summary["state_name"] == state_name]
            .sort_values("actual_mm", ascending=False)
            .head(5)["district_name"]
            .tolist()
        )
        state_daily = district_daily[(district_daily["state_name"] == state_name) & (district_daily["district_name"].isin(top_districts))].copy()
        if not state_daily.empty:
            path = chart_multiseries_line(
                state_daily.sort_values("date"),
                "date",
                "actual_mm",
                "district_name",
                f"{state_name}: top district daily rainfall time series",
                "Daily rainfall for the wettest districts in the state.",
                CHART_DIR / f"state_scope_{state_slug}_top_district_daily_time_series.png",
                max_series=5,
            )
            charts.append(ChartSpec(path, f"{state_name}: top district daily rainfall time series", "District daily rainfall time series", f"{state_name}'s district daily time series compares short-run rainfall spikes across its wettest districts.", "State district time series"))

        state_month = monthly[monthly["state_name"] == state_name]
        state_month_series = state_month[state_month["district_name"].isin(top_districts)].copy()
        state_month_series["date"] = pd.to_datetime(state_month_series["year"].astype(str) + "-" + state_month_series["month"].astype(str) + "-01")
        if not state_month_series.empty:
            path = chart_multiseries_line(
                state_month_series.sort_values("date"),
                "date",
                "actual_mm",
                "district_name",
                f"{state_name}: top district monthly rainfall time series",
                "Month-by-month rainfall for the wettest districts in the state.",
                CHART_DIR / f"state_scope_{state_slug}_top_district_monthly_time_series.png",
                max_series=5,
            )
            charts.append(ChartSpec(path, f"{state_name}: top district monthly rainfall time series", "District monthly rainfall time series", f"{state_name}'s district monthly time series shows which wettest districts dominate specific months.", "State district time series"))

        state_week_series = weekly[(weekly["state_name"] == state_name) & (weekly["district_name"].isin(top_districts))].copy()
        state_week_series["date"] = pd.to_datetime(state_week_series["iso_year"].astype(str) + "-W" + state_week_series["week"].astype(str).str.zfill(2) + "-1", format="%G-W%V-%u", errors="coerce")
        if not state_week_series.empty:
            path = chart_multiseries_line(
                state_week_series.sort_values("date"),
                "date",
                "actual_mm",
                "district_name",
                f"{state_name}: top district weekly rainfall time series",
                "ISO-week rainfall for the wettest districts in the state.",
                CHART_DIR / f"state_scope_{state_slug}_top_district_weekly_time_series.png",
                max_series=5,
            )
            charts.append(ChartSpec(path, f"{state_name}: top district weekly rainfall time series", "District weekly rainfall time series", f"{state_name}'s district weekly time series shows rainfall progression across the wettest districts.", "State district time series"))

        state_annual_series = annual[(annual["state_name"] == state_name) & (annual["district_name"].isin(top_districts))].copy()
        if not state_annual_series.empty:
            path = chart_multiseries_line(
                state_annual_series.sort_values("year"),
                "year",
                "actual_mm",
                "district_name",
                f"{state_name}: top district yearly rainfall time series",
                "Annual rainfall for the wettest districts in the state.",
                CHART_DIR / f"state_scope_{state_slug}_top_district_yearly_time_series.png",
                max_series=5,
            )
            charts.append(ChartSpec(path, f"{state_name}: top district yearly rainfall time series", "District yearly rainfall time series", f"{state_name}'s district yearly time series compares long-run rainfall movement across the wettest districts.", "State district time series"))

        state_season_series = seasonal[(seasonal["state_name"] == state_name) & (seasonal["district_name"].isin(top_districts))].copy()
        if not state_season_series.empty:
            state_season_totals = state_season_series.groupby(["district_name", "year"]).agg(actual_mm=("actual_mm", "sum")).reset_index()
            path = chart_multiseries_line(
                state_season_totals.sort_values("year"),
                "year",
                "actual_mm",
                "district_name",
                f"{state_name}: top district seasonal-year rainfall time series",
                "Sum of requested seasons by year for the wettest districts in the state.",
                CHART_DIR / f"state_scope_{state_slug}_top_district_seasonal_year_time_series.png",
                max_series=5,
            )
            charts.append(ChartSpec(path, f"{state_name}: top district seasonal-year rainfall time series", "District seasonal-year rainfall time series", f"{state_name}'s district seasonal-year time series compares seasonal rainfall burden across its wettest districts.", "State district time series"))

        month_matrix = state_month.pivot_table(index="district_name", columns="month", values="actual_mm", aggfunc="mean").fillna(0)
        month_matrix = month_matrix.loc[month_matrix.mean(axis=1).sort_values(ascending=False).index]
        path = chart_heatmap(
            month_matrix,
            f"{state_name}: district month-wise rainfall",
            "Average monthly rainfall by district inside the state.",
            CHART_DIR / f"state_scope_{state_slug}_district_month_wise_rainfall.png",
            label_mode="month",
        )
        charts.append(ChartSpec(path, f"{state_name}: district month-wise rainfall", "District month-wise matrix", f"{state_name}'s district month-wise matrix shows which districts drive each calendar month.", "State district month-wise"))

        season_frame = state_month.groupby(["district_name", "season"]).agg(actual_mm=("actual_mm", "mean")).reset_index()
        season_frame["season"] = pd.Categorical(season_frame["season"], categories=SEASON_ORDER, ordered=True)
        season_matrix = season_frame.pivot_table(index="district_name", columns="season", values="actual_mm", aggfunc="mean").reindex(columns=SEASON_ORDER).fillna(0)
        season_matrix = season_matrix.loc[season_matrix.mean(axis=1).sort_values(ascending=False).index]
        path = chart_heatmap(
            season_matrix,
            f"{state_name}: district season-wise rainfall",
            "Average rainfall by requested season definition for districts inside the state.",
            CHART_DIR / f"state_scope_{state_slug}_district_season_wise_rainfall.png",
        )
        charts.append(ChartSpec(path, f"{state_name}: district season-wise rainfall", "District season-wise matrix", f"{state_name}'s season-wise district analysis shows which districts dominate winter, pre-monsoon, monsoon, and post-monsoon rainfall.", "State district season-wise"))

        state_week = weekly[weekly["state_name"] == state_name]
        week_matrix = state_week.pivot_table(index="district_name", columns="week", values="actual_mm", aggfunc="mean").fillna(0)
        week_matrix = week_matrix.loc[week_matrix.mean(axis=1).sort_values(ascending=False).index]
        path = chart_heatmap(
            week_matrix,
            f"{state_name}: district week-wise rainfall",
            "Average ISO-week rainfall by district inside the state.",
            CHART_DIR / f"state_scope_{state_slug}_district_week_wise_rainfall.png",
            label_mode="week",
        )
        charts.append(ChartSpec(path, f"{state_name}: district week-wise rainfall", "District week-wise matrix", f"{state_name}'s week-wise district analysis shows how rainfall moves through districts across the year.", "State district week-wise"))


def create_temporal_variation_charts(tables: dict[str, pd.DataFrame], charts: list[ChartSpec]) -> None:
    national_weekday = tables["national_weekday"].sort_values("weekday_num")
    national_day = tables["national_day_of_month"].sort_values("day_of_month")
    national_season = tables["national_season"].copy()
    national_monthly = tables["national_monthly"].sort_values("date")
    national_forecast = tables["national_forecast_next_six_months"].sort_values("date")
    state_weekday = tables["state_weekday"].copy()
    state_monthly = tables["state_monthly"].copy()
    state_forecast = tables["state_forecast_next_six_months"].copy()

    path = chart_bar(
        national_weekday,
        "weekday",
        "actual_per_observation",
        "Weekday rainfall variation",
        "Average rainfall per district-day observation by weekday; daily records have no hour field.",
        CHART_DIR / "national_weekday_rainfall_variation.png",
        color="olive",
        horizontal=False,
    )
    charts.append(ChartSpec(path, "Weekday rainfall variation", "Average rainfall by weekday", "Weekday differences are descriptive sampling patterns from daily observations, not a causal weekday effect.", "Weekday variation"))

    path = chart_bar(
        national_weekday,
        "weekday",
        "wet_day_rate",
        "Weekday wet-day rate",
        "Percent of district-day records with at least 2.5 mm rainfall by weekday.",
        CHART_DIR / "national_weekday_wet_day_rate.png",
        color="blue",
        horizontal=False,
    )
    charts.append(ChartSpec(path, "Weekday wet-day rate", "Wet-day rate by weekday", "Wet-day rate gives a frequency view that complements rainfall depth.", "Weekday variation"))

    path = chart_line(
        national_day,
        "day_of_month",
        ["actual_per_observation", "normal_per_observation"],
        "Day-of-month rainfall variation",
        "Average rainfall per district-day observation by calendar day of month.",
        CHART_DIR / "national_day_of_month_variation.png",
        ["Actual", "Normal"],
    )
    charts.append(ChartSpec(path, "Day-of-month rainfall variation", "Daily calendar-position view", "Day-of-month variation checks whether month-start or month-end records have visible sampling or rainfall differences.", "Daily variation"))

    path = chart_bar(
        national_season,
        "season",
        "actual_mm",
        "Season-wise rainfall under requested calendar",
        "Winter: Jan-Feb; Pre-monsoon: Mar-Jun; Monsoon: Jul-Oct; Post-monsoon: Nov-Dec.",
        CHART_DIR / "national_requested_season_rainfall.png",
        color="orange",
        horizontal=False,
    )
    charts.append(ChartSpec(path, "Season-wise rainfall under requested calendar", "Average rainfall by requested season", "The requested season mapping shifts June into pre-monsoon and October into monsoon, changing the seasonal read from the earlier meteorological convention.", "Seasonality"))

    path = chart_forecast(
        national_monthly,
        national_forecast,
        "Safe six-month rainfall forecast",
        "Seasonal median forecast for the next six months after the latest source month, with 10th-90th percentile historical bands.",
        CHART_DIR / "national_six_month_forecast.png",
    )
    charts.append(ChartSpec(path, "Safe six-month rainfall forecast", "Seasonal calendar-median forecast", "The forecast is a conservative pattern estimate with uncertainty bands, suitable for planning watchlists rather than precise weather prediction.", "Forecast"))

    state_forecast_total = (
        state_forecast.groupby("state_name")
        .agg(forecast_actual_mm=("forecast_actual_mm", "sum"), forecast_departure_pct=("forecast_departure_pct", "mean"))
        .reset_index()
        .sort_values("forecast_actual_mm", ascending=False)
        .head(20)
    )
    path = chart_bar(
        state_forecast_total,
        "state_name",
        "forecast_actual_mm",
        "Highest state rainfall forecasts over next six months",
        "Sum of six monthly state seasonal-median forecasts.",
        CHART_DIR / "state_six_month_forecast_totals.png",
        color="pink",
    )
    charts.append(ChartSpec(path, "Highest state rainfall forecasts over next six months", "State forecast totals", "State forecast totals identify where near-term rainfall monitoring should remain most active.", "Forecast"))

    weekday_matrix = state_weekday.pivot_table(index="state_name", columns="weekday", values="actual_per_observation", aggfunc="mean")
    weekday_matrix = weekday_matrix.reindex(columns=WEEKDAY_NAMES)
    weekday_matrix = weekday_matrix.loc[weekday_matrix.mean(axis=1).sort_values(ascending=False).index]
    path = chart_heatmap(
        weekday_matrix.fillna(0),
        "State-by-weekday rainfall matrix",
        "Average rainfall per district-day observation; columns use Monday-through-Sunday order.",
        CHART_DIR / "state_weekday_rainfall_matrix.png",
    )
    charts.append(ChartSpec(path, "State-by-weekday rainfall matrix", "State weekday matrix", "The matrix adds micro-detail for weekday patterns without implying sub-daily or hourly behavior.", "Weekday variation"))

    season_matrix = state_monthly.groupby(["state_name", "season"]).agg(actual_mm=("actual_mm", "mean")).reset_index()
    season_matrix["season"] = pd.Categorical(season_matrix["season"], categories=SEASON_ORDER, ordered=True)
    season_matrix = season_matrix.pivot_table(index="state_name", columns="season", values="actual_mm", aggfunc="mean")
    season_matrix = season_matrix.reindex(columns=SEASON_ORDER)
    season_matrix = season_matrix.loc[season_matrix.mean(axis=1).sort_values(ascending=False).index]
    path = chart_heatmap(
        season_matrix.fillna(0),
        "State-by-season rainfall matrix",
        "Average monthly rainfall by requested season definition.",
        CHART_DIR / "state_requested_season_matrix.png",
    )
    charts.append(ChartSpec(path, "State-by-season rainfall matrix", "State seasonal matrix", "The state-season matrix shows where winter, pre-monsoon, monsoon, and post-monsoon rainfall are most differentiated.", "Seasonality"))


def normalize_state_name(name: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    clean = re.sub(r"^the\s+", "", clean)
    replacements = {
        "andaman nicobar": "andaman and nicobar islands",
        "andaman and nicobar": "andaman and nicobar islands",
        "dadra nagar haveli": "dadra and nagar haveli and daman and diu",
        "dadra nagar haveli daman diu": "dadra and nagar haveli and daman and diu",
        "daman diu": "dadra and nagar haveli and daman and diu",
        "delhi": "nct of delhi",
        "jammu kashmir": "jammu and kashmir",
        "orissa": "odisha",
        "pondicherry": "puducherry",
        "uttaranchal": "uttarakhand",
    }
    return replacements.get(clean, clean)


def normalize_district_name(name: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    replacements = {
        "nicobar islands": "nicobars",
        "nicobars": "nicobars",
        "andaman islands": "andaman islands",
        "north middle andaman": "north and middle andaman",
        "north and middle andaman": "north and middle andaman",
        "south andaman": "south andamans",
        "south andamans": "south andamans",
        "ysr": "y s r",
        "spsr nellore": "sri potti sriramulu nellore",
        "banas kantha": "banaskantha",
        "sabar kantha": "sabarkantha",
        "panch mahals": "panchmahal",
        "the dangs": "dang",
        "dangs": "dang",
        "barddhaman": "bardhaman",
        "hugli": "hooghly",
        "koch bihar": "cooch behar",
        "haora": "howrah",
    }
    return replacements.get(clean, clean)


def download_geojson() -> bool:
    if STATE_GEOJSON.exists() and STATE_GEOJSON.stat().st_size > 1000:
        return True
    try:
        print(f"Downloading state GeoJSON from {STATE_GEOJSON_URL}")
        urllib.request.urlretrieve(STATE_GEOJSON_URL, STATE_GEOJSON)
        return STATE_GEOJSON.exists() and STATE_GEOJSON.stat().st_size > 1000
    except Exception as exc:
        print(f"Map boundary download failed: {exc}")
        return False


def download_district_geojson() -> bool:
    if DISTRICT_GEOJSON.exists() and DISTRICT_GEOJSON.stat().st_size > 1000:
        return True
    try:
        print(f"Downloading district GeoJSON from {DISTRICT_GEOJSON_URL}")
        urllib.request.urlretrieve(DISTRICT_GEOJSON_URL, DISTRICT_GEOJSON)
        return DISTRICT_GEOJSON.exists() and DISTRICT_GEOJSON.stat().st_size > 1000
    except Exception as exc:
        print(f"District boundary download failed: {exc}")
        return False


def iter_polygons(geometry: dict) -> Iterable[list[tuple[float, float]]]:
    if not geometry:
        return
    if geometry["type"] == "Polygon":
        for ring in geometry.get("coordinates", []):
            yield [(float(x), float(y)) for x, y in ring]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry.get("coordinates", []):
            for ring in polygon:
                yield [(float(x), float(y)) for x, y in ring]


def feature_state_name(feature: dict) -> str:
    props = feature.get("properties", {})
    for key in ["NAME_1", "ST_NM", "st_nm", "State_Name", "state", "name"]:
        if key in props:
            return str(props[key])
    return ""


def feature_district_name(feature: dict) -> str:
    props = feature.get("properties", {})
    for key in ["NAME_2", "DISTRICT", "district", "dtname", "DIST_NAME", "name"]:
        if key in props:
            return str(props[key])
    return ""


def format_geo_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}E"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}N"))
    ax.grid(True, color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(axis="both", labelsize=8, colors=TOKENS["muted"])


def create_state_map(metric_df: pd.DataFrame, value_col: str, title: str, subtitle: str, path: Path) -> Path | None:
    if not STATE_GEOJSON.exists():
        return None
    with STATE_GEOJSON.open("r", encoding="utf-8") as f:
        geo = json.load(f)
    values = {normalize_state_name(row["state_name"]): row[value_col] for _, row in metric_df.iterrows()}
    valid_values = [v for v in values.values() if pd.notna(v)]
    if not valid_values:
        return None
    norm = mcolors.Normalize(vmin=float(np.nanpercentile(valid_values, 5)), vmax=float(np.nanpercentile(valid_values, 95)))
    cmap = mcolors.LinearSegmentedColormap.from_list("rainfall_map", ["#FFFFFF", "#CEDFFE", TOKENS["blue"], "#2E4780"])
    patches = []
    patch_values = []
    for feature in geo.get("features", []):
        state_key = normalize_state_name(feature_state_name(feature))
        value = values.get(state_key, np.nan)
        for ring in iter_polygons(feature.get("geometry")):
            if len(ring) > 3:
                patches.append(Polygon(ring, closed=True))
                patch_values.append(value)
    if not patches:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    facecolors = [cmap(norm(v)) if pd.notna(v) else "#F4F5F7" for v in patch_values]
    collection = PatchCollection(patches, facecolor=facecolors, edgecolor="#FFFFFF", linewidth=0.35)
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal")
    format_geo_axes(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(value_col.replace("_", " ").title())
    fig.text(0.08, 0.97, textwrap.fill(title, 72), fontsize=15, fontweight="bold", color=TOKENS["ink"], va="top")
    fig.text(0.08, 0.925, textwrap.fill(subtitle, 110), fontsize=9.5, color=TOKENS["muted"], va="top")
    save_chart(fig, path)
    return path


def create_state_district_map(geo: dict, state_name: str, metric_df: pd.DataFrame, value_col: str, title: str, subtitle: str, path: Path) -> Path | None:
    state_key = normalize_state_name(state_name)
    state_features = [feature for feature in geo.get("features", []) if normalize_state_name(feature_state_name(feature)) == state_key]
    if not state_features:
        return None
    values = {normalize_district_name(row["district_name"]): row[value_col] for _, row in metric_df[metric_df["state_name"] == state_name].iterrows()}
    matched_values = []
    patches = []
    patch_values = []
    for feature in state_features:
        district_key = normalize_district_name(feature_district_name(feature))
        value = values.get(district_key, np.nan)
        if pd.notna(value):
            matched_values.append(value)
        for ring in iter_polygons(feature.get("geometry")):
            if len(ring) > 3:
                patches.append(Polygon(ring, closed=True))
                patch_values.append(value)
    if not patches:
        return None
    if not matched_values:
        matched_values = [0.0]
    vmax = float(np.nanpercentile(matched_values, 95)) if len(matched_values) > 1 else float(max(matched_values))
    vmin = float(np.nanpercentile(matched_values, 5)) if len(matched_values) > 1 else 0.0
    if vmax <= vmin:
        vmax = vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list("district_map", ["#FFFFFF", "#FFEDDE", TOKENS["orange_light"], TOKENS["orange"]])
    facecolors = [cmap(norm(v)) if pd.notna(v) else "#F4F5F7" for v in patch_values]
    fig, ax = plt.subplots(figsize=(9.5, 9.2))
    collection = PatchCollection(patches, facecolor=facecolors, edgecolor="#FFFFFF", linewidth=0.45)
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal")
    format_geo_axes(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(value_col.replace("_", " ").title())
    fig.text(0.08, 0.965, textwrap.fill(title, 70), fontsize=14, fontweight="bold", color=TOKENS["ink"], va="top")
    fig.text(0.08, 0.92, textwrap.fill(subtitle, 105), fontsize=9.2, color=TOKENS["muted"], va="top")
    save_chart(fig, path)
    return path


def create_map_charts(tables: dict[str, pd.DataFrame], charts: list[ChartSpec], allow_download: bool) -> None:
    if allow_download:
        download_geojson()
        download_district_geojson()
    if not STATE_GEOJSON.exists():
        state_available = False
    else:
        state_available = True
    state_summary = tables["state_summary"]
    if state_available:
        map_specs = [
            ("avg_district_actual_mm", "India map: average district rainfall by state", "Mean annual rainfall per reporting district by state, with longitude and latitude axes.", "map_average_district_actual.png"),
            ("departure_pct", "India map: average departure from normal by state", "Mean annual percentage departure from normal by state, with longitude and latitude axes.", "map_average_departure_pct.png"),
            ("heavy_days", "India map: heavy rainfall days by state", "Mean annual number of district-level heavy-rainfall days by state.", "map_heavy_days.png"),
            ("extreme_days", "India map: extreme rainfall days by state", "Mean annual number of district-level extreme-rainfall days by state.", "map_extreme_days.png"),
        ]
        for value_col, title, subtitle, filename in map_specs:
            path = create_state_map(state_summary, value_col, title, subtitle, CHART_DIR / filename)
            if path:
                charts.append(ChartSpec(path, title, subtitle, "The India choropleth shows state-level rainfall intensity with longitude and latitude reference axes.", "India maps"))
    if DISTRICT_GEOJSON.exists():
        with DISTRICT_GEOJSON.open("r", encoding="utf-8") as f:
            district_geo = json.load(f)
        district_summary = tables["district_summary"]
        for state_name in sorted(district_summary["state_name"].dropna().unique()):
            path = create_state_district_map(
                district_geo,
                state_name,
                district_summary,
                "actual_mm",
                f"{state_name}: district rainfall map",
                "District boundaries colored by average annual rainfall; grey districts did not match the rainfall table name exactly.",
                CHART_DIR / f"district_map_{slugify(state_name)}.png",
            )
            if path:
                charts.append(ChartSpec(path, f"{state_name}: district rainfall map", "District boundary rainfall map", f"{state_name}'s district map shows within-state rainfall concentration using district boundaries and longitude/latitude axes.", "State district maps"))


def create_more_micro_charts(tables: dict[str, pd.DataFrame], charts: list[ChartSpec]) -> None:
    annual = tables["annual_district"]
    monthly = tables["monthly_district"]
    district_summary = tables["district_summary"]

    for metric, color in [("wet_days", "olive"), ("heavy_days", "orange"), ("departure_pct", "pink")]:
        top = district_summary.dropna(subset=[metric]).sort_values(metric, ascending=False).head(25).copy()
        top["district_label"] = top["district_name"] + ", " + top["state_name"]
        title = f"Top districts by {metric.replace('_', ' ')}"
        path = chart_bar(top, "district_label", metric, title, "Average annual district-level metric across available years.", CHART_DIR / f"micro_{metric}_top_districts.png", color=color)
        charts.append(ChartSpec(path, title, metric, "The ranked district view surfaces the most exposed local geographies for follow-up planning.", "District"))

    latest_year = int(annual["year"].max())
    latest = annual[annual["year"] == latest_year].copy()
    latest["district_label"] = latest["district_name"] + ", " + latest["state_name"]
    for metric, color in [("actual_mm", "blue"), ("departure_pct", "olive"), ("heavy_days", "orange")]:
        title = f"{latest_year}: highest district {metric.replace('_', ' ')}"
        top = latest.dropna(subset=[metric]).sort_values(metric, ascending=False).head(25)
        path = chart_bar(top, "district_label", metric, title, "Latest available year in the dataset; 2024 may be partial if the CSV ends before December.", CHART_DIR / f"latest_year_{metric}_districts.png", color=color)
        charts.append(ChartSpec(path, title, metric, "Latest-year district rankings show near-term monitoring priorities but should be read with the partial-year caveat if applicable.", "District"))

    season = monthly.groupby(["state_name", "season"]).agg(actual_mm=("actual_mm", "mean")).reset_index()
    for season_name in ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]:
        part = season[season["season"] == season_name].sort_values("actual_mm", ascending=False).head(20)
        title = f"{season_name}: wettest states"
        path = chart_bar(part, "state_name", "actual_mm", title, "Average seasonal actual rainfall by state.", CHART_DIR / f"season_{slugify(season_name)}_wettest_states.png", color="blue")
        charts.append(ChartSpec(path, title, "Seasonal rainfall", f"{season_name} rainfall concentration identifies which states matter most outside the all-year average.", "Seasonality"))


def create_chart_catalog(tables: dict[str, pd.DataFrame], allow_download: bool) -> list[ChartSpec]:
    setup_plot_theme()
    charts: list[ChartSpec] = []
    create_national_charts(tables, charts)
    create_india_scope_state_time_charts(tables, charts)
    create_temporal_variation_charts(tables, charts)
    create_state_charts(tables, charts)
    create_state_scope_district_time_charts(tables, charts)
    create_more_micro_charts(tables, charts)
    create_map_charts(tables, charts, allow_download=allow_download)
    chart_map = pd.DataFrame([c.__dict__ | {"path": str(c.path)} for c in charts])
    chart_map.to_csv(TABLE_DIR / "chart_map.csv", index=False)
    return charts


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCustom", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, textColor=colors.HexColor(TOKENS["ink"]), alignment=TA_CENTER, spaceAfter=16),
        "h1": ParagraphStyle("H1Custom", parent=base["Heading1"], fontSize=15, leading=18, textColor=colors.HexColor(TOKENS["ink"]), spaceBefore=10, spaceAfter=8),
        "h2": ParagraphStyle("H2Custom", parent=base["Heading2"], fontSize=12.5, leading=15, textColor=colors.HexColor(TOKENS["blue"]), spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("BodyCustom", parent=base["BodyText"], fontSize=9.4, leading=13.2, textColor=colors.HexColor(TOKENS["ink"]), alignment=TA_LEFT, spaceAfter=6),
        "small": ParagraphStyle("SmallCustom", parent=base["BodyText"], fontSize=7.5, leading=9.5, textColor=colors.HexColor(TOKENS["muted"])),
        "table_header": ParagraphStyle("TableHeaderCustom", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=8.8, textColor=colors.white, alignment=TA_CENTER),
        "table_cell": ParagraphStyle("TableCellCustom", parent=base["BodyText"], fontSize=7.2, leading=8.8, textColor=colors.HexColor(TOKENS["ink"]), alignment=TA_CENTER),
        "code": ParagraphStyle("CodeCustom", parent=base["Code"], fontName="Courier", fontSize=6.6, leading=8.2, textColor=colors.HexColor(TOKENS["ink"])),
    }


def image_flowable(path: Path, max_width: float = 24.5 * cm, max_height: float = 14.8 * cm) -> Image:
    img = Image(str(path))
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


def image_block(path: Path, max_width: float = 24.5 * cm, max_height: float = 14.8 * cm, bottom_space: float = 0.55 * cm) -> KeepTogether:
    return KeepTogether([image_flowable(path, max_width=max_width, max_height=max_height), Spacer(1, bottom_space)])


def summary_table(data: list[list[str]], widths: list[float]) -> Table:
    styles = build_styles()
    wrapped = []
    for row_index, row in enumerate(data):
        style = styles["table_header"] if row_index == 0 else styles["table_cell"]
        wrapped.append([KeepInFrame(widths[col_index] - 8, 2.1 * cm, [Paragraph(str(cell), style)], mode="shrink") for col_index, cell in enumerate(row)])
    table = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TOKENS["blue"])),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(TOKENS["grid"])),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFE")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_analysis_pdf(tables: dict[str, pd.DataFrame], charts: list[ChartSpec]) -> Path:
    path = writable_output_path(REPORT_DIR / "rainfall_analysis_report.pdf")
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), rightMargin=1.1 * cm, leftMargin=1.1 * cm, topMargin=0.9 * cm, bottomMargin=0.9 * cm)
    styles = build_styles()
    story = [
        Paragraph("India District Rainfall Micro-Analysis", styles["title"]),
        Paragraph("Executive Summary", styles["h1"]),
    ]
    qa = tables["quality_summary"].iloc[0]
    national = tables["national_annual"]
    forecast = tables["national_forecast_next_six_months"]
    state_summary = tables["state_summary"].sort_values("avg_district_actual_mm", ascending=False)
    district_summary = tables["district_summary"].sort_values("actual_mm", ascending=False)
    date_range = f"{pd.to_datetime(qa['start_date']).date()} to {pd.to_datetime(qa['end_date']).date()}"
    latest_year = int(national["year"].max())
    wettest_state = state_summary.iloc[0]["state_name"]
    wettest_district = district_summary.iloc[0]["district_name"] + ", " + district_summary.iloc[0]["state_name"]
    corr = weighted_corr(national, "actual_mm", "normal_mm")
    story.extend(
        [
            Paragraph(f"<b>Coverage:</b> {int(qa['rows']):,} district-day records from {date_range}. The source has state and district grain; no block column is present, so block-wise analysis is not statistically supportable from this file alone.", styles["body"]),
            Paragraph(f"<b>National signal:</b> annual actual-vs-normal correlation is {corr:.2f}, which means normal rainfall explains broad seasonal structure but district and state departures still matter.", styles["body"]),
            Paragraph(f"<b>Concentration:</b> {wettest_state} has the highest average district rainfall depth, while {wettest_district} is the wettest district in the district summary.", styles["body"]),
            Paragraph("<b>Time variation:</b> the report includes month-wise, weekday-wise, day-of-month, and requested season-wise cuts. Hourly 24-hour variation is not computed because the CSV has no hour or timestamp field.", styles["body"]),
            Paragraph("<b>Forecast:</b> the next six months use a safe seasonal calendar-median model with 10th-90th percentile historical bands, so it is a planning forecast rather than a precise weather prediction.", styles["body"]),
            Paragraph(f"<b>Latest-year caution:</b> the latest year is {latest_year}; if the data ends before December, use latest-year charts as current-year-to-date views rather than full-year verdicts.", styles["body"]),
            Paragraph("Key Findings With Visual Evidence", styles["h1"]),
        ]
    )

    top_states = state_summary.head(12)
    table_data = [["State", "Avg district actual mm", "Avg district normal mm", "Departure", "Wet days", "Heavy days", "Corr."]] + [
        [
            r["state_name"],
            num(r["avg_district_actual_mm"]),
            num(r["avg_district_normal_mm"]),
            pct(r["departure_pct"]),
            num(r["wet_days"], 0),
            num(r["heavy_days"], 0),
            f"{r['rainfall_normal_corr']:.2f}" if pd.notna(r["rainfall_normal_corr"]) else "n/a",
        ]
        for _, r in top_states.iterrows()
    ]
    story.append(summary_table(table_data, [5.5 * cm, 3 * cm, 3 * cm, 2.4 * cm, 2.5 * cm, 2.5 * cm, 2 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    forecast_data = [["Month", "Season", "Forecast actual", "Lower band", "Upper band", "Departure", "Risk"]] + [
        [
            pd.to_datetime(r["date"]).strftime("%b %Y"),
            str(r["season"]),
            num(r["forecast_actual_mm"]),
            num(r["lower_actual_mm"]),
            num(r["upper_actual_mm"]),
            pct(r["forecast_departure_pct"]),
            str(r["risk_band"]),
        ]
        for _, r in forecast.iterrows()
    ]
    story.append(Paragraph("Six-Month Forecast Summary", styles["h2"]))
    story.append(Paragraph("The model uses the historical median for the same calendar month, plus 10th and 90th percentile bands. It is intentionally conservative because rainfall is seasonal, volatile, and not fully predictable from this single table.", styles["body"]))
    story.append(summary_table(forecast_data, [2.8 * cm, 3.0 * cm, 3.0 * cm, 2.7 * cm, 2.7 * cm, 2.4 * cm, 2.8 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    for chart in charts[:10]:
        story.append(Paragraph(chart.title, styles["h2"]))
        story.append(Paragraph(chart.insight, styles["body"]))
        story.append(image_block(chart.path))

    story.append(PageBreak())
    story.append(Paragraph("India-Level State Analysis", styles["h1"]))
    story.append(Paragraph("India-level analysis compares states using fair per-reporting-district rainfall depth, departure from normal, heavy-rainfall day counts, and actual-normal correlation. The table below is the state-level view; individual state pages that follow drill into district-level patterns.", styles["body"]))
    all_state_rows = [["State", "Avg district actual", "Avg district normal", "Departure", "Wet days", "Heavy days", "Districts", "Corr."]] + [
        [
            r["state_name"],
            num(r["avg_district_actual_mm"]),
            num(r["avg_district_normal_mm"]),
            pct(r["departure_pct"]),
            num(r["wet_days"], 0),
            num(r["heavy_days"], 0),
            num(r["districts"], 0),
            f"{r['rainfall_normal_corr']:.2f}" if pd.notna(r["rainfall_normal_corr"]) else "n/a",
        ]
        for _, r in state_summary.iterrows()
    ]
    story.append(summary_table(all_state_rows, [4.3 * cm, 2.5 * cm, 2.5 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm]))
    india_time_titles = [
        "India daily rainfall time series",
        "India weekly rainfall time series",
        "Monthly rainfall time series",
        "India annual rainfall against normal",
        "India seasonal rainfall time series",
        "India scope: state month-wise rainfall",
        "India scope: state season-wise rainfall",
        "India scope: state week-wise rainfall",
    ]
    chart_by_title = {chart.title: chart for chart in charts}
    for title in india_time_titles:
        chart = chart_by_title.get(title)
        if chart:
            story.append(Paragraph(chart.title, styles["h2"]))
            story.append(Paragraph(chart.insight, styles["body"]))
            story.append(image_block(chart.path, max_height=11.5 * cm))
    story.append(PageBreak())

    district_map_by_state = {
        chart.title.replace(": district rainfall map", ""): chart
        for chart in charts
        if chart.title.endswith(": district rainfall map")
    }
    story.append(Paragraph("State-Wise District Analysis", styles["h1"]))
    story.append(Paragraph("Each state page below uses the state as the container and districts as the analytical grain. Tables list the wettest districts by average annual rainfall and show departure, wet days, heavy days, and extreme-day exposure where available.", styles["body"]))
    for _, state_row in state_summary.iterrows():
        state_name = state_row["state_name"]
        state_districts = district_summary[district_summary["state_name"] == state_name].sort_values("actual_mm", ascending=False)
        if state_districts.empty:
            continue
        story.append(PageBreak())
        story.append(Paragraph(f"{state_name}: District-Level Analysis", styles["h1"]))
        story.append(Paragraph(f"{state_name} averages {num(state_row['avg_district_actual_mm'])} mm per reporting district annually, with {pct(state_row['departure_pct'])} average departure from normal and {num(state_row['heavy_days'], 0)} heavy-rainfall district-days per year at the state aggregate.", styles["body"]))
        map_chart = district_map_by_state.get(state_name)
        if map_chart:
            story.append(Paragraph("District Boundary Map", styles["h2"]))
            story.append(Paragraph(map_chart.insight, styles["body"]))
            story.append(image_block(map_chart.path, max_height=11.0 * cm))
        state_chart_titles = [
            f"{state_name}: daily rainfall time series",
            f"{state_name}: weekly rainfall time series",
            f"{state_name}: monthly rainfall time series",
            f"{state_name}: annual rainfall against normal",
            f"{state_name}: seasonal rainfall time series",
            f"{state_name}: average monthly rainfall profile",
            f"{state_name}: wettest districts",
            f"{state_name}: top district daily rainfall time series",
            f"{state_name}: top district weekly rainfall time series",
            f"{state_name}: top district monthly rainfall time series",
            f"{state_name}: top district yearly rainfall time series",
            f"{state_name}: top district seasonal-year rainfall time series",
            f"{state_name}: district month-wise rainfall",
            f"{state_name}: district season-wise rainfall",
            f"{state_name}: district week-wise rainfall",
        ]
        for title in state_chart_titles:
            chart = chart_by_title.get(title)
            if chart:
                story.append(Paragraph(chart.title, styles["h2"]))
                story.append(Paragraph(chart.insight, styles["body"]))
                story.append(image_block(chart.path, max_height=9.3 * cm, bottom_space=0.65 * cm))
        district_rows = [["District", "Avg actual", "Avg normal", "Departure", "Wet days", "Heavy days", "Extreme days"]] + [
            [
                r["district_name"],
                num(r["actual_mm"]),
                num(r["normal_mm"]),
                pct(r["departure_pct"]),
                num(r["wet_days"], 0),
                num(r["heavy_days"], 0),
                num(r["extreme_days"], 0),
            ]
            for _, r in state_districts.head(15).iterrows()
        ]
        story.append(Paragraph("District Ranking Table", styles["h2"]))
        story.append(summary_table(district_rows, [4.8 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm]))

    story.extend(
        [
            PageBreak(),
            Paragraph("Recommended Next Steps", styles["h1"]),
            Paragraph("1. Use the state and district rankings to prioritize local validation, especially where high rainfall combines with high departure from normal.", styles["body"]),
            Paragraph("2. Treat 2024 as year-to-date if the source file ends before December; refresh the CSV before operational decisions about full-year rainfall.", styles["body"]),
            Paragraph("3. Add block-level or sub-district boundary data if block-wise planning is required; this source cannot infer block rainfall without risking false precision.", styles["body"]),
            Paragraph("4. Add hourly timestamped rainfall data if true 24-hour variation is required; daily records cannot support intra-day patterns.", styles["body"]),
            Paragraph("Further Questions", styles["h1"]),
            Paragraph("Which districts need flood-risk thresholds, crop-calendar overlays, reservoir inflow comparisons, or disaster-response triggers? Those require additional source layers.", styles["body"]),
            Paragraph("Caveats And Assumptions", styles["h1"]),
            Paragraph("Missing actual and normal values are treated as zero for aggregation but tracked in quality tables. Deviation percentages are recomputed from actual and normal totals when possible.", styles["body"]),
            Paragraph("Maps use a public India state GeoJSON boundary file when downloaded successfully; district-level maps require district boundary data and robust district-name matching.", styles["body"]),
            Paragraph("Season definitions follow the requested calendar: January-February winter, March-June pre-monsoon, July-October monsoon, and November-December post-monsoon.", styles["body"]),
            PageBreak(),
            Paragraph("Chart Atlas", styles["h1"]),
            Paragraph(f"The generated catalog contains {len(charts)} charts. Each page below carries one chart with its interpretive note.", styles["body"]),
        ]
    )

    for i, chart in enumerate(charts, 1):
        story.append(Paragraph(f"{i}. {chart.title}", styles["h2"]))
        story.append(Paragraph(f"<b>Section:</b> {chart.section}. <b>Insight:</b> {chart.insight}", styles["body"]))
        story.append(image_block(chart.path, max_height=13.2 * cm))
        if i % 2 == 0:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 0.15 * cm))

    doc.build(story)
    return path


def explain_code_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "Blank line used to separate logical blocks and keep the program readable."
    if stripped.startswith("import ") or stripped.startswith("from "):
        return "Import statement: brings a library, function, or type into the script namespace."
    if stripped.startswith("@dataclass"):
        return "Dataclass decorator: asks Python to create a compact data container with generated methods."
    if stripped.startswith("class "):
        return "Class definition: declares a structured object used by the analysis pipeline."
    if stripped.startswith("def "):
        return "Function definition: names a reusable algorithmic step and its inputs."
    if stripped.startswith("for "):
        return "Loop: repeats the following block for every item in a sequence."
    if stripped.startswith("if "):
        return "Conditional branch: runs the following block only when the stated condition is true."
    if stripped.startswith("elif ") or stripped.startswith("else"):
        return "Conditional continuation: handles an alternative path from the previous condition."
    if stripped.startswith("return "):
        return "Return statement: sends the computed result back to the caller."
    if "=" in stripped and not stripped.startswith(("==", ">=", "<=")):
        return "Assignment or configuration: stores a value so later analysis steps can reuse it."
    if stripped.startswith("with "):
        return "Context manager: opens a resource and guarantees clean-up after the block finishes."
    if stripped.startswith("try"):
        return "Error-handling block: attempts a risky operation while preparing for failure."
    if stripped.startswith("except"):
        return "Error handler: catches a failure and keeps the pipeline from crashing abruptly."
    return "Executable statement: performs one concrete step in the analysis workflow."


def build_code_explanation_pdf(source_path: Path) -> Path:
    path = writable_output_path(REPORT_DIR / "code_explanation.pdf")
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.0 * cm, leftMargin=1.0 * cm, topMargin=1.0 * cm, bottomMargin=1.0 * cm)
    styles = build_styles()
    story = [
        Paragraph("Code Explanation Companion", styles["title"]),
        Paragraph("Algorithm Map", styles["h1"]),
        Paragraph("The script follows a production analysis flow: create folders, read the large CSV in chunks, normalize dates and numeric rainfall fields, aggregate district/state/national tables, build daily, weekly, monthly, yearly, and seasonal time-series layers, render a chart catalog, create map visuals when GeoJSON is available, then compile two PDF deliverables.", styles["body"]),
        Paragraph("Key words: pandas handles tabular aggregation; matplotlib renders charts; reportlab builds PDFs; chunking keeps memory stable for the 266 MB input file; correlation measures how actual rainfall co-moves with normal rainfall across years; time-series logic preserves ordered daily, weekly, monthly, yearly, and seasonal movement.", styles["body"]),
        PageBreak(),
        Paragraph("Line-by-Line Explanation", styles["h1"]),
    ]
    lines = source_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        code = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(f"<b>Line {i}</b>", styles["h2"]))
        story.append(Paragraph(f"<font face='Courier'>{code or ' '}</font>", styles["code"]))
        story.append(Paragraph(explain_code_line(line), styles["body"]))
    doc.build(story)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create rainfall analysis charts and PDFs.")
    parser.add_argument("--csv", type=Path, default=DATA_FILE, help="Path to district-level daily rainfall CSV.")
    parser.add_argument("--chunksize", type=int, default=350_000, help="CSV rows per processing chunk.")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading the optional India state GeoJSON map file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    print("Aggregating rainfall source data...")
    tables = load_and_aggregate(args.csv, chunksize=args.chunksize)
    print("Rendering chart catalog...")
    charts = create_chart_catalog(tables, allow_download=not args.no_download)
    print(f"Rendered {len(charts)} charts.")
    print("Building analysis PDF...")
    report_pdf = build_analysis_pdf(tables, charts)
    print("Building code explanation PDF...")
    code_pdf = build_code_explanation_pdf(Path(__file__))
    print(f"Analysis PDF: {report_pdf}")
    print(f"Code explanation PDF: {code_pdf}")


if __name__ == "__main__":
    main()
