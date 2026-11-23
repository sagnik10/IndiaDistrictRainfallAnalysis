<!-- SAGNIK10-STYLE-HEADER -->
<div align="center">
  <a href="https://github.com/sagnik10"><img src="https://img.shields.io/badge/Sagnik_Sen-Atmospheric%20%26%20Science-275cce?style=flat-square" alt="Category" /></a>
  <img src="https://img.shields.io/badge/Stack-Python-192a43?style=flat-square" alt="Stack" />
  <img src="https://img.shields.io/badge/License-MIT-54647a?style=flat-square" alt="License" />
</div>

<p align="center">
  <a href="https://sagnik10.github.io/SagnikSenPortfolio/">Portfolio</a> ·
  <a href="https://github.com/sagnik10">GitHub Profile</a> ·
  <a href="https://github.com/sagnik10/sagnik-sen-antigravity-plugin">Antigravity Plugin</a>
</p>

---
<!-- /SAGNIK10-STYLE-HEADER -->
# Rainfall Data Analysis

Production-grade rainfall analysis for the local file:

```text
daily-rainfall-data-district-level.csv
```

The pipeline reads the large CSV in chunks, builds national, state, district, daily, weekly, monthly, yearly, weekday, day-of-month, and seasonal aggregates, renders an atlas of 100+ charts, creates India-level state maps, creates state-level district-boundary maps, adds time-series variation, creates a conservative six-month forecast, and exports two PDFs:

- `outputs/reports/rainfall_analysis_report.pdf`
- `outputs/reports/code_explanation.pdf`

## Quick Start

```powershell
py Rainfall_Data_Anaqalysis.py
```

Use a smaller chunk size if RAM is tight:

```powershell
py Rainfall_Data_Anaqalysis.py --chunksize 150000
```

Skip map boundary download:

```powershell
py Rainfall_Data_Anaqalysis.py --no-download
```

## Outputs

```text
outputs/
  charts/       PNG chart atlas
  tables/       Aggregated CSV tables and chart map
  reports/      Final PDFs
assets/
  india_state.geojson and india_district_modern.geojson, downloaded only when map generation is enabled
```

## Data Grain

The source file contains:

- `date`
- `state_code`, `state_name`
- `district_code`, `district_name`
- `actual`
- `rfs`
- `normal`
- `deviation`

There is no block-level field in the CSV, so block-wise analysis is not inferred. To add block-level analysis, provide a block-grain rainfall file or a reliable district-to-block allocation method.

There is also no hour or timestamp field, so true 24-hour variation is not inferred. The pipeline saves `outputs/tables/hourly_variation_status.csv` with the exact reason and needed data.

## Season Definitions

The script follows the requested calendar:

- Winter: January and February
- Pre-monsoon: March, April, May, and June
- Monsoon: July, August, September, and October
- Post-monsoon: November and December

## Forecasting

The six-month forecast is intentionally safe and conservative. It uses the historical median for the same calendar month and adds 10th-90th percentile historical bands. This is useful for planning and monitoring, not for precise weather prediction.

Forecast outputs:

- `outputs/tables/national_forecast_next_six_months.csv`
- `outputs/tables/state_forecast_next_six_months.csv`
- `outputs/tables/state_weekly.csv`
- `outputs/tables/weekly_district.csv`
- `outputs/tables/national_daily.csv`
- `outputs/tables/national_weekly.csv`
- `outputs/tables/national_monthly.csv`
- `outputs/tables/national_annual.csv`
- `outputs/tables/national_seasonal_year.csv`
- `outputs/tables/state_seasonal_year.csv`
- `outputs/tables/district_seasonal_year.csv`
- `outputs/charts/national_six_month_forecast.png`
- `outputs/charts/state_six_month_forecast_totals.png`

## Scope Rules

The report is organized by analytical scope:

- India scope contains state-level analysis, including state month-wise, week-wise, and season-wise comparisons.
- India scope also contains national daily, weekly, monthly, yearly, and seasonal time-series graphs.
- State scope contains state daily, weekly, monthly, yearly, and seasonal time-series graphs.
- State scope contains district-level analysis, including district daily, weekly, monthly, yearly, season-wise, and seasonal-year time-series comparisons for each state or union territory.
- State maps use district boundaries and centered chart placement in the PDF.
- PDF tables use wrapped centered cells and white table-header text.

## Map Source

India-level state maps use the public India state GeoJSON file from `geohacker/india`:

```text
https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson
```

State-level district maps use the public district GeoJSON file from `datta07/INDIAN-SHAPEFILES`:

```text
https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_DISTRICTS.geojson
```

All states and union territories in the rainfall source receive an individual state-wise district analysis section. District map polygons are colored when the district name can be reconciled to the rainfall table; unmatched polygons are kept neutral rather than forcing false matches.

## Dependencies

The script uses packages available in the current environment:

- pandas
- numpy
- matplotlib
- reportlab

Optional map download uses Python standard-library networking.

## Reproducibility

Every generated table is saved under `outputs/tables/`. The `chart_map.csv` file records each chart path, title, section, and insight used in the PDF atlas.


<!-- SAGNIK10-STYLE-FOOTER -->
---
<p align="center">
  Maintained by <a href="https://github.com/sagnik10"><strong>Sagnik Sen</strong></a> ·
  <a href="https://sagnik10.github.io/SagnikSenPortfolio/">Portfolio &amp; 1-on-1 Teaching</a> ·
  <a href="https://www.linkedin.com/in/sagnik-sen-0a70431a8/">LinkedIn</a>
</p>
<!-- /SAGNIK10-STYLE-FOOTER -->