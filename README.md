# Olympic Medal Prediction

A solo machine-learning mini-project to estimate Summer Olympic medal counts by country using historical Olympic results and country-level development indicators.

## Project question

Can past Olympic performance and socioeconomic indicators help estimate how many medals each country will win? The first version will use country-year records and compare a simple previous-Games baseline with a small set of regression models. If time allows, it will also try the paper's two-stage approach: classify whether a country wins any medal, then estimate the medal count for medal-winning countries.

## Folder guide

```text
data/
  raw/          # Downloaded source files (kept out of Git)
  processed/    # Cleaned/merged datasets (kept out of Git)
notebooks/
  01_data_preparation.ipynb
src/            # Reusable Python code as the project grows
outputs/figures/ # Charts for the report and presentation
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the project packages:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
jupyter lab
```

Open `notebooks/01_data_preparation.ipynb` and run the cells from the top.

## Download the datasets

### 1. Historical Olympic results (same source used by the assigned paper)

Go to [Kaggle: 120 years of Olympic history - athletes and results](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results), sign in, choose **Download**, and extract the archive. Copy these two files into `data/raw/`:

- `athlete_events.csv` - athlete/event results for Summer and Winter Games from 1896 through 2016
- `noc_regions.csv` - mapping from Olympic NOC codes to region/country names

This is the source cited in the assigned report. In `athlete_events.csv`, filter `Season` to `Summer`. Medal rows for team events repeat once per athlete, so do not simply count rows as medals; check your country-year totals against the official medal table described below.

### 2. World Bank development indicators

Open the official [World Development Indicators DataBank](https://databank.worldbank.org/source/world-development-indicators). Under **Variables**, select countries, then search and select these three series by code:

- GDP (current US$) - `NY.GDP.MKTP.CD`
- Population, total - `SP.POP.TOTL`
- Land area (sq. km) - `AG.LND.TOTL.K2`

Under **Time**, select 1960-2019. Download as CSV from **Download options** and extract the downloaded ZIP. Keep the source CSV and metadata files in `data/raw/`; rename the data CSV to `world_bank_data.csv`. Filter out regional/income aggregates when cleaning. The World Bank allows downloads without an account. For the retrospective 2020 forecast, use 2019 or earlier indicator values, not values recorded after the Games.

### 3. Official medal totals for the 2020 comparison

The Kaggle athlete dataset stops at 2016, so get the 2020 actuals separately from the [IOC-sourced country/year medal table on Dataful](https://dataful.in/datasets/19674/). Choose **Download CSV**. This table includes both Summer and Winter Games and separate Gold/Silver/Bronze/Total rows. For the 2020 comparison, retain `year = 2020`, `olympic_type = Summer`, and `medal_type = Total`, excluding the `All Countries` aggregate row. It includes host-country and country-name fields, which also lets you check Japan's host flag and country mapping. Cite the IOC as the original source and Dataful as the download page.

Put the downloaded files under `data/raw/`. Do not commit raw datasets to Git; this repository's `.gitignore` already excludes them. Record the download date and any cleaning or country-code decisions in your write-up.

The notebook checks for the two Kaggle files and reports their columns, size, and missing values. It intentionally starts with inspection; finalize medal aggregation and country matching after looking at the exact source columns.

## Suggested modelling workflow

1. Aggregate Olympic outcomes to one row per country per Summer Games.
2. Match country codes consistently before merging World Bank indicators.
3. Add previous-Games medal count and a host-country feature if reliable host data is available.
4. Explore zero-medal records and missing values; document cleaning choices.
5. Compare a previous-Games baseline, Ridge/Linear Regression, and Random Forest or Gradient Boosting.
6. Split by Olympic year (for example, train through 2012 and evaluate on 2016); do not randomly split country-year rows.
7. Evaluate with MAE/RMSE, inspect leading-country predictions, and discuss the small dataset and other limitations.

Because the 2020 Games have already taken place, treat the project as a retrospective forecast: build the 2020 prediction using only information that would have been available before those Games, then compare with actual 2020 outcomes at the end.

## Deliverables checklist

- Private GitHub repository with this README and reproducible code
- Two-page PDF write-up (the assignment heading says “One-Page,” while the detailed requirement says two pages; follow the detailed two-page instruction unless the faculty clarifies)
- Presentation slides and a live demonstration

## Team note

This repository is being completed by one student. Keep the first version small and make sure every result can be rerun from the README instructions.
