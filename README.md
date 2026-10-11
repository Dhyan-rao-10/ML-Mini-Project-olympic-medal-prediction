# Olympic Medal Prediction

A solo machine-learning mini-project to estimate Summer Olympic medal counts by country using historical Olympic results and country-level development indicators. The repository contains a reproducible data-preparation pipeline, time-aware model evaluation, a benchmark of the assigned paper's model families, and a retrospective 2020 forecast.

## Project question

Can past Olympic performance and socioeconomic indicators help estimate how many medals each country will win? The project compares a last-observed-Games baseline, a two-stage medal-winner classifier plus Poisson count model, and a momentum-adjusted Ridge model. The main test is chronological: 2012 is used for model selection and 2016 is held out as the final historical test.

## Folder guide

```text
data/
  raw/          # Downloaded source files (kept out of Git)
  processed/    # Cleaned/merged datasets (kept out of Git)
notebooks/
  01_data_preparation.ipynb
  02_modeling.ipynb
src/            # Reusable Python code as the project grows
tests/          # Offline regression tests for data preparation and model runs
outputs/figures/ # Charts for the report and presentation
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Install the project packages in the environment:

```bash
# Windows PowerShell
.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS/Linux
python -m pip install -r requirements.txt
```

Open `notebooks/01_data_preparation.ipynb` in VS Code and select the `.venv` Python kernel. It prepares the country-Games targets and merges pre-Games World Bank features. Open `notebooks/02_modeling.ipynb` to run the time-based backtest. From the project root, you can also run either stage in the terminal:

```bash
# Windows PowerShell
.venv\Scripts\python.exe -m src.data_preparation
.venv\Scripts\python.exe -m src.modeling
.venv\Scripts\python.exe -m src.visualization

# macOS/Linux
python -m src.data_preparation
python -m src.modeling
python -m src.visualization
```

This writes the prepared country-Games table, reshaped World Bank indicators, and NOC crosswalk under `data/processed/`. The modeling command also writes validation/holdout predictions and metrics, the retrospective 2020 forecast, and a forecast-versus-actual comparison. To benchmark the model families from the assigned paper, run:

```bash
# Windows PowerShell
.venv\Scripts\python.exe -m src.paper_benchmark

# macOS/Linux
python -m src.paper_benchmark
```

The benchmark compares Logistic Regression, SVM, Gaussian Naive Bayes, MLP, and Random Forest classifiers, plus Linear, Ridge, Lasso, Poisson, SVR, and Random Forest regressors. It tunes on Games through 2008, selects using 2012, refits the selected pair through 2012, and evaluates once on the 2016 holdout. It saves `paper_model_benchmarks.csv` and `paper_model_2016_predictions.csv` under `data/processed/`. The visualization command saves two charts in `outputs/figures/`. Generated data and charts remain excluded from Git.

Run the offline tests from the project root with:

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

On macOS/Linux, use `python -m unittest discover -s tests -v` from the activated environment.

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

Under **Time**, select 1960-2019. Download as CSV from **Download options** and extract the downloaded ZIP. Keep the source CSV and metadata files in `data/raw/`; preserve the downloaded filename ending in `_Data.csv` or rename it to `world_bank_data.csv`. The preparation code reshapes the three selected indicators and matches country codes; `..` values stay missing and are not treated as zero. For the retrospective 2020 forecast, use 2019 or earlier indicator values, not values recorded after the Games.

### 3. Official medal totals for the 2020 comparison

The Kaggle athlete dataset stops at 2016, so get the 2020 actuals separately. The IOC-sourced country/year medal table on [Dataful](https://dataful.in/datasets/19674/) is one option, but if its download is paid, use the [Tokyo 2020 Olympic Summer Games Kaggle dataset](https://www.kaggle.com/datasets/piterfm/tokyo-2020-olympics) and its `medals_total.csv`. That file has one row per medal-winning NOC; countries absent from it should not automatically be treated as nonparticipants. Cross-check the final totals against the [Tokyo 2020 medal standings](https://www.nbcolympics.com/tokyo-2020-medal-standings). Keep the downloaded file in `data/raw/` and cite the original Olympic results source and the dataset used for the download.

For this project, the downloaded medal table is saved as `data/raw/tokyo_2020_medals_total.csv`. The comparison maps Tokyo's `ROC` code to historical `RUS`. It compares the model's 2016-based candidate list with actual medal totals, adds actual medal winners that the forecast did not include, and treats missing medal-table rows as zero recorded medals. Since the source contains medal winners only, a zero may also mean that an NOC did not participate; discuss this limitation when interpreting the scores.

Put the downloaded files under `data/raw/`. Do not commit raw datasets to Git; this repository's `.gitignore` already excludes them. Record the download date and any cleaning or country-code decisions in your write-up.

The preparation notebook checks the source files, builds country-Games medal counts, applies the NOC-to-World Bank crosswalk, and joins indicators from the year before each Games. The crosswalk and final modeling table are saved under `data/processed/` so the cleaning can be reviewed and repeated.

## Suggested modelling workflow

1. Aggregate Olympic outcomes to one row per country per Summer Games.
2. Match country codes consistently before merging World Bank indicators, using the year before each Games for features.
3. Add the previous observed Games medal count and a host-country feature if reliable host data is available.
4. Explore zero-medal records and missing values; document cleaning choices.
5. Compare the last-observed-Games baseline with the two-stage Logistic/Poisson model and momentum-adjusted Ridge model; also run the assigned-paper benchmark across its classifier and regressor families.
6. Use 2012 as the validation/model-selection Games and 2016 as the final historical holdout. Features for every Games use only earlier observed Olympic results and the prior year's World Bank indicators; do not randomly split country-Games rows.
7. Evaluate with MAE/RMSE and a separately reported classification score for medal-winning NOCs. Inspect leading-country predictions and discuss the small historical sample, the limits of the Tokyo medal-only file, and other assumptions.

Because the 2020 Games have already taken place, treat the project as a retrospective forecast: build the 2020 prediction using only information that would have been available before those Games, then compare with actual 2020 outcomes at the end.

## Current evaluation (reproduce before quoting)

The latest local run of `python -m src.data_preparation` and `python -m src.modeling` produced these results:

| Evaluation | Model | MAE | RMSE | Notes |
| --- | --- | ---: | ---: | --- |
| 2012 validation | Momentum-adjusted Ridge | 1.129 | 2.380 | Ridge/blend settings selected on this Games; score is optimistic |
| 2012 validation | Previous-Games baseline | 1.305 | 2.855 | Same evaluated NOCs |
| 2016 final holdout | Momentum-adjusted Ridge | 1.303 | 3.417 | Settings fixed from 2012; trained with data available before 2016 |
| 2016 final holdout | Previous-Games baseline | 1.265 | 3.230 | Same evaluated NOCs; baseline is better on this holdout |
| 2016 final holdout | Paper-family benchmark (SVM + Random Forest) | 1.418 | 3.563 | Selected using 2012 only; baseline is better on this holdout |
| Tokyo 2020 retrospective | Momentum-adjusted Ridge | 1.556 | 3.558 | Uses 2016 medal totals and delegation size |
| Tokyo 2020 retrospective | Previous-Games baseline | 1.683 | 3.673 | Same comparison table |

The assigned-paper benchmark records every candidate's 2012 validation scores and selected parameters. On the current data, SVM had the highest 2012 validation accuracy (0.892), and Random Forest had the lowest validation residual standard deviation among medal winners (3.815); their selected two-stage model's 2016 medal-count MAE/RMSE were 1.418/3.563. This is a leakage-aware comparison of the paper's model families, not an exact reproduction: the available World Bank export contains GDP, population, and land area, and this benchmark uses previous-Games delegation size rather than same-Games athlete counts, which would not be available for a pre-Games forecast. Thus its GaussianNB + tuned Ridge selection and residual statistic are not directly comparable. On the 2016 holdout, both our Ridge model and the paper-family benchmark lose to the previous-Games baseline (MAE 1.265; RMSE 3.230). Ridge improves count errors in the retrospective Tokyo comparison, but that comparison has the participation-data limitation below; its Logistic medal-winner classifier (threshold 0.56) has recall 0.753 and precision 0.933. Do not claim that this project beats every prior model. Full metrics and country-level predictions are saved under `data/processed/` after running the pipeline.

The Tokyo file contains medal-winning NOCs only. The evaluation treats forecast candidates with no row in that file as zero recorded medals, but cannot distinguish a non-participant from a participant that won no medal. This limits interpretation of all Tokyo-wide error and classification scores. The file also does not provide an official non-medalist participation table. Chinese Taipei's NOC is retained through a manual `TPE` → World Bank `TWN` code mapping, but this particular World Bank export has no Taiwan indicator rows, so those values are imputed by the models.

## Deliverables checklist

- Private GitHub repository with this README, reproducible code, and offline tests
- Two-page PDF write-up (the assignment heading says “One-Page,” while the detailed requirement says two pages; follow the detailed two-page instruction unless the faculty clarifies)
- Presentation slides and a live demonstration

## Team note

This repository is being completed by one student. Keep the first version small and make sure every result can be rerun from the README instructions.
