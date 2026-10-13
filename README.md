# Olympic Medal Prediction

This machine-learning project estimates Summer Olympic medal counts by country using historical Olympic results and country-level development indicators. The repository contains a reproducible data-preparation pipeline, time-aware model evaluation, a benchmark of the assigned paper's model families, and a retrospective 2020 forecast.

**Team:** Dhyan Rao (PES1UG24AM090) and J Karthik (PES1UG25AM804)

## Project question

Can past Olympic performance and socioeconomic indicators help estimate how many medals each country will win? The project compares a last-observed-Games baseline, a two-stage medal-winner classifier plus Poisson count model, and a momentum-adjusted Ridge model. The main test is chronological: 2012 is used for model selection and 2016 is held out as the final historical test.

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

From the project root, prepare the country-Games data and run the main evaluation:

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

The benchmark compares Logistic Regression, SVM, Gaussian Naive Bayes, MLP, and Random Forest classifiers, plus Linear, Ridge, Lasso, Poisson, SVR, and Random Forest regressors. It tunes on Games through 2008, selects using 2012, refits the selected pair through 2012, and evaluates once on the 2016 holdout. Results are saved in `data/processed/paper_model_benchmarks.csv` and `data/processed/paper_model_2016_predictions.csv`. The visualization command saves charts in `outputs/figures/`. Generated data and charts are excluded from Git.

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

## Results

The evaluation is chronological: 2012 is used for validation/model selection and 2016 is held out for final historical testing. The 2020 exercise is retrospective and uses only information available before those Games.

| Evaluation | Model | MAE | RMSE | Notes |
| --- | --- | ---: | ---: | --- |
| 2012 validation | Momentum-adjusted Ridge | 1.129 | 2.380 | Ridge/blend settings selected on this Games; score is optimistic |
| 2012 validation | Previous-Games baseline | 1.305 | 2.855 | Same evaluated NOCs |
| 2016 final holdout | Momentum-adjusted Ridge | 1.303 | 3.417 | Settings fixed from 2012; trained with data available before 2016 |
| 2016 final holdout | Previous-Games baseline | 1.265 | 3.230 | Same evaluated NOCs; baseline is better on this holdout |
| 2016 final holdout | Paper-family benchmark (SVM + Random Forest) | 1.418 | 3.563 | Selected using 2012 only; baseline is better on this holdout |
| Tokyo 2020 retrospective | Momentum-adjusted Ridge | 1.556 | 3.558 | Uses 2016 medal totals and delegation size |
| Tokyo 2020 retrospective | Previous-Games baseline | 1.683 | 3.673 | Same comparison table |

In the paper-family benchmark, SVM had the highest 2012 validation accuracy (0.892), while Random Forest had the lowest validation residual standard deviation among medal winners (3.815). The selected SVM + Random Forest pair scored 1.418 MAE and 3.563 RMSE on the 2016 holdout. This benchmark is not an exact reproduction: the available World Bank export provides GDP, population, and land area, and this project uses previous-Games delegation size rather than same-Games athlete counts, which would not be available before a Games. The paper's model selection and error statistics are therefore not directly comparable. On the 2016 holdout, both Ridge and the paper-family benchmark score worse than the previous-Games baseline. Ridge improves count errors in the retrospective Tokyo comparison, but that comparison has the participation-data limitation below. Full metrics and country-level predictions are saved under `data/processed/` after running the pipeline.

The Tokyo source lists medal-winning NOCs only. An absent row may indicate non-participation or participation without a medal, so Tokyo-wide error and classification scores should be interpreted cautiously. Chinese Taipei is mapped from NOC `TPE` to World Bank code `TWN`, but this World Bank export has no Taiwan indicator values; those features are imputed during model fitting.
