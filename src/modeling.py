"""Prepare, evaluate, and forecast Summer Olympic medal totals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "Log_GDP_USD",
    "Log_Population",
    "Log_Land_Area_Sq_Km",
    "Previous_Observed_Total",
    "Is_Host",
    "Year",
]
REPORT_CLASSIFIER_FEATURES = [
    "Log_GDP_USD",
    "Log_Population",
    "Log_Land_Area_Sq_Km",
    "Log_GDP_Per_Capita",
    "Previous_Observed_Total",
    "Previous_Observed_Total_2",
    "Recent_Total_Trend",
    "Is_Host",
    "Year",
]
COUNT_FEATURES = [
    "Previous_Observed_Total",
    "Previous_Observed_Athlete_Count",
    "Is_Host",
    "Year",
]
RIDGE_ALPHA_CANDIDATES = (0.1, 1.0, 10.0, 100.0, 1000.0)
RIDGE_BLEND_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
CLASSIFIER_THRESHOLD = 0.56
HOST_NOC_BY_YEAR = {
    1964: "JPN", 1968: "MEX", 1976: "CAN", 1980: "URS", 1984: "USA",
    1988: "KOR", 1992: "ESP", 1996: "USA", 2000: "AUS", 2004: "GRE",
    2008: "CHN", 2012: "GBR", 2016: "BRA", 2020: "JPN",
}
TEST_YEARS = [2012, 2016]
TRAIN_THROUGH_YEAR = 2008


def _fit_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute and standardize using training rows only."""
    feature_columns = FEATURES if features is None else features
    train_values = train[feature_columns].to_numpy(dtype=float)
    test_values = test[feature_columns].to_numpy(dtype=float)
    medians = np.nanmedian(train_values, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("At least one training feature is entirely missing.")
    train_values = np.where(np.isnan(train_values), medians, train_values)
    test_values = np.where(np.isnan(test_values), medians, test_values)
    means = train_values.mean(axis=0)
    scales = train_values.std(axis=0)
    scales[scales == 0] = 1.0
    return (train_values - means) / scales, (test_values - means) / scales


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    difference = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(difference**2)))


def _residual_std(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.std(np.asarray(actual) - np.asarray(predicted)))


def _classification_metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    true_positive = int(np.sum(actual & predicted))
    true_negative = int(np.sum(~actual & ~predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    positive_scores = np.asarray(probability)[actual]
    negative_scores = np.asarray(probability)[~actual]
    auc = float(
        ((positive_scores[:, None] > negative_scores).sum()
         + 0.5 * (positive_scores[:, None] == negative_scores).sum())
        / (len(positive_scores) * len(negative_scores))
    ) if len(positive_scores) and len(negative_scores) else float("nan")
    return {
        "Accuracy": float(np.mean(actual == predicted)),
        "Balanced accuracy": (recall + specificity) / 2,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC AUC": auc,
    }


def _select_ridge_settings(data: pd.DataFrame) -> tuple[float, float]:
    """Choose Ridge strength/blend on 2012 validation only, never the 2016 holdout."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    history = data.loc[data["Year"].lt(2012)]
    validation = data.loc[data["Year"].eq(2012)]
    if history.empty or validation.empty:
        raise ValueError("Ridge model selection requires pre-2012 history and 2012 validation rows.")

    baseline = validation["Previous_Observed_Total"].to_numpy(dtype=float)
    actual = validation["Total"].to_numpy(dtype=float)
    candidates = []
    for alpha in RIDGE_ALPHA_CANDIDATES:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=alpha),
        )
        model.fit(history[COUNT_FEATURES], history["Total"].astype(float))
        ridge_prediction = np.maximum(model.predict(validation[COUNT_FEATURES]), 0)
        for blend_weight in RIDGE_BLEND_CANDIDATES:
            prediction = (1 - blend_weight) * baseline + blend_weight * ridge_prediction
            candidates.append(
                (_rmse(actual, prediction), _mae(actual, prediction), alpha, blend_weight)
            )
    _, _, selected_alpha, selected_blend = min(candidates)
    return float(selected_alpha), float(selected_blend)


def run_backtest(processed_dir: Path, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run chronological validation/holdout comparisons without random row splits."""
    from sklearn.linear_model import LogisticRegression, PoissonRegressor

    processed_dir = Path(processed_dir)
    source = processed_dir / "country_year_modeling_table.csv"
    if not source.exists():
        raise FileNotFoundError(f"Prepared data not found: {source}. Run src.data_preparation first.")

    data = pd.read_csv(source)
    required = {
        "Year", "NOC", "GDP_USD", "Population", "Land_Area_Sq_Km", "Athlete_Count", "Total"
    }
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Prepared data is missing required columns: {sorted(missing_columns)}")
    if data.duplicated(["Year", "NOC"]).any():
        raise ValueError("Prepared data must have at most one row per NOC per Games year.")

    data = data.sort_values(["NOC", "Year"]).copy()
    data["Previous_Observed_Total"] = data.groupby("NOC")["Total"].shift().fillna(0)
    data["Previous_Observed_Total_2"] = data.groupby("NOC")["Total"].shift(2).fillna(0)
    data["Recent_Total_Trend"] = data["Previous_Observed_Total"] - data["Previous_Observed_Total_2"]
    data["Previous_Observed_Athlete_Count"] = data.groupby("NOC")["Athlete_Count"].shift().fillna(0)
    data["Is_Host"] = [int(HOST_NOC_BY_YEAR.get(year) == noc) for year, noc in zip(data["Year"], data["NOC"])]
    data["Log_GDP_USD"] = np.log1p(data["GDP_USD"].where(data["GDP_USD"] >= 0))
    data["Log_Population"] = np.log1p(data["Population"].where(data["Population"] >= 0))
    data["Log_Land_Area_Sq_Km"] = np.log1p(
        data["Land_Area_Sq_Km"].where(data["Land_Area_Sq_Km"] >= 0)
    )
    data["Log_GDP_Per_Capita"] = np.log1p(
        (data["GDP_USD"] / data["Population"]).where(data["GDP_USD"].ge(0) & data["Population"].gt(0))
    )
    data["Won_Any_Medal"] = data["Total"].gt(0).astype("int8")
    selected_alpha, selected_blend = _select_ridge_settings(data)

    train = data.loc[data["Year"].le(TRAIN_THROUGH_YEAR)].copy()
    test = data.loc[data["Year"].isin(TEST_YEARS)].copy()
    if train.empty or test.empty:
        raise ValueError("The prepared data does not contain the configured train/test years.")

    x_train, x_test = _fit_transform(train, test)
    y_train_class = train["Won_Any_Medal"]
    y_test_class = test["Won_Any_Medal"]
    y_train_count = train["Total"].astype(float)
    y_test_count = test["Total"].astype(float)

    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=random_state
    )
    classifier.fit(x_train, y_train_class)
    win_probability = classifier.predict_proba(x_test)[:, 1]
    predicted_winner = (win_probability >= 0.5).astype("int8")

    positive_train = train.loc[train["Total"].gt(0)]
    positive_values, x_test_positive = _fit_transform(positive_train, test)
    regressor = PoissonRegressor(alpha=1.0, max_iter=2000)
    regressor.fit(positive_values, positive_train["Total"].astype(float))
    predicted_if_winner = np.maximum(regressor.predict(x_test_positive), 0)
    expected_total = win_probability * predicted_if_winner
    previous_total_baseline = test["Previous_Observed_Total"].to_numpy(dtype=float)

    metrics = [
        {"Model": "Two-stage classifier + Poisson regressor", "Metric": "MAE total medals", "Value": _mae(y_test_count, expected_total)},
        {"Model": "Two-stage classifier + Poisson regressor", "Metric": "RMSE total medals", "Value": _rmse(y_test_count, expected_total)},
        {"Model": "Last-observed-Games baseline", "Metric": "MAE total medals", "Value": _mae(y_test_count, previous_total_baseline)},
        {"Model": "Last-observed-Games baseline", "Metric": "RMSE total medals", "Value": _rmse(y_test_count, previous_total_baseline)},
    ]
    for metric_name, metric_value in _classification_metrics(
        y_test_class, predicted_winner, win_probability
    ).items():
        metrics.append(
            {"Model": "Medal-winner classifier", "Metric": metric_name, "Value": metric_value}
        )
    winner_test = y_test_count.gt(0)
    if winner_test.any():
        metrics.append(
            {
                "Model": "Positive-count Poisson regressor",
                "Metric": "MAE among actual medal winners",
                "Value": _mae(y_test_count[winner_test], predicted_if_winner[winner_test]),
            }
        )

    # Walk-forward benchmark selected on Games before the final evaluation
    # period. Medal momentum is the anchor; Ridge contributes a modest
    # adjustment using prior delegation size and host status.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rolling_rows = []
    for evaluation_year in TEST_YEARS:
        historical = data.loc[data["Year"].lt(evaluation_year)].copy()
        evaluation = data.loc[data["Year"].eq(evaluation_year)].copy()
        if historical.empty or evaluation.empty:
            continue
        count_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=selected_alpha),
        )
        count_model.fit(historical[COUNT_FEATURES], historical["Total"].astype(float))
        ridge_prediction = np.maximum(count_model.predict(evaluation[COUNT_FEATURES]), 0)
        momentum_prediction = (
            (1 - selected_blend) * evaluation["Previous_Observed_Total"].to_numpy(dtype=float)
            + selected_blend * ridge_prediction
        )

        classifier_history = historical.loc[historical["Year"].ge(1988)]
        classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=5.0, max_iter=2000, random_state=42),
        )
        x_train, x_test = _fit_transform(
            classifier_history, evaluation, REPORT_CLASSIFIER_FEATURES
        )
        classifier.fit(x_train, classifier_history["Total"].gt(0).astype("int8"))
        rolling_win_probability = classifier.predict_proba(x_test)[:, 1]
        rolling_predicted_winner = (rolling_win_probability >= CLASSIFIER_THRESHOLD).astype("int8")
        rolling_rows.append(
            pd.DataFrame(
                {
                    "Year": evaluation["Year"].to_numpy(),
                    "Evaluation_Role": (
                        "validation (used for model selection)"
                        if evaluation_year == 2012
                        else "final holdout"
                    ),
                    "NOC": evaluation["NOC"].to_numpy(),
                    "Actual_Total": evaluation["Total"].to_numpy(),
                    "Previous_Observed_Total": evaluation["Previous_Observed_Total"].to_numpy(),
                    "Ridge_Count_Prediction": ridge_prediction,
                    "Momentum_Ridge_Prediction": momentum_prediction,
                    "Predicted_Win_Probability": rolling_win_probability,
                    "Predicted_Winner_At_Tuned_Threshold": rolling_predicted_winner,
                }
            )
        )

    rolling_predictions = pd.concat(rolling_rows, ignore_index=True)
    rolling_predictions["Selected_Ridge_Alpha"] = selected_alpha
    rolling_predictions["Momentum_Blend_Weight"] = selected_blend
    for evaluation_year, games in rolling_predictions.groupby("Year", sort=True):
        role = games["Evaluation_Role"].iloc[0]
        actual = games["Actual_Total"].to_numpy(dtype=float)
        predicted = games["Momentum_Ridge_Prediction"].to_numpy(dtype=float)
        baseline = games["Previous_Observed_Total"].to_numpy(dtype=float)
        top_countries = games.nlargest(10, "Actual_Total")
        metric_rows = [
            ("Momentum-adjusted Ridge", "MAE total medals", _mae(actual, predicted)),
            ("Momentum-adjusted Ridge", "RMSE total medals", _rmse(actual, predicted)),
            ("Last-observed-Games baseline", "MAE total medals", _mae(actual, baseline)),
            ("Last-observed-Games baseline", "RMSE total medals", _rmse(actual, baseline)),
            (
                "Momentum-adjusted Ridge",
                "MSE among 10 highest-medal NOCs",
                _rmse(
                    top_countries["Actual_Total"].to_numpy(),
                    top_countries["Momentum_Ridge_Prediction"].to_numpy(),
                ) ** 2,
            ),
        ]
        for metric_name, metric_value in _classification_metrics(
            games["Actual_Total"].gt(0),
            games["Predicted_Winner_At_Tuned_Threshold"],
            games["Predicted_Win_Probability"],
        ).items():
            metric_rows.append(
            ("Logistic medal-winner classifier (threshold 0.56)", metric_name, metric_value)
            )
        metrics.extend(
            {
                "Evaluation_Year": int(evaluation_year),
                "Evaluation_Role": role,
                "Model": model_name,
                "Metric": metric_name,
                "Value": float(metric_value),
            }
            for model_name, metric_name, metric_value in metric_rows
        )
    rolling_predictions.to_csv(processed_dir / "walk_forward_model_comparison.csv", index=False)

    predictions = test[["Year", "NOC", "Olympic_Region", "WB_Country_Code", "Total"]].copy()
    predictions = predictions.rename(columns={"Total": "Actual_Total"})
    predictions["Actual_Won_Any_Medal"] = y_test_class.to_numpy()
    predictions["Predicted_Win_Probability"] = win_probability
    predictions["Predicted_Winner_At_0_5"] = predicted_winner
    predictions["Predicted_Medals_If_Winner"] = predicted_if_winner
    predictions["Expected_Total_Prediction"] = expected_total
    predictions["Previous_Observed_Total_Baseline"] = previous_total_baseline
    predictions["Rounded_Total_Prediction"] = np.rint(expected_total).astype(int)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame["Value"] = metrics_frame["Value"].astype(float)
    metrics_frame.to_csv(processed_dir / "backtest_metrics.csv", index=False)
    predictions.sort_values(["Year", "Expected_Total_Prediction"], ascending=[True, False]).to_csv(
        processed_dir / "backtest_2012_2016_predictions.csv", index=False
    )
    return metrics_frame, predictions


def run_2020_forecast(processed_dir: Path, random_state: int = 42) -> pd.DataFrame:
    """Fit using available Olympic history and produce a retrospective Tokyo forecast."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    processed_dir = Path(processed_dir)
    history_path = processed_dir / "country_year_modeling_table.csv"
    world_bank_path = processed_dir / "world_bank_country_year_features.csv"
    if not history_path.exists() or not world_bank_path.exists():
        raise FileNotFoundError("Run src.data_preparation first to create the history and feature tables.")

    data = pd.read_csv(history_path).sort_values(["NOC", "Year"]).copy()
    if "Athlete_Count" not in data.columns:
        raise ValueError("Prepared data needs Athlete_Count. Rerun src.data_preparation first.")
    world_bank = pd.read_csv(world_bank_path)
    last_games_year = int(data["Year"].max())
    if last_games_year >= 2020:
        raise ValueError("The forecast routine expects historical Games ending before 2020.")

    data["Previous_Observed_Total"] = data.groupby("NOC")["Total"].shift().fillna(0)
    data["Previous_Observed_Total_2"] = data.groupby("NOC")["Total"].shift(2).fillna(0)
    data["Recent_Total_Trend"] = data["Previous_Observed_Total"] - data["Previous_Observed_Total_2"]
    data["Previous_Observed_Athlete_Count"] = data.groupby("NOC")["Athlete_Count"].shift().fillna(0)
    data["Is_Host"] = [int(HOST_NOC_BY_YEAR.get(year) == noc) for year, noc in zip(data["Year"], data["NOC"])]
    data["Log_GDP_USD"] = np.log1p(data["GDP_USD"].where(data["GDP_USD"] >= 0))
    data["Log_Population"] = np.log1p(data["Population"].where(data["Population"] >= 0))
    data["Log_Land_Area_Sq_Km"] = np.log1p(
        data["Land_Area_Sq_Km"].where(data["Land_Area_Sq_Km"] >= 0)
    )
    data["Log_GDP_Per_Capita"] = np.log1p(
        (data["GDP_USD"] / data["Population"]).where(data["GDP_USD"].ge(0) & data["Population"].gt(0))
    )
    data["Won_Any_Medal"] = data["Total"].gt(0).astype("int8")
    selected_alpha, selected_blend = _select_ridge_settings(data)

    candidates = data.loc[data["Year"].eq(last_games_year)].copy()
    candidates["Year"] = 2020
    # The selected 2016 row is itself the latest known Games. Promote its
    # observed totals and team size to the 2020 lag features (rather than the
    # 2012 lag features attached to the 2016 row during model construction).
    candidates["Previous_Observed_Total_2"] = candidates["Previous_Observed_Total"]
    candidates["Previous_Observed_Total"] = candidates["Total"]
    candidates["Recent_Total_Trend"] = candidates["Previous_Observed_Total"] - candidates["Previous_Observed_Total_2"]
    candidates["Previous_Observed_Athlete_Count"] = candidates["Athlete_Count"]
    candidates["Is_Host"] = [int(HOST_NOC_BY_YEAR.get(2020) == noc) for noc in candidates["NOC"]]
    candidates["Feature_Year"] = 2019
    wb_2019 = world_bank.loc[world_bank["Year"].eq(2019)].rename(
        columns={"Country Code": "WB_Country_Code", "Country Name": "WB_Country_Name"}
    )
    candidates = candidates.drop(
        columns=["GDP_USD", "Population", "Land_Area_Sq_Km", "Feature_Year"]
    ).merge(
        wb_2019.drop(columns="Year"),
        on=["WB_Country_Code", "WB_Country_Name"],
        how="left",
        validate="many_to_one",
    )
    candidates["Feature_Year"] = 2019
    candidates["Log_GDP_USD"] = np.log1p(candidates["GDP_USD"].where(candidates["GDP_USD"] >= 0))
    candidates["Log_Population"] = np.log1p(
        candidates["Population"].where(candidates["Population"] >= 0)
    )
    candidates["Log_Land_Area_Sq_Km"] = np.log1p(
        candidates["Land_Area_Sq_Km"].where(candidates["Land_Area_Sq_Km"] >= 0)
    )
    candidates["Log_GDP_Per_Capita"] = np.log1p(
        (candidates["GDP_USD"] / candidates["Population"]).where(
            candidates["GDP_USD"].ge(0) & candidates["Population"].gt(0)
        )
    )

    x_train, x_future = _fit_transform(data, candidates, REPORT_CLASSIFIER_FEATURES)
    classifier = LogisticRegression(C=5.0, max_iter=2000, random_state=random_state)
    classifier.fit(x_train, data["Won_Any_Medal"])
    win_probability = classifier.predict_proba(x_future)[:, 1]

    positive_train = data.loc[data["Total"].gt(0)]
    positive_values, future_positive_values = _fit_transform(
        positive_train, candidates, REPORT_CLASSIFIER_FEATURES
    )
    regressor = PoissonRegressor(alpha=1.0, max_iter=2000)
    regressor.fit(positive_values, positive_train["Total"].astype(float))
    predicted_if_winner = np.maximum(regressor.predict(future_positive_values), 0)
    ridge_count_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=selected_alpha),
    )
    ridge_count_model.fit(data[COUNT_FEATURES], data["Total"].astype(float))
    ridge_count_prediction = np.maximum(ridge_count_model.predict(candidates[COUNT_FEATURES]), 0)
    momentum_ridge_prediction = (
        (1 - selected_blend) * candidates["Previous_Observed_Total"].to_numpy(dtype=float)
        + selected_blend * ridge_count_prediction
    )

    forecast = candidates[
        ["Year", "NOC", "Olympic_Region", "WB_Country_Code", "WB_Country_Name", "Feature_Year",
         "GDP_USD", "Population", "Land_Area_Sq_Km", "Previous_Observed_Total",
         "Previous_Observed_Athlete_Count", "Is_Host"]
    ].copy()
    forecast["Predicted_Win_Probability"] = win_probability
    forecast["Predicted_Winner_At_Tuned_Threshold"] = (win_probability >= CLASSIFIER_THRESHOLD).astype("int8")
    forecast["Predicted_Medals_If_Winner"] = predicted_if_winner
    forecast["Two_Stage_Expected_Total_Prediction"] = win_probability * predicted_if_winner
    forecast["Ridge_Count_Prediction"] = ridge_count_prediction
    forecast["Expected_Total_Prediction"] = momentum_ridge_prediction
    forecast["Rounded_Total_Prediction"] = np.rint(forecast["Expected_Total_Prediction"]).astype(int)
    forecast["Ridge_Alpha_Selected_On_2012"] = selected_alpha
    forecast["Momentum_Blend_Weight_Selected_On_2012"] = selected_blend
    forecast["Classifier_Threshold"] = CLASSIFIER_THRESHOLD
    forecast = forecast.sort_values("Expected_Total_Prediction", ascending=False)
    forecast.to_csv(processed_dir / "forecast_2020.csv", index=False)
    return forecast


def evaluate_2020_forecast(processed_dir: Path, actuals_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the retrospective forecast with Tokyo 2020 medal totals.

    Tokyo used ROC for the Russian Olympic Committee. Map that delegation to
    the RUS NOC used in the historical forecast, and include actual medal
    winners absent from the forecast candidate list as zero-prediction misses.
    """
    processed_dir = Path(processed_dir)
    forecast_path = processed_dir / "forecast_2020.csv"
    if actuals_path is None:
        actuals_path = processed_dir.parent / "raw" / "tokyo_2020_medals_total.csv"
    actuals_path = Path(actuals_path)
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast not found: {forecast_path}. Run run_2020_forecast first.")
    if not actuals_path.exists():
        raise FileNotFoundError(f"Tokyo 2020 results not found: {actuals_path}")

    forecast = pd.read_csv(forecast_path)
    actuals = pd.read_csv(actuals_path)
    required_actuals = {"Country Code", "Gold Medal", "Silver Medal", "Bronze Medal", "Total"}
    if required_actuals.difference(actuals.columns):
        raise ValueError(f"Tokyo 2020 results need columns: {sorted(required_actuals)}")
    actuals = actuals.rename(columns={
        "Country Code": "NOC", "Gold Medal": "Actual_Gold", "Silver Medal": "Actual_Silver",
        "Bronze Medal": "Actual_Bronze", "Total": "Actual_Total",
    })
    actuals["NOC"] = actuals["NOC"].astype(str).str.strip()
    if actuals["NOC"].duplicated().any():
        raise ValueError("Tokyo results must contain one row per medal-winning NOC.")
    # ROC was the Russian delegation's code at Tokyo 2020; RUS is the stable
    # historical code in the assigned Olympic dataset and in our forecast.
    actuals["NOC"] = actuals["NOC"].replace({"ROC": "RUS"})
    if actuals["NOC"].duplicated().any():
        raise ValueError("NOC normalization created duplicate actual medal rows.")
    for column in ["Actual_Gold", "Actual_Silver", "Actual_Bronze", "Actual_Total"]:
        actuals[column] = pd.to_numeric(actuals[column], errors="raise")
    if not (actuals["Actual_Gold"] + actuals["Actual_Silver"] + actuals["Actual_Bronze"]).equals(actuals["Actual_Total"]):
        raise ValueError("Tokyo total medal counts do not equal gold + silver + bronze.")

    forecast = forecast.rename(columns={
        "Expected_Total_Prediction": "Predicted_Total", "Rounded_Total_Prediction": "Rounded_Predicted_Total",
        "Previous_Observed_Total": "Last_Games_Baseline",
        "Two_Stage_Expected_Total_Prediction": "Two_Stage_Predicted_Total",
    })
    if forecast["NOC"].duplicated().any():
        raise ValueError("Forecast must contain one row per NOC.")
    comparison = forecast.merge(
        actuals[["NOC", "Actual_Gold", "Actual_Silver", "Actual_Bronze", "Actual_Total"]],
        on="NOC", how="outer", validate="one_to_one", indicator=True,
    )
    comparison["Forecast_Status"] = comparison["_merge"].map({
        "both": "forecasted and won medals", "left_only": "forecasted; no Tokyo medal recorded",
        "right_only": "actual medal winner missing from forecast candidates",
    })
    comparison["Predicted_Total"] = comparison["Predicted_Total"].fillna(0.0)
    comparison["Two_Stage_Predicted_Total"] = comparison["Two_Stage_Predicted_Total"].fillna(0.0)
    comparison["Predicted_Win_Probability"] = comparison["Predicted_Win_Probability"].fillna(0.0)
    comparison["Predicted_Winner_At_Tuned_Threshold"] = comparison[
        "Predicted_Winner_At_Tuned_Threshold"
    ].fillna(0).astype("int8")
    comparison["Rounded_Predicted_Total"] = comparison["Rounded_Predicted_Total"].fillna(0).astype(int)
    comparison["Last_Games_Baseline"] = comparison["Last_Games_Baseline"].fillna(0.0)
    comparison["Actual_Total"] = comparison["Actual_Total"].fillna(0).astype(int)
    comparison["Actual_Gold"] = comparison["Actual_Gold"].fillna(0).astype(int)
    comparison["Actual_Silver"] = comparison["Actual_Silver"].fillna(0).astype(int)
    comparison["Actual_Bronze"] = comparison["Actual_Bronze"].fillna(0).astype(int)
    comparison = comparison.drop(columns="_merge").sort_values("Predicted_Total", ascending=False)
    comparison.to_csv(processed_dir / "forecast_2020_vs_actual.csv", index=False)

    actual_winner = comparison["Actual_Total"].gt(0).to_numpy()
    predicted_winner = comparison["Predicted_Winner_At_Tuned_Threshold"].astype(bool).to_numpy()
    top_countries = comparison.nlargest(10, "Actual_Total")
    metrics = [
        {"Model": "Momentum-adjusted Ridge", "Metric": "MAE total medals", "Value": _mae(comparison["Actual_Total"], comparison["Predicted_Total"])},
        {"Model": "Momentum-adjusted Ridge", "Metric": "RMSE total medals", "Value": _rmse(comparison["Actual_Total"], comparison["Predicted_Total"])},
        {"Model": "Two-stage classifier + Poisson", "Metric": "MAE total medals", "Value": _mae(comparison["Actual_Total"], comparison["Two_Stage_Predicted_Total"])},
        {"Model": "Two-stage classifier + Poisson", "Metric": "RMSE total medals", "Value": _rmse(comparison["Actual_Total"], comparison["Two_Stage_Predicted_Total"])},
        {"Model": "Last-observed-Games baseline", "Metric": "MAE total medals", "Value": _mae(comparison["Actual_Total"], comparison["Last_Games_Baseline"])},
        {"Model": "Last-observed-Games baseline", "Metric": "RMSE total medals", "Value": _rmse(comparison["Actual_Total"], comparison["Last_Games_Baseline"])},
        {"Model": "Momentum-adjusted Ridge", "Metric": "Residual standard deviation", "Value": _residual_std(comparison["Actual_Total"], comparison["Predicted_Total"])},
        {"Model": "Two-stage classifier + Poisson", "Metric": "Residual standard deviation", "Value": _residual_std(comparison["Actual_Total"], comparison["Two_Stage_Predicted_Total"])},
        {"Model": "Last-observed-Games baseline", "Metric": "Residual standard deviation", "Value": _residual_std(comparison["Actual_Total"], comparison["Last_Games_Baseline"])},
        {"Model": "Momentum-adjusted Ridge", "Metric": "Top-10 medal-country MSE", "Value": _rmse(top_countries["Actual_Total"], top_countries["Predicted_Total"]) ** 2},
        {"Model": "Two-stage classifier + Poisson", "Metric": "Top-10 medal-country MSE", "Value": _rmse(top_countries["Actual_Total"], top_countries["Two_Stage_Predicted_Total"]) ** 2},
        {"Model": "Last-observed-Games baseline", "Metric": "Top-10 medal-country MSE", "Value": _rmse(top_countries["Actual_Total"], top_countries["Last_Games_Baseline"]) ** 2},
        {"Model": "Logistic medal-winner classifier (threshold 0.56)", "Metric": "Predicted medal-winning NOCs", "Value": float(predicted_winner.sum())},
        {"Model": "Tokyo 2020 actuals", "Metric": "Medal-winning NOCs", "Value": float(actual_winner.sum())},
        {"Model": "Momentum-adjusted Ridge", "Metric": "Sum of predicted medal totals", "Value": float(comparison["Predicted_Total"].sum())},
        {"Model": "Two-stage classifier + Poisson", "Metric": "Sum of predicted medal totals", "Value": float(comparison["Two_Stage_Predicted_Total"].sum())},
        {"Model": "Tokyo 2020 actuals", "Metric": "Sum of actual medal totals", "Value": float(comparison["Actual_Total"].sum())},
    ]
    metrics.extend(
        {"Model": "Logistic medal-winner classifier (threshold 0.56)", "Metric": name, "Value": value}
        for name, value in _classification_metrics(
            actual_winner, predicted_winner, comparison["Predicted_Win_Probability"].to_numpy()
        ).items()
    )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(processed_dir / "forecast_2020_actual_metrics.csv", index=False)
    return metrics_frame, comparison


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    metrics, predictions = run_backtest(processed_dir)
    print("Chronological comparison: 2012 validation/model selection; 2016 final holdout")
    print(metrics.to_string(index=False))
    print("\nTop predicted medal totals in each test Games:")
    print(
        predictions.sort_values(["Year", "Expected_Total_Prediction"], ascending=[True, False])
        .groupby("Year", group_keys=False)
        .head(10)
        .to_string(index=False)
    )
    forecast = run_2020_forecast(processed_dir)
    print("\nRetrospective 2020 forecast using 2019 indicators and 2016 medal totals:")
    print(forecast.head(15).to_string(index=False))
    actual_metrics, actual_comparison = evaluate_2020_forecast(processed_dir)
    print("\nRetrospective 2020 forecast compared with Tokyo 2020 medal results:")
    print(actual_metrics.to_string(index=False))
    print("\nTop actual-versus-predicted medal totals:")
    print(actual_comparison[["NOC", "Predicted_Total", "Actual_Total", "Forecast_Status"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
