"""Compare the assigned paper's model families with a leakage-aware split.

The original report uses features that are not available before the Games
(notably same-Games athlete counts). This benchmark uses the previous Games'
delegation size as a pre-Games proxy and documents that adaptation in its
outputs and README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_END = 2008
VALIDATION_YEAR = 2012
TEST_YEAR = 2016
FEATURES = [
    "Year",
    "Log_GDP_USD",
    "Log_GDP_Per_Capita",
    "GDP_Growth",
    "World_GDP_Share",
    "Log_Population",
    "Population_Growth",
    "World_Population_Share",
    "Log_Land_Area_Sq_Km",
    "Previous_Observed_Total",
    "Previous_Observed_Athlete_Count",
    "Previous_Observed_Total_2",
]


def _prepare_features(processed_dir: Path) -> pd.DataFrame:
    data_path = processed_dir / "country_year_modeling_table.csv"
    world_path = processed_dir / "world_bank_country_year_features.csv"
    if not data_path.exists() or not world_path.exists():
        raise FileNotFoundError("Run src.data_preparation before the paper benchmark.")

    data = pd.read_csv(data_path).sort_values(["NOC", "Year"]).copy()
    world = pd.read_csv(world_path)
    lagged = data.groupby("NOC", sort=False)
    data["Previous_Observed_Total"] = lagged["Total"].shift(1).fillna(0)
    data["Previous_Observed_Total_2"] = lagged["Total"].shift(2).fillna(0)
    data["Previous_Observed_Athlete_Count"] = lagged["Athlete_Count"].shift(1).fillna(0)

    wb_cols = ["Country Code", "Year", "GDP_USD", "Population"]
    wb = world[wb_cols].copy()
    previous = wb.assign(Year=wb["Year"] + 1).rename(
        columns={"Year": "Feature_Year", "Country Code": "WB_Country_Code",
                 "GDP_USD": "Previous_Year_GDP", "Population": "Previous_Year_Population"}
    )
    data = data.merge(
        previous, on=["WB_Country_Code", "Feature_Year"], how="left", validate="many_to_one"
    )

    world_totals = world.loc[world["Country Code"].eq("WLD"),
                             ["Year", "GDP_USD", "Population"]].rename(
        columns={"Year": "Feature_Year", "GDP_USD": "World_GDP_USD",
                 "Population": "World_Population"}
    )
    data = data.merge(world_totals, on="Feature_Year", how="left", validate="many_to_one")
    data["Log_GDP_USD"] = np.log1p(data["GDP_USD"].where(data["GDP_USD"].ge(0)))
    data["Log_Population"] = np.log1p(data["Population"].where(data["Population"].ge(0)))
    data["Log_Land_Area_Sq_Km"] = np.log1p(
        data["Land_Area_Sq_Km"].where(data["Land_Area_Sq_Km"].ge(0))
    )
    data["Log_GDP_Per_Capita"] = np.log1p(
        (data["GDP_USD"] / data["Population"]).where(
            data["GDP_USD"].ge(0) & data["Population"].gt(0)
        )
    )
    data["GDP_Growth"] = data["GDP_USD"] / data["Previous_Year_GDP"] - 1
    data["Population_Growth"] = data["Population"] / data["Previous_Year_Population"] - 1
    data["World_GDP_Share"] = data["GDP_USD"] / data["World_GDP_USD"]
    data["World_Population_Share"] = data["Population"] / data["World_Population"]
    data["Won_Any_Medal"] = data["Total"].gt(0).astype("int8")
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    # Host is useful context but not among the original paper's listed features.
    return data.loc[data["Year"].ge(1988)].reset_index(drop=True)


def _scoring(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "Residual_SD": float(np.std(residual)),
    }


def _classification_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "Balanced_Accuracy": float(balanced_accuracy_score(actual, predicted)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
    }


def run_paper_benchmark(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Tune paper model families on pre-2012 data, select on 2012, test on 2016."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, PoissonRegressor, Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    data = _prepare_features(Path(processed_dir))
    train = data.loc[data["Year"].le(TRAIN_END)].copy()
    validation = data.loc[data["Year"].eq(VALIDATION_YEAR)].copy()
    test = data.loc[data["Year"].eq(TEST_YEAR)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Paper benchmark requires 1988-2008 training, 2012 validation, and 2016 test data.")

    cv = GroupKFold(n_splits=3)
    classifier_specs = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                                {"model__C": [0.1, 1.0, 10.0]}),
        "SVM": (SVC(class_weight="balanced"),
                {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]}),
        "Gaussian Naive Bayes": (GaussianNB(), {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}),
        "MLP": (MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
                {"model__hidden_layer_sizes": [(32,), (64,), (32, 16)],
                 "model__alpha": [0.0001, 0.01]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  random_state=42, n_jobs=-1),
                          {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 3]}),
    }
    regressor_specs = {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "Lasso": (Lasso(max_iter=10000), {"model__alpha": [0.01, 0.1, 1.0]}),
        "Poisson": (PoissonRegressor(max_iter=2000), {"model__alpha": [0.1, 1.0, 10.0]}),
        "SVR": (SVR(), {"model__C": [1.0, 10.0], "model__kernel": ["linear", "rbf"]}),
        "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                          {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 3]}),
    }

    def tune_models(specs: dict, frame: pd.DataFrame, target: str, scoring: str) -> dict:
        results = {}
        for name, (estimator, grid) in specs.items():
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ])
            search = GridSearchCV(
                pipeline, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True,
                error_score="raise", return_train_score=False,
            )
            search.fit(frame[FEATURES], frame[target], groups=frame["Year"])
            results[name] = (search.best_estimator_, search.best_params_)
        return results

    tuned_classifiers = tune_models(classifier_specs, train, "Won_Any_Medal", "accuracy")
    train_winners = train.loc[train["Total"].gt(0)].copy()
    if train_winners.empty:
        raise ValueError("No medal-winning training countries are available for regression.")
    tuned_regressors = tune_models(regressor_specs, train_winners, "Total", "neg_mean_absolute_error")

    metric_rows = []
    validation_classifier_rank = []
    classifier_validation_predictions = {}
    for name, (model, params) in tuned_classifiers.items():
        predicted = model.predict(validation[FEATURES]).astype(int)
        scores = _classification_scores(validation["Won_Any_Medal"].to_numpy(), predicted)
        validation_classifier_rank.append((scores["Accuracy"], scores["Balanced_Accuracy"], name))
        classifier_validation_predictions[name] = predicted
        metric_rows.extend({"Year": VALIDATION_YEAR, "Stage": "validation", "Family": "classifier",
                            "Model": name, "Metric": metric, "Value": value,
                            "Best_Parameters": str(params)} for metric, value in scores.items())

    validation_regressor_rank = []
    regressor_validation_predictions = {}
    validation_winners = validation.loc[validation["Total"].gt(0)]
    for name, (model, params) in tuned_regressors.items():
        predicted = np.maximum(model.predict(validation_winners[FEATURES]), 0)
        scores = _scoring(validation_winners["Total"].to_numpy(), predicted)
        validation_regressor_rank.append((scores["Residual_SD"], scores["MAE"], name))
        regressor_validation_predictions[name] = (model, scores)
        metric_rows.extend({"Year": VALIDATION_YEAR, "Stage": "validation", "Family": "positive-count regressor",
                            "Model": name, "Metric": metric, "Value": value,
                            "Best_Parameters": str(params)} for metric, value in scores.items())

    selected_classifier = min(validation_classifier_rank, key=lambda item: (-item[0], -item[1], item[2]))[2]
    selected_regressor = min(validation_regressor_rank)[2]
    selected_params = {
        "classifier": selected_classifier,
        "regressor": selected_regressor,
    }

    validation_classifier = tuned_classifiers[selected_classifier][0]
    validation_regressor = tuned_regressors[selected_regressor][0]
    validation_winner_flag = validation_classifier.predict(validation[FEATURES]).astype(int)
    validation_conditional_count = np.maximum(
        validation_regressor.predict(validation[FEATURES]), 0
    )
    validation_staged_total = validation_winner_flag * validation_conditional_count
    for metric, value in _scoring(validation["Total"].to_numpy(), validation_staged_total).items():
        metric_rows.append({"Year": VALIDATION_YEAR, "Stage": "validation", "Family": "selected two-stage total",
                            "Model": f"{selected_classifier} + {selected_regressor}",
                            "Metric": metric, "Value": value, "Best_Parameters": "selected on 2012"})
    for year, frame in ((VALIDATION_YEAR, validation), (TEST_YEAR, test)):
        for metric, value in _scoring(
            frame["Total"].to_numpy(), frame["Previous_Observed_Total"].to_numpy()
        ).items():
            metric_rows.append({"Year": year, "Stage": "validation" if year == VALIDATION_YEAR else "final test",
                                "Family": "baseline", "Model": "Previous Games medal total",
                                "Metric": metric, "Value": value, "Best_Parameters": "not tuned"})

    # Refit the validation-selected pair using the paper's tuning + validation
    # data, then evaluate exactly once on the final 2016 holdout.
    refit_data = data.loc[data["Year"].le(VALIDATION_YEAR)].copy()
    refit_winners = refit_data.loc[refit_data["Total"].gt(0)]
    classifier = clone(tuned_classifiers[selected_classifier][0]).fit(
        refit_data[FEATURES], refit_data["Won_Any_Medal"]
    )
    regressor = clone(tuned_regressors[selected_regressor][0]).fit(
        refit_winners[FEATURES], refit_winners["Total"]
    )
    predicted_winner = classifier.predict(test[FEATURES]).astype(int)
    conditional_count = np.maximum(regressor.predict(test[FEATURES]), 0)
    staged_total = predicted_winner * conditional_count
    for metric, value in _classification_scores(
        test["Won_Any_Medal"].to_numpy(), classifier.predict(test[FEATURES]).astype(int)
    ).items():
        metric_rows.append({"Year": TEST_YEAR, "Stage": "final test", "Family": "selected classifier",
                            "Model": selected_classifier, "Metric": metric, "Value": value,
                            "Best_Parameters": str(tuned_classifiers[selected_classifier][1])})
    for metric, value in _scoring(test["Total"].to_numpy(), staged_total).items():
        metric_rows.append({"Year": TEST_YEAR, "Stage": "final test", "Family": "selected two-stage total",
                            "Model": f"{selected_classifier} + {selected_regressor}",
                            "Metric": metric, "Value": value, "Best_Parameters": "selected on 2012"})

    predictions = test[["Year", "NOC", "Olympic_Region", "Total"]].rename(
        columns={"Total": "Actual_Total"}
    ).copy()
    predictions["Predicted_Medal_Winner"] = predicted_winner
    predictions["Predicted_Conditional_Medals"] = conditional_count
    predictions["Predicted_Total"] = staged_total
    predictions["Selected_Classifier"] = selected_classifier
    predictions["Selected_Regressor"] = selected_regressor

    metrics = pd.DataFrame(metric_rows)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    metrics.to_csv(Path(processed_dir) / "paper_model_benchmarks.csv", index=False)
    predictions.to_csv(Path(processed_dir) / "paper_model_2016_predictions.csv", index=False)
    return metrics, predictions, selected_params


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metrics, _, selected = run_paper_benchmark(project_root / "data" / "processed")
    print("Assigned-paper model-family comparison; 2012 validation and untouched 2016 test")
    print(f"Selected on 2012: {selected}")
    print(metrics.to_string(index=False))
    print("Saved paper_model_benchmarks.csv and paper_model_2016_predictions.csv")


if __name__ == "__main__":
    main()
