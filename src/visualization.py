"""Create presentation-ready figures from the generated evaluation tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_figures(project_root: Path) -> list[Path]:
    """Save a Tokyo 2020 forecast comparison and a 2016 holdout comparison."""
    project_root = Path(project_root)
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tokyo_path = processed_dir / "forecast_2020_vs_actual.csv"
    holdout_path = processed_dir / "walk_forward_model_comparison.csv"
    for required in (tokyo_path, holdout_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Evaluation output not found: {required}. Run `python -m src.modeling` first."
            )

    tokyo = pd.read_csv(tokyo_path).nlargest(10, "Actual_Total").sort_values("Actual_Total")
    labels = tokyo["NOC"].astype(str)
    positions = range(len(tokyo))
    bar_height = 0.24
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.barh(
        [position - bar_height for position in positions],
        tokyo["Actual_Total"],
        height=bar_height,
        label="Actual Tokyo 2020",
        color="#263b5a",
    )
    axis.barh(
        positions,
        tokyo["Predicted_Total"],
        height=bar_height,
        label="Momentum-adjusted Ridge",
        color="#e08b36",
    )
    axis.barh(
        [position + bar_height for position in positions],
        tokyo["Last_Games_Baseline"],
        height=bar_height,
        label="2016 baseline",
        color="#70a6a1",
    )
    axis.set_yticks(list(positions), labels)
    axis.set_xlabel("Total medals")
    axis.set_title("Tokyo 2020: forecast vs actual medal totals")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    tokyo_figure = figures_dir / "tokyo_2020_forecast_vs_actual.png"
    figure.savefig(tokyo_figure, dpi=180, bbox_inches="tight")
    plt.close(figure)

    holdout = pd.read_csv(holdout_path)
    holdout = holdout.loc[holdout["Year"].eq(2016)].nlargest(10, "Actual_Total")
    holdout = holdout.sort_values("Actual_Total")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    positions = range(len(holdout))
    labels = holdout["NOC"].astype(str)
    axis.barh(
        [position - bar_height / 2 for position in positions],
        holdout["Actual_Total"],
        height=bar_height,
        label="Actual Rio 2016",
        color="#263b5a",
    )
    axis.barh(
        [position + bar_height / 2 for position in positions],
        holdout["Momentum_Ridge_Prediction"],
        height=bar_height,
        label="Momentum-adjusted Ridge",
        color="#e08b36",
    )
    axis.set_yticks(list(positions), labels)
    axis.set_xlabel("Total medals")
    axis.set_title("2016 final holdout: actual vs predicted top medal NOCs")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    holdout_figure = figures_dir / "rio_2016_holdout_predictions.png"
    figure.savefig(holdout_figure, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return [tokyo_figure, holdout_figure]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for path in create_figures(project_root):
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
