"""Repeatable offline tests for the assigned-paper model benchmark."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.paper_benchmark import FEATURES, _prepare_features, run_paper_benchmark


class PaperBenchmarkTests(unittest.TestCase):
    def _write_synthetic_data(self, processed_dir: Path) -> None:
        """Write a compact, balanced country-Games panel and World Bank table."""
        games = (1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016)
        country_count = 10
        country_rows = []
        for country_index in range(country_count):
            noc = f"T{country_index:02d}"
            for year in games:
                # Four countries win medals in every Games; counts vary enough
                # for the classifier and conditional-count models to be tested.
                total = 1 + country_index % 3 if country_index < 4 else 0
                country_rows.append(
                    {
                        "Year": year,
                        "NOC": noc,
                        "Olympic_Region": f"Synthetic Country {country_index:02d}",
                        "WB_Country_Code": noc,
                        "Feature_Year": year - 1,
                        "GDP_USD": (country_index + 1) * 1_000_000 + year * 100,
                        "Population": (country_index + 1) * 100_000 + year,
                        "Land_Area_Sq_Km": (country_index + 1) * 1_000,
                        "Athlete_Count": 20 + country_index,
                        "Total": total,
                    }
                )

        world_rows = []
        for year in range(1987, 2016):
            world_rows.append(
                {
                    "Country Code": "WLD",
                    "Year": year,
                    "GDP_USD": 100_000_000 + year * 10_000,
                    "Population": 1_000_000_000 + year * 100_000,
                }
            )
            for country_index in range(country_count):
                noc = f"T{country_index:02d}"
                world_rows.append(
                    {
                        "Country Code": noc,
                        "Year": year,
                        "GDP_USD": (country_index + 1) * 1_000_000 + year * 100,
                        "Population": (country_index + 1) * 100_000 + year,
                    }
                )

        processed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(country_rows).to_csv(
            processed_dir / "country_year_modeling_table.csv", index=False
        )
        pd.DataFrame(world_rows).to_csv(
            processed_dir / "world_bank_country_year_features.csv", index=False
        )

    def test_games_outcome_does_not_change_same_games_input_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            processed_dir = Path(temporary)
            self._write_synthetic_data(processed_dir)
            before = _prepare_features(processed_dir)

            country_path = processed_dir / "country_year_modeling_table.csv"
            country_games = pd.read_csv(country_path)
            changed_row = country_games["Year"].eq(2012) & country_games["NOC"].eq("T00")
            country_games.loc[changed_row, "Total"] = 99
            country_games.to_csv(country_path, index=False)
            after = _prepare_features(processed_dir)

            before_games = before.loc[before["Year"].eq(2012)].set_index("NOC")
            after_games = after.loc[after["Year"].eq(2012)].set_index("NOC")
            pd.testing.assert_frame_equal(
                before_games[FEATURES].sort_index(), after_games[FEATURES].sort_index()
            )
            self.assertNotEqual(
                before_games.loc["T00", "Total"], after_games.loc["T00", "Total"]
            )

    def test_benchmark_emits_validation_and_final_test_outputs_with_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            processed_dir = Path(temporary)
            self._write_synthetic_data(processed_dir)

            metrics, predictions, selected = run_paper_benchmark(processed_dir)

            self.assertEqual(set(selected), {"classifier", "regressor"})
            self.assertTrue(selected["classifier"])
            self.assertTrue(selected["regressor"])
            self.assertTrue({"validation", "final test"}.issubset(set(metrics["Stage"])))
            self.assertTrue(np.isfinite(metrics["Value"].to_numpy(dtype=float)).all())

            validation = metrics.loc[metrics["Stage"].eq("validation")]
            self.assertTrue(validation["Family"].eq("baseline").any())
            self.assertTrue(validation["Family"].eq("selected two-stage total").any())
            final_test = metrics.loc[metrics["Stage"].eq("final test")]
            self.assertTrue(final_test["Family"].eq("selected classifier").any())
            self.assertTrue(final_test["Family"].eq("selected two-stage total").any())
            self.assertEqual(set(predictions["Year"].unique()), {2016})
            self.assertEqual(len(predictions), 10)
            self.assertTrue(
                {"Actual_Total", "Predicted_Medal_Winner", "Predicted_Conditional_Medals",
                 "Predicted_Total", "Selected_Classifier", "Selected_Regressor"}.issubset(
                    predictions.columns
                )
            )
            self.assertTrue(predictions["Predicted_Total"].ge(0).all())
            self.assertTrue(
                (processed_dir / "paper_model_benchmarks.csv").is_file()
            )
            self.assertTrue(
                (processed_dir / "paper_model_2016_predictions.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
