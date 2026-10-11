"""Small, offline regression tests for the Olympic prediction pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_preparation import _build_crosswalk, build_modeling_table
from src.modeling import evaluate_2020_forecast, run_2020_forecast, run_backtest
from src.paper_benchmark import FEATURES as PAPER_FEATURES, _prepare_features


class PipelineIntegrationTests(unittest.TestCase):
    def test_taiwan_noc_is_kept_even_when_world_bank_indicators_are_not_in_export(self) -> None:
        crosswalk = _build_crosswalk(
            pd.DataFrame({"NOC": ["TPE"]}),
            pd.DataFrame({"NOC": ["TPE"], "region": ["Taiwan"]}),
            pd.DataFrame({"Country Code": ["USA"], "Country Name": ["United States"]}),
        )
        self.assertEqual(crosswalk.loc[0, "WB_Country_Code"], "TWN")
        self.assertIn("no World Bank profile", crosswalk.loc[0, "Match_Type"])

    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        raw_dir = root / "data" / "raw"
        processed_dir = root / "data" / "processed"
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)

        countries = [
            (f"C{i:02d}", f"Test Country {i:02d}") for i in range(14)
        ]
        athlete_rows = []
        for year in (2008, 2012, 2016):
            for index, (noc, _) in enumerate(countries):
                # Two athletes in the team event must count as one medal.
                athlete_rows.extend(
                    [
                        {"Name": f"{noc}-{year}-A", "Season": "Summer", "Year": year,
                         "NOC": noc, "Sport": "Test sport", "Event": "Team event",
                         "Medal": "Gold" if index < 7 else None},
                        {"Name": f"{noc}-{year}-B", "Season": "Summer", "Year": year,
                         "NOC": noc, "Sport": "Test sport", "Event": "Team event",
                         "Medal": "Gold" if index < 7 else None},
                        {"Name": f"{noc}-{year}-C", "Season": "Summer", "Year": year,
                         "NOC": noc, "Sport": "Test sport", "Event": "Individual event",
                         "Medal": "Silver" if index < 4 else None},
                    ]
                )
        pd.DataFrame(athlete_rows).to_csv(raw_dir / "athlete_events.csv", index=False)
        pd.DataFrame(
            [{"NOC": noc, "region": name} for noc, name in countries]
        ).to_csv(raw_dir / "noc_regions.csv", index=False)

        years = ("2007", "2011", "2015", "2019")
        indicator_codes = ("NY.GDP.MKTP.CD", "SP.POP.TOTL", "AG.LND.TOTL.K2")
        wb_rows = []
        for index, (noc, name) in enumerate(countries):
            for series_code in indicator_codes:
                base = (index + 1) * 1_000_000
                wb_rows.append(
                    {
                        "Country Name": name,
                        "Country Code": noc,
                        "Series Name": series_code,
                        "Series Code": series_code,
                        **{f"{year} [YR{year}]": base + int(year) for year in years},
                    }
                )
        pd.DataFrame(wb_rows).to_csv(raw_dir / "synthetic_Data.csv", index=False)
        return raw_dir, processed_dir

    def test_preparation_counts_team_medal_once_and_keeps_zero_medal_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir, processed_dir = self._write_fixture(Path(temporary))
            table, _ = build_modeling_table(raw_dir, processed_dir)

            first_country = table.loc[(table["NOC"] == "C00") & (table["Year"] == 2012)].iloc[0]
            no_medal_country = table.loc[(table["NOC"] == "C13") & (table["Year"] == 2012)].iloc[0]
            self.assertEqual(first_country["Gold"], 1)
            self.assertEqual(first_country["Silver"], 1)
            self.assertEqual(first_country["Total"], 2)
            self.assertEqual(first_country["Athlete_Count"], 3)
            self.assertEqual(no_medal_country["Total"], 0)
            self.assertEqual(int(first_country["Feature_Year"]), 2011)
            self.assertFalse(table.duplicated(["Year", "NOC"]).any())

    def test_backtest_and_2020_forecast_run_on_a_small_synthetic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir, processed_dir = self._write_fixture(Path(temporary))
            build_modeling_table(raw_dir, processed_dir)

            paper_features = _prepare_features(processed_dir)
            first_2012 = paper_features.loc[
                paper_features["NOC"].eq("C00") & paper_features["Year"].eq(2012)
            ].iloc[0]
            self.assertEqual(first_2012["Previous_Observed_Total"], 2)
            self.assertEqual(first_2012["Previous_Observed_Athlete_Count"], 3)
            self.assertTrue(set(PAPER_FEATURES).issubset(paper_features.columns))
            self.assertNotIn("Total", PAPER_FEATURES)

            metrics, predictions = run_backtest(processed_dir)
            self.assertIn(2012, predictions["Year"].unique())
            self.assertIn(2016, predictions["Year"].unique())
            self.assertIn("Evaluation_Role", pd.read_csv(processed_dir / "walk_forward_model_comparison.csv"))
            self.assertTrue(metrics["Value"].dropna().map(lambda value: value == value).all())

            forecast = run_2020_forecast(processed_dir)
            self.assertTrue(forecast["Year"].eq(2020).all())
            self.assertTrue(forecast["Feature_Year"].eq(2019).all())
            self.assertTrue((forecast["Previous_Observed_Total"] >= 0).all())

            actuals_path = raw_dir / "tokyo_2020_medals_total.csv"
            pd.DataFrame(
                [{"Country Code": "C00", "Gold Medal": 1, "Silver Medal": 1,
                  "Bronze Medal": 0, "Total": 2}]
            ).to_csv(actuals_path, index=False)
            actual_metrics, comparison = evaluate_2020_forecast(processed_dir, actuals_path)
            self.assertFalse(comparison["NOC"].duplicated().any())
            self.assertIn("Momentum-adjusted Ridge", actual_metrics["Model"].unique())


if __name__ == "__main__":
    unittest.main()
