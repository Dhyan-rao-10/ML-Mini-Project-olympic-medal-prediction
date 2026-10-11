"""Prepare country-Games medal targets and pre-Games World Bank features."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP_USD",
    "SP.POP.TOTL": "Population",
    "AG.LND.TOTL.K2": "Land_Area_Sq_Km",
}

# Codes whose Olympic delegation cannot be represented by one World Bank country.
NON_COUNTRY_NOCS = {
    "AHO",  # Netherlands Antilles, later split into several territories
    "EUN",  # Unified Team
    "FRG",  # West Germany
    "GDR",  # East Germany
    "IOA",  # Individual Olympic Athletes
    "ROT",  # Refugee Olympic Team
    "SCG",  # Serbia and Montenegro
    "TCH",  # Czechoslovakia
    "UAR",  # United Arab Republic
    "UNK",  # Unidentified NOC
    "URS",  # Soviet Union
    "WIF",  # West Indies Federation
    "YAR",  # North Yemen
    "YMD",  # South Yemen
    "YUG",  # Yugoslavia
}

# NOC names/codes that do not match the World Bank export's country names/codes.
NOC_TO_WB_OVERRIDES = {
    "ANT": "ATG",  # Antigua and Barbuda
    "BAH": "BHS",  # Bahamas
    "BRU": "BRN",  # Brunei Darussalam
    "CGO": "COG",  # Republic of the Congo
    "GAM": "GMB",  # The Gambia
    "HKG": "HKG",  # Hong Kong, listed as China in the Kaggle region field
    "IRI": "IRN",  # Iran
    "IVB": "VGB",  # British Virgin Islands
    "PLE": "PSE",  # West Bank and Gaza
    "PUR": "PRI",  # Puerto Rico
    "SKN": "KNA",  # St. Kitts and Nevis
    "TPE": "TWN",  # Chinese Taipei / Taiwan (World Bank code)
    "VIN": "VCT",  # St. Vincent and the Grenadines
}


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _load_world_bank(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(raw_dir.glob("*_Data.csv"))
    renamed_export = raw_dir / "world_bank_data.csv"
    if renamed_export.exists() and renamed_export not in files:
        files.append(renamed_export)
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one World Bank *_Data.csv in {raw_dir}; found {len(files)}."
        )

    raw = pd.read_csv(files[0], dtype={"Country Code": "string", "Series Code": "string"})
    required_columns = {"Country Name", "Country Code", "Series Code"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(
            "World Bank export is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    year_columns = [c for c in raw.columns if re.fullmatch(r"\d{4} \[YR\d{4}\]", c)]
    if not year_columns:
        raise ValueError("No World Bank year columns were found in the DataBank CSV.")

    core = raw.dropna(subset=["Country Code"]).loc[
        lambda frame: frame["Series Code"].isin(INDICATORS)
    ].copy()
    missing_indicators = set(INDICATORS).difference(core["Series Code"].dropna().unique())
    if missing_indicators:
        raise ValueError(
            "World Bank export is missing required indicator codes: "
            f"{sorted(missing_indicators)}"
        )
    duplicate_keys = core.duplicated(["Country Code", "Series Code"], keep=False)
    if duplicate_keys.any():
        duplicate_rows = core.loc[duplicate_keys, ["Country Code", "Series Code"]]
        raise ValueError(f"Duplicate World Bank indicator rows found:\n{duplicate_rows}")

    long = core.melt(
        id_vars=["Country Name", "Country Code", "Series Code"],
        value_vars=year_columns,
        var_name="Year",
        value_name="Value",
    )
    long["Year"] = long["Year"].str.extract(r"(\d{4})").astype("int16")
    long["Value"] = pd.to_numeric(long["Value"].replace("..", pd.NA), errors="coerce")
    features = (
        long.pivot(
            index=["Country Name", "Country Code", "Year"],
            columns="Series Code",
            values="Value",
        )
        .rename(columns=INDICATORS)
        .reset_index()
    )
    features.columns.name = None
    return raw, features


def _build_crosswalk(
    country_year: pd.DataFrame,
    noc_regions: pd.DataFrame,
    wb_features: pd.DataFrame,
) -> pd.DataFrame:
    wb_countries = (
        wb_features[["Country Code", "Country Name"]]
        .drop_duplicates()
        .sort_values(["Country Name", "Country Code"])
    )
    code_to_name = dict(zip(wb_countries["Country Code"], wb_countries["Country Name"]))
    codes = set(code_to_name)
    name_to_code: dict[str, str] = {}
    for code, name in zip(wb_countries["Country Code"], wb_countries["Country Name"]):
        name_to_code.setdefault(_normalize_name(name), code)

    noc_names = noc_regions[["NOC", "region"]].drop_duplicates("NOC")
    noc_names = country_year[["NOC"]].drop_duplicates().merge(noc_names, on="NOC", how="left")
    rows = []
    for row in noc_names.itertuples(index=False):
        noc = str(row.NOC)
        region = row.region
        wb_code = None
        match_type = "unmatched"

        if noc not in NON_COUNTRY_NOCS:
            if noc in NOC_TO_WB_OVERRIDES:
                candidate = NOC_TO_WB_OVERRIDES[noc]
                wb_code = candidate
                match_type = (
                    "manual NOC crosswalk"
                    if candidate in codes
                    else "manual NOC crosswalk; no World Bank profile in selected export"
                )
            elif pd.notna(region):
                wb_code = name_to_code.get(_normalize_name(region))
                if wb_code is not None:
                    match_type = "NOC region matches World Bank country name"
            if wb_code is None and noc in codes:
                # Some NOC codes collide with a different World Bank ISO3 code (e.g. BRN).
                # Prefer a matching country name, then use the code only as a fallback.
                wb_code = noc
                match_type = "NOC code matches World Bank code"

        rows.append(
            {
                "NOC": noc,
                "Olympic_Region": region,
                "WB_Country_Code": wb_code,
                "WB_Country_Name": code_to_name.get(wb_code),
                "Match_Type": match_type,
            }
        )

    return pd.DataFrame(rows).sort_values("NOC").reset_index(drop=True)


def build_modeling_table(raw_dir: Path, processed_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Create a country-Games table using indicators from the year before each Games."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    athletes = pd.read_csv(raw_dir / "athlete_events.csv")
    noc_regions = pd.read_csv(raw_dir / "noc_regions.csv")
    required_athlete_columns = {"Name", "Season", "Year", "NOC", "Sport", "Event", "Medal"}
    missing_athlete_columns = required_athlete_columns.difference(athletes.columns)
    if missing_athlete_columns:
        raise ValueError(
            "Olympic athlete file is missing required columns: "
            f"{sorted(missing_athlete_columns)}"
        )
    required_region_columns = {"NOC", "region"}
    missing_region_columns = required_region_columns.difference(noc_regions.columns)
    if missing_region_columns:
        raise ValueError(
            "NOC region file is missing required columns: "
            f"{sorted(missing_region_columns)}"
        )

    summer = athletes.loc[athletes["Season"].eq("Summer") & athletes["Year"].ne(1906)].copy()
    summer = summer.drop_duplicates()
    medal_rows = summer.loc[summer["Medal"].notna()].copy()
    invalid_medals = set(medal_rows["Medal"].dropna().unique()).difference(
        {"Gold", "Silver", "Bronze"}
    )
    if invalid_medals:
        raise ValueError(f"Unexpected medal labels in Olympic data: {sorted(invalid_medals)}")
    event_medals = medal_rows.drop_duplicates(
        ["Year", "NOC", "Sport", "Event", "Medal"]
    )
    medal_counts = (
        event_medals.groupby(["Year", "NOC", "Medal"])
        .size()
        .unstack(fill_value=0)
    )
    for medal in ["Gold", "Silver", "Bronze"]:
        if medal not in medal_counts.columns:
            medal_counts[medal] = 0
    medal_counts["Total"] = medal_counts[["Gold", "Silver", "Bronze"]].sum(axis=1)
    medal_counts = medal_counts.reset_index()

    participation = summer[["Year", "NOC"]].drop_duplicates()
    athlete_counts = (
        summer.groupby(["Year", "NOC"])["Name"]
        .nunique()
        .rename("Athlete_Count")
        .reset_index()
    )
    country_year = participation.merge(
        medal_counts, on=["Year", "NOC"], how="left", validate="one_to_one"
    )
    country_year = country_year.merge(
        athlete_counts, on=["Year", "NOC"], how="left", validate="one_to_one"
    )
    for medal in ["Gold", "Silver", "Bronze", "Total"]:
        country_year[medal] = country_year[medal].fillna(0).astype("int64")
    country_year = country_year.merge(
        noc_regions[["NOC", "region"]].drop_duplicates("NOC"),
        on="NOC",
        how="left",
        validate="many_to_one",
    )
    region_overrides = {"SGP": "Singapore", "TUV": "Tuvalu"}
    country_year["region"] = country_year["region"].where(
        country_year["region"].notna(), country_year["NOC"].map(region_overrides)
    )

    _, wb_features = _load_world_bank(raw_dir)
    feature_years = set(wb_features["Year"].unique())
    candidate_country_year = country_year.loc[
        (country_year["Year"] - 1).isin(feature_years)
    ].copy()
    crosswalk = _build_crosswalk(candidate_country_year, noc_regions, wb_features)
    mapped = candidate_country_year.merge(
        crosswalk[["NOC", "WB_Country_Code", "WB_Country_Name", "Match_Type"]],
        on="NOC",
        how="left",
        validate="many_to_one",
    )
    mapped["Feature_Year"] = mapped["Year"] - 1
    wb_for_merge = wb_features.rename(
        columns={"Country Code": "WB_Country_Code", "Country Name": "WB_Country_Name", "Year": "Feature_Year"}
    )
    merged = mapped.merge(
        wb_for_merge,
        on=["WB_Country_Code", "WB_Country_Name", "Feature_Year"],
        how="left",
        validate="many_to_one",
    )

    available = merged.loc[
        merged["WB_Country_Code"].notna() & merged["Feature_Year"].isin(feature_years)
    ].copy()
    available = available.rename(columns={"region": "Olympic_Region"})

    processed_dir.mkdir(parents=True, exist_ok=True)
    available.to_csv(processed_dir / "country_year_modeling_table.csv", index=False)
    crosswalk.to_csv(processed_dir / "noc_world_bank_crosswalk.csv", index=False)
    wb_features.to_csv(processed_dir / "world_bank_country_year_features.csv", index=False)

    diagnostics = {
        "Olympic country-Games rows before feature matching": len(country_year),
        "World Bank indicator rows after reshaping": len(wb_features),
        "Olympic rows with a World Bank prior-year available": len(candidate_country_year),
        "Olympic rows with a World Bank country mapping and prior-year record": len(available),
        "Unmapped or non-country NOCs": crosswalk.loc[
            crosswalk["WB_Country_Code"].isna(), ["NOC", "Olympic_Region", "Match_Type"]
        ].to_dict("records"),
        "Missing GDP values after merge": int(available["GDP_USD"].isna().sum()),
        "Missing population values after merge": int(available["Population"].isna().sum()),
        "Missing land-area values after merge": int(available["Land_Area_Sq_Km"].isna().sum()),
        "First Games year in modeling table": int(available["Year"].min()) if len(available) else None,
        "Last Games year in modeling table": int(available["Year"].max()) if len(available) else None,
        "Output file": str(processed_dir / "country_year_modeling_table.csv"),
        "World Bank country-year features file": str(processed_dir / "world_bank_country_year_features.csv"),
        "NOC crosswalk file": str(processed_dir / "noc_world_bank_crosswalk.csv"),
    }
    return available, diagnostics


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    _, diagnostics = build_modeling_table(
        project_root / "data" / "raw",
        project_root / "data" / "processed",
    )
    for label, value in diagnostics.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
