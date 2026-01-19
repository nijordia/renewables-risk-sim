"""
Renewables Risk Simulator - Data Fetching Module

Fetches electricity price and generation data from REData API (apidatos.ree.es).
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

from utils import fetch_with_retry, date_range_chunks, get_season, print_header


# API Endpoints
PRICES_URL = (
    "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
    "?start_date={start}&end_date={end}&time_trunc=hour"
)
BALANCE_URL = (
    "https://apidatos.ree.es/es/datos/balance/balance-electrico"
    "?start_date={start}&end_date={end}&time_trunc=day"
)

# Generation source mappings
RENEWABLE_SOURCES = {
    "10288": "hydro",           # Hidraulica
    "10291": "wind",            # Eolica
    "1458": "solar_pv",         # Solar fotovoltaica
    "1459": "solar_thermal",    # Solar termica
    "1455": "hydro_wind",       # Hidroeolica
    "10292": "other_renewable", # Otras renovables
    "10295": "renewable_waste", # Residuos renovables
}

NON_RENEWABLE_SOURCES = {
    "1446": "nuclear",          # Nuclear
    "1454": "combined_cycle",   # Ciclo combinado (gas)
    "10289": "coal",            # Carbon
    "10344": "diesel",          # Motores diesel
    "1450": "gas_turbine",      # Turbina de gas
    "1451": "steam_turbine",    # Turbina de vapor
    "10293": "cogeneration",    # Cogeneracion
    "10294": "non_renew_waste", # Residuos no renovables
}

STORAGE_SOURCES = {
    "1445": "pumped_hydro_gen",  # Turbinacion bombeo
    "1472": "pumped_hydro_cons", # Consumo bombeo
    "2180": "battery_discharge", # Entrega bateria
    "2181": "battery_charge",    # Carga bateria
}

ALL_SOURCES = {**RENEWABLE_SOURCES, **NON_RENEWABLE_SOURCES, **STORAGE_SOURCES}


def fetch_spot_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch spot market electricity prices from REData API.

    The API returns 15-minute interval data for "Precio mercado spot" (id: 600).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: datetime, price_eur_mwh
    """
    print(f"Fetching spot market prices from {start_date} to {end_date}...")

    all_records = []
    chunks = list(date_range_chunks(start_date, end_date))

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        url = PRICES_URL.format(start=chunk_start, end=chunk_end)
        print(f"  Requesting: {chunk_start[:10]} to {chunk_end[:10]}")

        try:
            data = fetch_with_retry(url)
        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch chunk {chunk_start} - {chunk_end}: {e}")
            time.sleep(1)
            continue

        if "included" not in data:
            print(f"  Warning: No 'included' data in response for {chunk_start}")
            continue

        # Find "Precio mercado spot" (id: "600") - the wholesale spot price
        for item in data.get("included", []):
            item_id = item.get("id", "")
            item_type = item.get("type", "")

            if item_id == "600" or "spot" in item_type.lower():
                values = item.get("attributes", {}).get("values", [])
                for val in values:
                    if "datetime" in val and "value" in val:
                        all_records.append({
                            "datetime": val["datetime"],
                            "price_eur_mwh": val["value"]
                        })
                if values:
                    break

        # Rate limit between requests
        if i < len(chunks) - 1:
            time.sleep(0.5)

    if not all_records:
        raise ValueError("No price data retrieved from API")

    df = pd.DataFrame(all_records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])

    print(f"  Retrieved {len(df)} price records (15-min intervals)")
    return df


def aggregate_prices_to_daily(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate price data (15-min or hourly intervals) to daily averages.

    Args:
        prices_df: DataFrame with datetime and price_eur_mwh columns

    Returns:
        DataFrame with columns: date, price_daily_avg
    """
    df = prices_df.copy()
    df["date"] = df["datetime"].dt.date

    daily = df.groupby("date").agg(
        price_daily_avg=("price_eur_mwh", "mean"),
        price_count=("price_eur_mwh", "count")
    ).reset_index()

    # Filter out days with incomplete data (require at least 20 records)
    min_records = 20
    daily = daily[daily["price_count"] >= min_records].drop(columns=["price_count"])
    daily["date"] = pd.to_datetime(daily["date"])

    print(f"  Aggregated to {len(daily)} daily price records")
    return daily


def fetch_generation_balance(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily generation balance from REData API.

    The API returns a nested structure with categories (Renovable, No-Renovable, Demanda)
    containing individual generators.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with all generation sources and aggregate shares
    """
    print(f"Fetching generation balance from {start_date} to {end_date}...")

    def empty_record():
        record = {name: 0 for name in ALL_SOURCES.values()}
        record["demand"] = 0
        return record

    all_records = {}
    chunks = list(date_range_chunks(start_date, end_date))

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        url = BALANCE_URL.format(start=chunk_start, end=chunk_end)
        print(f"  Requesting: {chunk_start[:10]} to {chunk_end[:10]}")

        try:
            data = fetch_with_retry(url)
        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch chunk {chunk_start} - {chunk_end}: {e}")
            time.sleep(1)
            continue

        if "included" not in data:
            print(f"  Warning: No 'included' data in response for {chunk_start}")
            continue

        # Parse nested structure
        for category in data.get("included", []):
            attributes = category.get("attributes", {})
            content_items = attributes.get("content", [])

            for item in content_items:
                item_type = item.get("type", "")
                item_id = str(item.get("id", ""))
                item_attrs = item.get("attributes", {})
                values = item_attrs.get("values", [])

                field_name = None
                if item_id in ALL_SOURCES:
                    field_name = ALL_SOURCES[item_id]
                elif item_type == "Demanda en b.c." or item_id == "Demanda en b.c.":
                    field_name = "demand"

                if field_name is None:
                    continue

                for val in values:
                    if "datetime" not in val or "value" not in val:
                        continue

                    dt = pd.to_datetime(val["datetime"], utc=True).date()
                    date_str = str(dt)

                    if date_str not in all_records:
                        all_records[date_str] = empty_record()

                    value = val.get("value", 0) or 0
                    all_records[date_str][field_name] = value

        # Rate limit
        if i < len(chunks) - 1:
            time.sleep(0.5)

    if not all_records:
        raise ValueError("No generation data retrieved from API")

    # Convert to DataFrame and calculate aggregates
    records = []
    for date_str, vals in all_records.items():
        demand = vals["demand"]
        if demand <= 0:
            continue

        renewable_total = sum(vals[name] for name in RENEWABLE_SOURCES.values())
        non_renewable_total = sum(vals[name] for name in NON_RENEWABLE_SOURCES.values())

        renewable_share = (renewable_total / demand) * 100
        non_renewable_share = (non_renewable_total / demand) * 100

        record = {
            "date": date_str,
            "renewable_share": renewable_share,
            "non_renewable_share": non_renewable_share,
            # Renewable sources (MWh)
            "hydro_mwh": vals["hydro"],
            "wind_mwh": vals["wind"],
            "solar_pv_mwh": vals["solar_pv"],
            "solar_thermal_mwh": vals["solar_thermal"],
            "hydro_wind_mwh": vals["hydro_wind"],
            "other_renewable_mwh": vals["other_renewable"],
            "renewable_waste_mwh": vals["renewable_waste"],
            # Non-renewable sources (MWh)
            "nuclear_mwh": vals["nuclear"],
            "combined_cycle_mwh": vals["combined_cycle"],
            "coal_mwh": vals["coal"],
            "diesel_mwh": vals["diesel"],
            "gas_turbine_mwh": vals["gas_turbine"],
            "steam_turbine_mwh": vals["steam_turbine"],
            "cogeneration_mwh": vals["cogeneration"],
            "non_renew_waste_mwh": vals["non_renew_waste"],
            # Storage (MWh)
            "pumped_hydro_gen_mwh": vals["pumped_hydro_gen"],
            "pumped_hydro_cons_mwh": vals["pumped_hydro_cons"],
            "battery_discharge_mwh": vals["battery_discharge"],
            "battery_charge_mwh": vals["battery_charge"],
            # Totals
            "renewable_total_mwh": renewable_total,
            "non_renewable_total_mwh": non_renewable_total,
            "demand_mwh": demand,
        }
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    print(f"  Retrieved {len(df)} daily generation records")
    return df


def merge_datasets(prices_df: pd.DataFrame, generation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price and generation data on date.

    Args:
        prices_df: Daily prices DataFrame
        generation_df: Daily generation DataFrame

    Returns:
        Merged DataFrame with all columns from both datasets
    """
    print("Merging datasets...")

    prices_df = prices_df.copy()
    generation_df = generation_df.copy()

    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.normalize()
    generation_df["date"] = pd.to_datetime(generation_df["date"]).dt.normalize()

    merged = pd.merge(prices_df, generation_df, on="date", how="inner")

    # Reorder columns: date, price, shares first
    priority_cols = ["date", "price_daily_avg", "renewable_share", "non_renewable_share"]
    other_cols = [c for c in merged.columns if c not in priority_cols]
    result = merged[priority_cols + other_cols].copy()

    result = result.sort_values("date").reset_index(drop=True)

    print(f"  Merged dataset: {len(result)} records")
    return result


def print_seasonal_stats(df: pd.DataFrame) -> None:
    """Print seasonal statistics for renewable share, non-renewable share, and price."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(get_season)

    print_header("Seasonal Statistics")

    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    print(f"{'Season':<10} {'Renewable%':>12} {'Non-Renew%':>12} {'Price EUR':>12} {'Days':>8}")
    print("-" * 56)

    for season in seasons:
        season_data = df[df["season"] == season]
        if len(season_data) == 0:
            continue

        renew_avg = season_data["renewable_share"].mean()
        non_renew_avg = season_data["non_renewable_share"].mean()
        price_avg = season_data["price_daily_avg"].mean()
        count = len(season_data)

        print(f"{season:<10} {renew_avg:>11.1f}% {non_renew_avg:>11.1f}% {price_avg:>11.2f} {count:>8}")

    print("-" * 56)
    renew_avg = df["renewable_share"].mean()
    non_renew_avg = df["non_renewable_share"].mean()
    price_avg = df["price_daily_avg"].mean()
    print(f"{'Annual':<10} {renew_avg:>11.1f}% {non_renew_avg:>11.1f}% {price_avg:>11.2f} {len(df):>8}")


def save_data(df: pd.DataFrame, output_dir: str = "/data") -> str:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        output_dir: Output directory path

    Returns:
        Path to saved file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, "spain_renewables_prices.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved data to {filepath}")
    return filepath


def load_data(data_dir: str = "/data") -> pd.DataFrame:
    """
    Load previously fetched data from CSV.

    Args:
        data_dir: Directory containing the CSV file

    Returns:
        DataFrame with date parsed
    """
    filepath = os.path.join(data_dir, "spain_renewables_prices.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}. Run 'fetch' command first.")

    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def fetch_and_save(start_date: str, end_date: str, output_dir: str = "/data") -> pd.DataFrame:
    """
    Complete data fetching pipeline: fetch prices, fetch generation, merge, and save.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV

    Returns:
        Merged DataFrame
    """
    # Fetch spot prices and aggregate to daily
    spot_prices = fetch_spot_prices(start_date, end_date)
    daily_prices = aggregate_prices_to_daily(spot_prices)

    print()

    # Fetch daily generation balance
    generation = fetch_generation_balance(start_date, end_date)

    print()

    # Merge datasets
    merged = merge_datasets(daily_prices, generation)

    # Print summary
    print_header("Sample Data (first 5 rows, key columns)")
    sample_cols = ["date", "price_daily_avg", "renewable_share", "non_renewable_share",
                   "wind_mwh", "solar_pv_mwh", "hydro_mwh", "nuclear_mwh"]
    sample_cols = [c for c in sample_cols if c in merged.columns]
    print(merged[sample_cols].head(5).to_string(index=False))

    print_header("Overall Statistics")
    print(f"Total records: {len(merged)}")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"Columns: {len(merged.columns)} ({len([c for c in merged.columns if 'mwh' in c.lower()])} generation sources)")
    print()
    print(f"Renewable share:     {merged['renewable_share'].mean():>6.1f}% avg "
          f"(min: {merged['renewable_share'].min():.1f}%, max: {merged['renewable_share'].max():.1f}%)")
    print(f"Non-renewable share: {merged['non_renewable_share'].mean():>6.1f}% avg "
          f"(min: {merged['non_renewable_share'].min():.1f}%, max: {merged['non_renewable_share'].max():.1f}%)")
    print(f"Daily price:         {merged['price_daily_avg'].mean():>6.2f} EUR/MWh avg "
          f"(min: {merged['price_daily_avg'].min():.2f}, max: {merged['price_daily_avg'].max():.2f})")

    # Seasonal statistics
    print_seasonal_stats(merged)

    # Save to CSV
    print()
    save_data(merged, output_dir)

    return merged
