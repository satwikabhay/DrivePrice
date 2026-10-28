"""Standardize the four source CSVs into one auditable car-listings table."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "used_cars_combined.csv"
SOURCES = {
    "car_data_301.csv": "CarDekho 301-row sample",
    "car_details.csv": "CarDekho details sample",
    "car_details_v3.csv": "CarDekho details v3",
    "car_details_v4.csv": "CarDekho details v4",
}
BRANDS = [
    "Mercedes-Benz", "Land Rover", "Maruti Suzuki", "Force Motors", "Volkswagen",
    "Mitsubishi", "Chevrolet", "Mahindra", "Hyundai", "Renault", "Datsun",
    "Toyota", "Nissan", "Skoda", "Honda", "Tata", "BMW", "Audi", "Volvo",
    "Jeep", "Kia", "Ford", "Fiat", "Isuzu", "Porsche", "Jaguar", "Lexus",
    "MG", "Mini", "Bajaj", "Ashok Leyland", "Opel", "Daewoo", "Maruti",
]
BRANDS = sorted(BRANDS, key=len, reverse=True)
TWO_WHEELER = re.compile(r"^(?:activa|bajaj\b|hero\b|honda (?:activa|cb\b|cbr\b|dream\b|karizma\b)|hyosung\b|ktm\b|mahindra mojo\b|royal enfield\b|suzuki access\b|tvs\b|um renegade\b|yamaha\b)", re.I)


def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def title_category(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip())
    special = {"bmw": "BMW", "mg": "MG", "cng": "CNG", "lpg": "LPG", "ev": "EV"}
    text = " ".join(special.get(w.lower(), w.capitalize()) for w in text.split())
    aliases = {"Maruti": "Maruti Suzuki", "Maruti Suzuki": "Maruti Suzuki", "Mercedes Benz": "Mercedes-Benz"}
    return aliases.get(text, text)


def extract_make(name):
    text = str(name).strip()
    for brand in BRANDS:
        if re.match(rf"^{re.escape(brand)}\b", text, flags=re.I):
            return title_category(brand)
    return np.nan


def extract_model_family(name, make):
    text = str(name).strip()
    if not text or text.lower() == "nan":
        return np.nan
    prefixes = ([str(make)] if pd.notna(make) else []) + BRANDS
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip(" -")
            break
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return " ".join(tokens[:2]).title() if tokens else np.nan


def canonicalize(filename, label):
    df = clean_columns(pd.read_csv(RAW / filename))
    out = pd.DataFrame(index=df.index)
    # Source-specific target and units: the 301-row file records lakh; others record rupees.
    name_col = "name" if "name" in df else "car_name"
    if "make" in df:
        out["make"] = df["make"].map(title_category)
        raw_name = (df["model"] if "model" in df else df[name_col]).astype("string").fillna("")
        out["name"] = (df["make"].astype("string").fillna("") + " " + raw_name).str.strip()
    else:
        out["name"] = df[name_col].astype("string").fillna("")
        out["make"] = out["name"].map(extract_make)
    out["model"] = [extract_model_family(n, m) for n, m in zip(out["name"], out["make"])]
    out["year"] = df["year"]
    out["selling_price_inr"] = pd.to_numeric(df["selling_price"] if "selling_price" in df else df["price"], errors="coerce")
    if filename == "car_data_301.csv":
        out["selling_price_inr"] *= 100_000
    kms_col = "kms_driven" if "kms_driven" in df else "kilometer" if "kilometer" in df else "km_driven"
    out["kms_driven"] = pd.to_numeric(df[kms_col], errors="coerce")
    out["fuel_type"] = df.get("fuel_type", df.get("fuel type", df.get("fuel"))).map(title_category)
    out["seller_type"] = df.get("seller_type", df.get("seller type")).map(title_category)
    out["transmission"] = df["transmission"].map(title_category)
    owner = df["owner"].astype("string").str.strip().str.lower()
    owner_map = {
        "0": "First Owner", "1": "Second Owner", "2": "Third Owner", "3": "Fourth Or More Owners",
        "first owner": "First Owner", "second owner": "Second Owner", "third owner": "Third Owner",
        "fourth & above owner": "Fourth Or More Owners", "fourth owner": "Fourth Or More Owners",
        "fourth & above owners": "Fourth Or More Owners", "test drive car": "Test Drive Vehicle",
        "first": "First Owner", "second": "Second Owner", "third": "Third Owner", "fourth": "Fourth Or More Owners", "4 or more": "Fourth Or More Owners", "unregistered car": "Unregistered Vehicle",
    }
    out["owner"] = owner.map(owner_map).fillna(owner.map(title_category))
    out["mileage_kmpl"] = df.get("mileage", pd.Series(index=df.index, dtype=object)).map(number)
    out["engine_cc"] = df.get("engine", pd.Series(index=df.index, dtype=object)).map(number)
    out["max_power_bhp"] = df.get("max_power", pd.Series(index=df.index, dtype=object)).map(number)
    seats_col = "seats" if "seats" in df else "seating_capacity"
    out["seats"] = df.get(seats_col, pd.Series(index=df.index, dtype=object)).map(number)
    out["source_dataset"] = label
    out["source_record_id"] = filename + ":" + df.index.astype(str)
    return out


def main():
    parts = [canonicalize(name, label) for name, label in SOURCES.items() if (RAW / name).exists()]
    if len(parts) != len(SOURCES):
        missing = [name for name in SOURCES if not (RAW / name).exists()]
        raise SystemExit("Missing source CSVs in data/raw: " + ", ".join(missing))
    df = pd.concat(parts, ignore_index=True)
    input_rows = len(df)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["selling_price_inr"] = pd.to_numeric(df["selling_price_inr"], errors="coerce")
    df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")
    before_validation = len(df)
    df = df.dropna(subset=["year", "selling_price_inr", "kms_driven", "fuel_type", "transmission"]).copy()
    df = df[(df["selling_price_inr"] > 0) & (df["kms_driven"] >= 0) & df["year"].between(1980, 2025)].copy()
    invalid_removed = before_validation - len(df)
    bike_mask = df["name"].str.match(TWO_WHEELER, na=False)
    removed_bikes = int(bike_mask.sum())
    df = df.loc[~bike_mask].copy()
    after_bikes = len(df)
    # If the same listing occurs in multiple snapshots, preserve the row with the most vehicle specs.
    key_cols = ["name", "year", "selling_price_inr", "kms_driven", "fuel_type", "transmission", "owner", "seller_type"]
    df["_spec_count"] = df[["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]].notna().sum(axis=1)
    df = df.sort_values("_spec_count", ascending=False).drop_duplicates(subset=key_cols, keep="first").drop(columns="_spec_count")
    duplicates_removed = after_bikes - len(df)
    df["selling_price_lakh"] = df["selling_price_inr"] / 100_000
    df = df.reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Source rows loaded: {input_rows:,}")
    print(f"Two-wheelers removed: {removed_bikes:,}")
    print(f"Duplicate listings removed: {duplicates_removed:,}")
    print(f"Missing/invalid essential records removed: {invalid_removed:,}")
    print(f"Combined car listings retained after cross-source deduplication: {len(df):,}")
    print(f"Recorded year range: {int(df.year.min())}–{int(df.year.max())}")
    print("Rows by source after cleaning:\n" + df.source_dataset.value_counts().to_string())
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()









