"""Profile the combined India used-car data and compare regression models."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "used_cars_combined.csv"
OUT = ROOT / "outputs" / "expanded_dataset"
FEATURES = ["make", "model", "year", "kms_driven", "fuel_type", "seller_type", "transmission", "owner", "mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]


def main():
    if not DATA.exists():
        raise SystemExit("Combined dataset not found. Run: python src/prepare_data.py")
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["selling_price_lakh", "year"]).copy()
    OUT.mkdir(parents=True, exist_ok=True)
    cutoff = 2019
    train_mask = df["year"] <= cutoff
    test_mask = df["year"] > cutoff
    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, "selling_price_lakh"]
    X_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, "selling_price_lakh"]
    numeric = [col for col in FEATURES if col in df.select_dtypes(include=np.number).columns]
    categorical = [col for col in FEATURES if col not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical),
    ])
    models = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, max_features=0.8, max_samples=0.8, random_state=42, n_jobs=-1),
    }
    rows, predictions = [], {}
    for name, estimator in models.items():
        if name == "Median baseline":
            pipeline = estimator
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
        else:
            pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
        predictions[name] = pred
        rows.append({"model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R2": r2_score(y_test, pred), "train_rows": len(X_train), "test_rows": len(X_test), "test_years": f"{int(df.loc[test_mask, 'year'].min())}–{int(df.loc[test_mask, 'year'].max())}"})
    scores = pd.DataFrame(rows).sort_values("MAE")
    scores.to_csv(OUT / "model_metrics.csv", index=False)
    print(f"Combined cleaned listings: {len(df):,}")
    print(f"Features: {len(FEATURES)} | train years <= {cutoff}: {len(X_train):,} | temporal test years > {cutoff}: {len(X_test):,}")
    print(f"Year coverage: {int(df.year.min())}–{int(df.year.max())}")
    print("Missing feature values:\n" + df[FEATURES].isna().sum().to_string())
    print("\nTemporal held-out results (MAE/RMSE in INR lakh):\n" + scores.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    pred = predictions["Random forest"]
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, pred, alpha=.45, color="#0f766e", edgecolors="none")
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "--", color="#334155", linewidth=1.5)
    plt.xlabel("Actual resale price (INR lakh)"); plt.ylabel("Predicted resale price (INR lakh)")
    plt.title("Random forest: actual vs predicted (newer-year test set)"); plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["year"], df["selling_price_lakh"], alpha=.22, color="#0f766e", edgecolors="none")
    plt.xlabel("Model year"); plt.ylabel("Resale price (INR lakh)")
    plt.title("Resale price by model year"); plt.tight_layout()
    plt.savefig(OUT / "price_by_year.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(df["selling_price_lakh"], bins=40, color="#147d75", edgecolor="white")
    plt.xlabel("Resale price (INR lakh)"); plt.ylabel("Number of listings")
    plt.title("Distribution of resale prices"); plt.tight_layout()
    plt.savefig(OUT / "price_distribution.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["kms_driven"], df["selling_price_lakh"], alpha=.22, color="#d97757", edgecolors="none")
    plt.xlabel("Kilometres driven"); plt.ylabel("Resale price (INR lakh)")
    plt.title("Resale price by kilometres driven"); plt.tight_layout()
    plt.savefig(OUT / "price_by_kilometres.png", dpi=160); plt.close()

    fuel_groups = df.groupby("fuel_type", dropna=True)["selling_price_lakh"].apply(list)
    plt.figure(figsize=(8, 5))
    plt.boxplot(fuel_groups.tolist(), showfliers=False)
    plt.xticks(range(1, len(fuel_groups) + 1), [str(v) for v in fuel_groups.index])
    plt.xlabel("Fuel type"); plt.ylabel("Resale price (INR lakh)")
    plt.title("Resale price by fuel type"); plt.tight_layout()
    plt.savefig(OUT / "price_by_fuel_type.png", dpi=160); plt.close()

    source_counts = df.groupby("source_dataset").agg(listings=("selling_price_lakh", "size"), median_price_lakh=("selling_price_lakh", "median"), first_year=("year", "min"), last_year=("year", "max"))
    source_counts.to_csv(OUT / "source_summary.csv")
    print(f"Outputs saved to {OUT}")


if __name__ == "__main__":
    main()









