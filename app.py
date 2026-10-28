"""Professional interactive demo for the expanded Indian used-car price project."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="AutoValue | Used-car estimator", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "used_cars_combined.csv"
METRICS_PATH = ROOT / "outputs" / "expanded_dataset" / "model_metrics.csv"
FEATURES = ["make", "model", "year", "kms_driven", "fuel_type", "seller_type", "transmission", "owner", "mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]
LATEST_INPUT_YEAR = 2025

st.markdown("""
<style>
:root { --ink:#142c3e; --muted:#62778a; --teal:#087f78; --teal-dark:#075e59; --line:#dce5eb; --surface:#ffffff; --canvas:#f3f6f9; }
.stApp { background:var(--canvas); color:var(--ink); }
.block-container { max-width:1320px; padding:2.2rem clamp(1rem,3vw,2.5rem) 3rem; }
[data-testid="stSidebar"] { background:#eaf0f4; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--ink); }
.hero { position:relative; overflow:hidden; padding:2rem clamp(1.3rem,4vw,3rem); margin:0 0 1.5rem; border-radius:22px; color:white; background:linear-gradient(118deg,#102b42 0%,#155b67 66%,#087f78 100%); box-shadow:0 14px 36px rgba(16,43,66,.15); }
.hero:after { content:""; position:absolute; width:270px; height:270px; right:-75px; top:-130px; border:1px solid rgba(255,255,255,.18); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025); }
.hero-kicker { color:#bce8df; letter-spacing:.16em; text-transform:uppercase; font-size:.76rem; font-weight:700; margin:0 0 .55rem; }
.hero h1 { color:#fff; margin:0; font-size:clamp(2rem,4vw,3.15rem); letter-spacing:-.035em; line-height:1.08; }
.hero p { color:#e0f0f1; max-width:720px; margin:.7rem 0 0; font-size:1.05rem; }
.section-title { color:var(--ink); font-size:1.25rem; font-weight:700; margin:.2rem 0 .2rem; }
.section-subtitle { color:var(--muted); margin:0 0 1.1rem; font-size:.94rem; }
[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:1rem 1.1rem; box-shadow:0 4px 16px rgba(16,43,66,.035); }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--ink); }
div[data-testid="stForm"] { padding:1.25rem clamp(.8rem,2vw,1.5rem); border:1px solid var(--line); border-radius:18px; background:var(--surface); box-shadow:0 8px 24px rgba(16,43,66,.045); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
div.stButton > button, div.stFormSubmitButton > button { border-radius:10px; min-height:2.9rem; font-weight:650; transition:transform .18s ease, box-shadow .18s ease, background-color .18s ease; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 18px rgba(8,127,120,.18); }
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { border-radius:9px; }
[data-testid="stTabs"] button { font-weight:650; }
.result-panel { padding:1.3rem 1.5rem; background:linear-gradient(120deg,#e8f5f2,#f9fcfb); border:1px solid #b7ded5; border-radius:18px; margin-top:1rem; }
.result-label { color:var(--muted); text-transform:uppercase; font-size:.78rem; letter-spacing:.12em; font-weight:700; }
.result-price { color:var(--teal-dark); font-size:clamp(2.2rem,5vw,3.6rem); font-weight:760; letter-spacing:-.04em; line-height:1.15; margin:.25rem 0; }
.info-panel { padding:1rem 1.2rem; border-left:4px solid var(--teal); background:#eaf4f5; border-radius:0 12px 12px 0; color:var(--ink); }
.small-note { color:var(--muted); font-size:.85rem; }
@media (max-width:760px) { .block-container {padding-top:1.1rem;} .hero {border-radius:16px;padding:1.35rem;} [data-testid="stMetric"] {padding:.8rem;} }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition:none !important; animation:none !important; scroll-behavior:auto !important; } }
</style>
""", unsafe_allow_html=True)


def make_preprocessor(df):
    numeric = [name for name in FEATURES if name in df.select_dtypes(include=np.number).columns]
    categorical = [name for name in FEATURES if name not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical),
    ])


@st.cache_data(show_spinner="Preparing the verified car listings…")
def load_data(path, modified_time):
    del modified_time
    df = pd.read_csv(path)
    required = set(FEATURES + ["selling_price_lakh", "source_dataset"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Combined dataset is missing columns: " + ", ".join(sorted(missing)))
    return df.dropna(subset=["selling_price_lakh", "year"]).copy()


@st.cache_resource(show_spinner="Loading the estimator. The first run may take a short while…")
def train_models(path, modified_time):
    df = load_data(path, modified_time)
    X, y = df[FEATURES], df["selling_price_lakh"]
    preprocessor = make_preprocessor(df)
    estimators = {
        "Random forest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, max_features=.8, max_samples=.8, random_state=42, n_jobs=-1),
        "Linear regression": LinearRegression(),
    }
    fitted = {}
    for name, estimator in estimators.items():
        model = Pipeline([("prepare", preprocessor), ("estimator", estimator)])
        model.fit(X, y)
        fitted[name] = model
    return df, fitted


def read_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return pd.read_csv(METRICS_PATH).set_index("model")
    except (OSError, ValueError, KeyError):
        return None


st.markdown("""
<div class="hero">
  <div class="hero-kicker">India · Used-car market</div>
  <h1>AutoValue</h1>
  <p>A clear, data-led estimate of a used car’s resale value, built from more than twelve thousand cleaned listings.</p>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("The combined dataset is missing. Run `python src/prepare_data.py` from the project folder first.")
    st.stop()

try:
    cars, models = train_models(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"Could not load the estimator: {exc}")
    st.stop()

metrics = read_metrics()
latest_training_year = int(cars["year"].max())
early_year = int(cars["year"].min())

headline = st.columns(3)
headline[0].metric("Clean listings", f"{len(cars):,}")
headline[1].metric("Model years in data", f"{early_year}–{latest_training_year}")
headline[2].metric("Median listed price", f"₹{cars['selling_price_lakh'].median():.2f} lakh")
st.markdown("<div class='small-note'>Four public CarDekho tables were standardized, duplicate listings removed, and two-wheelers excluded. The source files remain available in <code>data/raw</code>.</div>", unsafe_allow_html=True)

estimate_tab, explore_tab, performance_tab = st.tabs(["Price estimate", "Explore the data", "Model performance"])

with st.sidebar:
    st.markdown("### Estimator settings")
    model_name = st.radio("Prediction model", ["Random forest", "Linear regression"], horizontal=False)
    st.caption("The model is trained on all cleaned listings. Evaluation scores use separate, newer model years.")
    if metrics is not None and model_name in metrics.index:
        score = metrics.loc[model_name]
        st.metric("Held-out MAE", f"₹{score['MAE']:.2f} lakh")
        st.metric("Held-out R²", f"{score['R2']:.3f}")
        st.caption(f"Tested on {int(score['test_rows']):,} listings from {score['test_years']} model years.")
    st.divider()
    st.markdown("**Price units**  ")
    st.caption("All predictions and model errors are shown in Indian rupees (lakh).")

with estimate_tab:
    st.markdown('<div class="section-title">Describe the car</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Use the model year, not the year you purchased the car. Add technical details if you know them.</div>', unsafe_allow_html=True)
    identity_left, identity_right = st.columns(2, gap="large")
    make_counts = cars["make"].dropna().astype(str).value_counts()
    makes = sorted(make_counts.index.tolist())
    default_make = make_counts.index[0]
    selected_make = identity_left.selectbox("Manufacturer", makes, index=makes.index(default_make))
    model_counts = cars.loc[cars["make"].astype(str) == selected_make, "model"].dropna().astype(str).value_counts()
    model_options = sorted(model_counts.index.tolist())
    model_choices = ["Not specified"] + model_options
    default_model = model_counts.index[0] if not model_counts.empty else "Not specified"
    selected_model = identity_right.selectbox("Model family", model_choices, index=model_choices.index(default_model))

    reference = cars[cars["make"].astype(str) == selected_make]
    if selected_model != "Not specified":
        model_reference = reference[reference["model"].astype(str) == selected_model]
        if not model_reference.empty:
            reference = model_reference

    def typical_value(column):
        subset = reference[column].dropna()
        if subset.empty:
            subset = cars[column].dropna()
        return float(subset.median())

    def common_index(options, column):
        modes = reference[column].dropna().mode()
        choice = modes.iloc[0] if not modes.empty and modes.iloc[0] in options else options[0]
        return options.index(choice)

    spec_toggle = st.checkbox("I know the car’s technical specifications", value=False)
    with st.form("estimator_form"):
        usage_left, usage_right = st.columns(2, gap="large")
        default_year = int(np.clip(reference["year"].median(), early_year, latest_training_year))
        year = usage_left.number_input("Model year", min_value=early_year, max_value=LATEST_INPUT_YEAR, value=default_year, step=1, help=f"Training listings cover {early_year}–{latest_training_year}. Later years are accepted for exploration, but lie beyond the evidence available to the model.")
        kms = usage_right.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=int(typical_value("kms_driven")), step=1000)
        mileage = engine = power = seats = np.nan
        if spec_toggle:
            spec_cols = st.columns(4, gap="medium")
            mileage = spec_cols[0].number_input("Mileage (km/l)", min_value=0.0, max_value=100.0, value=typical_value("mileage_kmpl"), step=.1)
            engine = spec_cols[1].number_input("Engine (cc)", min_value=0.0, max_value=10000.0, value=typical_value("engine_cc"), step=50.0)
            power = spec_cols[2].number_input("Power (bhp)", min_value=0.0, max_value=2000.0, value=typical_value("max_power_bhp"), step=5.0)
            seats = spec_cols[3].number_input("Seats", min_value=1.0, max_value=20.0, value=typical_value("seats"), step=1.0)
        category_cols = st.columns(4, gap="medium")
        fuel_options = sorted(cars["fuel_type"].dropna().unique().tolist())
        seller_options = sorted(cars["seller_type"].dropna().unique().tolist())
        transmission_options = sorted(cars["transmission"].dropna().unique().tolist())
        owner_order = ["First Owner", "Second Owner", "Third Owner", "Fourth Or More Owners", "Test Drive Vehicle", "Unregistered Vehicle"]
        owner_values = [value for value in owner_order if value in cars["owner"].unique()]
        fuel = category_cols[0].selectbox("Fuel type", fuel_options, index=common_index(fuel_options, "fuel_type"))
        transmission = category_cols[1].selectbox("Transmission", transmission_options, index=common_index(transmission_options, "transmission"))
        owner = category_cols[2].selectbox("Ownership history", owner_values, index=common_index(owner_values, "owner"))
        seller = category_cols[3].selectbox("Seller type", seller_options, index=common_index(seller_options, "seller_type"))
        submitted = st.form_submit_button("Estimate resale price", type="primary", width="stretch")
    if submitted:
        model_value = np.nan if selected_model == "Not specified" else selected_model
        input_row = pd.DataFrame([{
            "make": selected_make, "model": model_value, "year": int(year), "kms_driven": int(kms),
            "fuel_type": fuel, "seller_type": seller, "transmission": transmission, "owner": owner,
            "mileage_kmpl": mileage, "engine_cc": engine, "max_power_bhp": power, "seats": seats,
        }], columns=FEATURES)
        estimate = float(models[model_name].predict(input_row)[0])
        if year > latest_training_year:
            st.warning(f"The selected model year ({int(year)}) is newer than the latest training listing ({latest_training_year}). This prediction is outside the observed range and may be unreliable.")
        if estimate <= 0:
            st.warning("This combination produced a non-positive estimate. Please review the details or try the other model.")
        else:
            st.markdown(f"""
            <div class="result-panel">
              <div class="result-label">Estimated resale value · {model_name}</div>
              <div class="result-price">₹{estimate:,.2f} lakh</div>
              <div>Approximately ₹{estimate * 100000:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            if metrics is not None and model_name in metrics.index:
                st.caption(f"For context, this model’s average absolute error on the newer-year holdout was ₹{metrics.loc[model_name, 'MAE']:.2f} lakh. This is not a guaranteed error range for this particular car.")
        with st.expander("How this estimate should be used"):
            st.write("This is a classroom model trained on historical asking prices. It does not observe vehicle condition, accident or service history, exact trim, negotiation, or live market changes. Prices may differ across cities and listing dates.")

with explore_tab:
    st.markdown('<div class="section-title">Explore the cleaned listings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Filter the data and inspect how asking prices vary across model years, mileage and fuel categories.</div>', unsafe_allow_html=True)
    filter_cols = st.columns([1.2, 1, 1])
    selected_makes = filter_cols[0].multiselect("Manufacturer filter", sorted(cars["make"].dropna().unique()), default=[])
    year_range = filter_cols[1].slider("Model year range", early_year, latest_training_year, (early_year, latest_training_year))
    color_by = filter_cols[2].selectbox("Color points by", ["fuel_type", "transmission", "seller_type"])
    filtered = cars[cars["year"].between(year_range[0], year_range[1])].copy()
    if selected_makes:
        filtered = filtered[filtered["make"].isin(selected_makes)]
    chart_cols = st.columns([1.15, 1])
    x_axis = chart_cols[0].selectbox("Horizontal axis", ["year", "kms_driven", "engine_cc", "max_power_bhp"], format_func=lambda col: {"year":"Model year", "kms_driven":"Kilometres driven", "engine_cc":"Engine size (cc)", "max_power_bhp":"Power (bhp)"}[col])
    chart = px.scatter(filtered, x=x_axis, y="selling_price_lakh", color=color_by, hover_data=["make", "model", "year", "kms_driven", "fuel_type", "owner"], opacity=.58, labels={"selling_price_lakh":"Resale price (₹ lakh)", "year":"Model year", "kms_driven":"Kilometres driven", "engine_cc":"Engine (cc)", "max_power_bhp":"Power (bhp)"}, color_discrete_sequence=["#087f78", "#e08b57", "#477aa8", "#9a6bb0", "#74a18b"])
    chart.update_layout(template="plotly_white", margin=dict(l=12,r=12,t=24,b=12), legend_title_text=color_by.replace("_", " ").title(), font=dict(family="Arial",color="#142c3e"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    chart_cols[0].plotly_chart(chart, width="stretch", config={"displayModeBar":False})
    distribution = px.histogram(filtered, x="selling_price_lakh", color=color_by, nbins=36, opacity=.78, labels={"selling_price_lakh":"Resale price (₹ lakh)"}, color_discrete_sequence=["#087f78", "#e08b57", "#477aa8", "#9a6bb0", "#74a18b"])
    distribution.update_layout(template="plotly_white", barmode="overlay", margin=dict(l=12,r=12,t=24,b=12), showlegend=False, font=dict(family="Arial",color="#142c3e"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    chart_cols[1].plotly_chart(distribution, width="stretch", config={"displayModeBar":False})
    st.caption(f"Showing {len(filtered):,} of {len(cars):,} cleaned listings. These are historical listings, not current sale prices.")

with performance_tab:
    st.markdown('<div class="section-title">Evaluation on later model years</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Models train on listings through 2019 and are evaluated on the later 2020–2022 listings, giving a more realistic check for newer cars.</div>', unsafe_allow_html=True)
    if metrics is not None:
        display_metrics = metrics.reset_index()[["model", "MAE", "RMSE", "R2", "train_rows", "test_rows", "test_years"]].copy()
        display_metrics = display_metrics.rename(columns={"model":"Model", "MAE":"MAE (₹ lakh)", "RMSE":"RMSE (₹ lakh)", "R2":"R²", "train_rows":"Training listings", "test_rows":"Test listings", "test_years":"Test years"})
        st.dataframe(display_metrics, hide_index=True, width="stretch", column_config={"MAE (₹ lakh)":st.column_config.NumberColumn(format="%.2f"), "RMSE (₹ lakh)":st.column_config.NumberColumn(format="%.2f"), "R²":st.column_config.NumberColumn(format="%.3f")})
        mae_fig = px.bar(display_metrics, x="Model", y="MAE (₹ lakh)", color="Model", color_discrete_map={"Random forest":"#087f78", "Linear regression":"#477aa8", "Median baseline":"#a6b7c5"}, text_auto=".2f")
        mae_fig.update_layout(template="plotly_white", showlegend=False, margin=dict(l=12,r=12,t=22,b=12), font=dict(family="Arial",color="#142c3e"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(mae_fig, width="stretch", config={"displayModeBar":False})
        st.info("MAE is the average absolute error; RMSE penalizes large misses more; R² compares the model with a constant-mean baseline. The temporal test includes only 476 listings, so scores are informative but still uncertain.")
    else:
        st.warning("Evaluation metrics are not available yet. Run `python src/analyze.py` from the project folder.")

with st.expander("Data sources and methodology"):
    st.write("The combined training table standardizes four CarDekho dataset files. Prices in the 301-row file were converted from lakh rupees to rupees before standardization. Exact and matching cross-file duplicates were removed, recognizable two-wheelers were excluded, and missing vehicle specifications are imputed within the modeling pipeline. Source CSVs are retained in `data/raw`; the original 301-row file remains unchanged at `data/car_data.csv`.")




