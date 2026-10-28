# Used Car Price Prediction

A solo Introduction to Data Science project that explores Indian used-car listings, prepares a reproducible dataset, compares regression models, and provides an interactive Streamlit estimator.

## Project question

How accurately can a used car’s recorded model year, make, mileage, ownership and technical specifications estimate its resale asking price?

## Data

The project combines four CSV tables from Kaggle’s [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho): the 301-row `car data.csv`, `CAR DETAILS FROM CAR DEKHO.csv`, `Car details v3.csv`, and `car details v4.csv`. Source files are retained in `data/raw/`; the original project file `data/car_data.csv` remains unchanged. See [data/README.md](data/README.md) for provenance and preparation decisions.

The preparation script standardizes prices to rupees, removes duplicate listings and recognizable two-wheelers, and saves `data/used_cars_combined.csv`. The current combined table contains 12,705 listings after cleaning and covers model years 1983–2022. Not every source has all specifications, so missing technical fields are imputed inside the model pipeline. `Present_Price` is excluded because it is not consistently available across the larger sources.

The data is historical, and its newest listing year is 2022. The app permits input through 2025 for exploration, but it warns that predictions after 2022 are outside the observed range. Adding rows alone does not guarantee improved accuracy; evaluation results and limitations must be reported.

## Set up

Use Python 3.12 or newer. The included dependency versions are pinned to the versions used to build and review this app. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Prepare and analyze

Run the data preparation step, then the analysis:

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\analyze.py
```

The analysis trains through model year 2019 and evaluates on later listings from 2020–2022. It writes metrics and plots to `outputs/expanded_dataset/`. This time-based split is more demanding than a random split and better reflects prediction on newer model years.

Current held-out results (prices in ₹ lakh):

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random forest | 7.11 | 16.13 | 0.739 |
| Linear regression | 9.43 | 20.66 | 0.573 |
| Median baseline | 22.71 | 38.86 | -0.512 |

The newer-year test contains 476 listings, so these scores have uncertainty. The random forest beats the simple baseline, but its errors are still material, especially for high-priced cars. Do not describe the model as market-accurate or use it for a real purchase decision.

## Run the interactive demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app lets users choose a manufacturer/model family, model year, mileage, fuel, transmission, seller and owner history, with optional technical specifications. It includes interactive filtered plots and a temporal model comparison. Its design uses the `.streamlit/config.toml` theme and responsive layout styles.

## Deploy

The app is structured for Streamlit Community Cloud: `app.py` and `requirements.txt` are at the project root, `.streamlit/config.toml` provides the theme, and the combined CSV plus score file are included in the project. Push the project to a GitHub repository, then choose that repository, branch and `app.py` as the entrypoint in Streamlit Community Cloud. Keep `data/raw/` with the repository if you want to reproduce preparation; the app itself reads the combined CSV.

## Limitations

- Historical asking prices do not necessarily equal final transaction prices.
- The four source tables have different fields, collection snapshots and possible source overlap; normalization and deduplication reduce but cannot eliminate those differences.
- Condition, accident and service history, exact trim and listing location/date are not consistently present.
- The model is trained on data only through 2022, so later model years are out-of-range estimates.
- Performance varies by price tier; MAE and RMSE are in ₹ lakh and should be interpreted alongside plots and the baseline.
- Associations in this observational data do not establish causation.



