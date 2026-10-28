# Solo Guide: Used Car Price Prediction

Your project folder is:

`C:\Users\SATWIK\OneDrive - Manipal Academy of Higher Education\Desktop\7sem\Introduction_to_Datascience\ProjectFolder`

The folders and scripts are already in place. You do not need to create them manually.

## 1. Open the project in VS Code

Open the folder above in VS Code and choose **Terminal → New Terminal**. Confirm the terminal is at the project root:

```powershell
Get-Location
```

If needed, move there with:

```powershell
Set-Location "C:\Users\SATWIK\OneDrive - Manipal Academy of Higher Education\Desktop\7sem\Introduction_to_Datascience\ProjectFolder"
```

## 2. Set up the Python environment

Run once:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You can use `.\.venv\Scripts\python.exe` for later commands without activating the environment.

## 3. Review the source data

Four CSV files from Kaggle’s Vehicle Dataset from CarDekho are preserved in `data\raw`. The original 301-row project file remains at `data\car_data.csv`. Read `data\README.md` for the dataset sources, license notes and cleaning decisions.

Do not edit the original source CSVs. The preparation script creates a separate standardized file.

## 4. Prepare the combined dataset

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py
```

This combines the four source tables, converts all prices to rupees, standardizes shared fields, removes two-wheelers and duplicate listings, and creates `data\used_cars_combined.csv`. The current run loads 14,828 source rows and retains 12,705 listings after removing 101 two-wheelers and 2,022 duplicate listings. The data covers model years 1983–2022.

Some sources do not contain mileage, engine, power or seat information. Those values remain missing in the combined file and are imputed by the modeling pipeline.

## 5. Run the analysis

```powershell
.\.venv\Scripts\python.exe src\analyze.py
```

The analysis trains on model years through 2019 and evaluates on later cars from 2020–2022. This time-based test contains 476 listings and is a tougher, more future-like evaluation than a random split. Outputs are in `outputs\expanded_dataset`.

Current results, in ₹ lakh:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random forest | 7.11 | 16.13 | 0.739 |
| Linear regression | 9.43 | 20.66 | 0.573 |
| Median baseline | 22.71 | 38.86 | -0.512 |

Random Forest improves on the baseline, but its typical error is still substantial. Discuss the error size and the high-priced listings in the test set; do not describe this as a dependable live-market valuation.

## 6. Interpret the charts and scores

Open the plots in `outputs\expanded_dataset`:

- `price_distribution.png`: how listing prices are distributed
- `price_by_year.png`: prices versus model year
- `price_by_kilometres.png`: prices versus kilometres driven
- `price_by_fuel_type.png`: price spread across fuel types
- `actual_vs_predicted.png`: how close Random Forest predictions are to held-out prices

For MAE, lower is better; it is the average absolute error in lakh rupees. RMSE penalizes larger misses more. R² compares the model with a mean-price baseline and can be negative. Describe patterns as associations, not causes.

## 7. Run the optional Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The interface accepts model years up to 2025, but the dataset ends in 2022. It warns when an input is newer than the training data. Predictions for those years are outside the observed range and may be unreliable. Press **Ctrl+C** in the terminal to stop the local app.

## 8. Prepare the final report

Include the project question, data sources and license, row counts and cleaning steps, exploratory plots, the temporal split, baseline/model metrics, a plain-language conclusion and limitations. Cite the Kaggle dataset and state that its newest model year is 2022. Mention the held-out MAE in rupees so readers understand the error scale.

## 9. Deployment

The app is organized for Streamlit Community Cloud. Before deployment, push the project root to GitHub with `app.py`, `requirements.txt`, `.streamlit\config.toml`, `data\used_cars_combined.csv` and `outputs\expanded_dataset\model_metrics.csv` included. Then select that GitHub repository, branch and `app.py` entrypoint in Streamlit Community Cloud. Keep the raw data too if you want the deployment repository to be reproducible.


