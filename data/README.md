# Dataset provenance

## Source

Downloaded on 27 October 2026 from Kaggle’s [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), published by `nehalbirla`. The Kaggle dataset contains multiple snapshots of Indian used-car listings. The dataset page identifies the reuse license; the Kaggle dataset has also been catalogued with CC0-1.0 rights in [research-data metadata](https://test.researchdata.tuwien.ac.at/records/k19vc-wa972/files/DMP_Predictive_Modeling_of_Used_Car_Prices_Based_on_Market_and_Technical_Attributes.pdf?download=1). Check the source page before redistribution, and cite the dataset in project submissions.

## Included CSVs

| File in `raw/` | Records before cleaning | Fields used |
|---|---:|---|
| `car_data_301.csv` | 301 | Name, year, selling price, kilometres, fuel, seller, transmission, owners |
| `car_details.csv` | 4,340 | Name, year, selling price, kilometres, fuel, seller, transmission, owners |
| `car_details_v3.csv` | 8,128 | Core listing fields plus mileage, engine, power, seats |
| `car_details_v4.csv` | 2,059 | Make, model, price, year, kilometres, fuel, seller, transmission, owner, engine, power, seats |

`data/car_data.csv` is the original 301-row file already used in the first project iteration; it is not overwritten. A copy of that source is also retained in `raw/` so the combined pipeline is reproducible from one folder.

## Standardization decisions

- Convert the 301-row sample’s selling price from ₹ lakh to rupees; other source files record rupees.
- Convert kilometre and specification strings to numeric values.
- Map owner-count labels to consistent categories and normalize make/model/fuel labels.
- Keep model year, make, a two-token model family, kilometres, fuel, seller, transmission, owner, mileage, engine, power and seats.
- Exclude `Present_Price`, location, color and other fields that do not exist consistently across the source tables.
- Remove rows missing essential target/year/kilometre/fuel/transmission values, invalid prices or distances, and recognizable motorcycles/scooters.
- Deduplicate using normalized name, year, price, kilometres, fuel, transmission, owner and seller fields. Where a duplicate has more technical specifications, keep that more complete record.
- Preserve source table and source-row identifiers for traceability. Do not use those identifiers as model features.

Run `python src/prepare_data.py` to produce `data/used_cars_combined.csv` and print the cleaning counts. The inputs are retained unchanged.
