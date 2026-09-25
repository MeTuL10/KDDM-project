# HDB Monthly Rent Prediction

This repository contains work for the CS5228 HDB monthly-rent prediction task. My implementation trains a time-aware CatBoost regression ensemble to predict `MONTHLY_RENT` for 50,000 test transactions.

## My implementation

The entry point is [`train_rental_model.py`](train_rental_model.py). It trains two CatBoost models using random seeds 42 and 2026 and averages their predictions.

Key model settings:

| Parameter | Value |
|---|---:|
| Maximum iterations | 1,400 |
| Learning rate | 0.045 |
| Tree depth | 9 |
| L2 leaf regularisation | 8.0 |
| Random strength | 0.4 |
| Bagging temperature | 0.5 |
| Early-stopping rounds | 150 |
| Validation period | Latest 6 training months |

The submitted model achieved a **Kaggle leaderboard RMSE of 454.38880**.

## Feature pipeline

The script builds 34 base variables and 29 historical variables, giving the model 63 inputs. The feature groups include:

- original flat and transaction attributes;
- year, month, quarter, cyclic month, and flat age;
- property, property-and-flat-type, street-and-flat-type, town-and-flat-type, and model-and-flat-type categories;
- HDB block coordinates and administrative areas;
- distance to the nearest MRT station, school, and shopping mall;
- monthly COE indicators; and
- smoothed rent means and counts based on strictly earlier months.

Categorical variables are passed directly to CatBoost. The code does not manually one-hot encode them. Historical target features are constructed causally so a training row cannot use its own month or any future rent.

## External data

The training script reads the following provided files under `data/auxiliary/`:

- `sg-hdb-block.csv`
- `sg-mrt-stations.csv`
- `sg-schools.csv`
- `sg-shopping-malls.csv`
- `sg-coe-prices.csv`

No synthetic observations are generated.

## Installation

Python 3.10 or later is recommended. Install the required packages with:

```bash
python -m pip install catboost numpy pandas
```

## Running the model

Run validation and then train the final models:

```bash
python train_rental_model.py \
  --data-dir data \
  --output submission_punyaa.csv
```

Run only the chronological validation stage:

```bash
python train_rental_model.py \
  --data-dir data \
  --validation-only
```

Optional arguments:

- `--validation-months`: number of recent months held out, default `6`;
- `--iterations`: maximum CatBoost iterations, default `1400`; and
- `--seeds`: one or more model seeds, default `42 2026`.

The generated submission has two columns:

```text
Id,Predicted
```

## Validation design

Validation is chronological rather than random. The latest six labelled months are held out, while all earlier transactions are used for fitting. Training history features use strictly earlier calendar months. Validation features use only the fitting period, and test history features use only the labelled training data.

The script reports validation RMSE and a monthly table containing actual mean rent, predicted mean rent, and prediction bias.

## Testing target-history leakage

Run the standard-library test suite with:

```bash
python -m unittest -v test_target_history.py
```

The tests verify that:

- rows cannot use labels from their own calendar month;
- changing a future target cannot change past features;
- the first month has no target-derived prior;
- row ordering does not alter month-based history;
- rolling counts use only the requested previous months;
- unseen future groups fall back to the labelled reference mean; and
- the feature builder creates the expected 29 history columns.

## Report

The full individual analysis is available in [`punyaa-progress-report.md`](punyaa-progress-report.md). It explains the prediction algorithm, parameters, feature engineering, categorical handling, validation, external data, limitations, and proposed improvements.
