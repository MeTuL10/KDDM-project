# CS5228 HDB monthly-rent prediction

Final model: residual blend of LightGBM, XGBoost and CatBoost around the official quarterly town × flat-type median rent, with block-level hierarchical target encodings and an honest (forecast-time) anchor for test quarters. Public leaderboard 433.65480.

## Reproduce

```
pip install -r requirements.txt        # macOS: brew install libomp first
python main.py                         # writes submission.csv, about 15-25 min on 8 cores
```

`main.ipynb` runs the same stages with commentary. Inputs are only the files in `data/` (`train.csv`, `test.csv`, `auxiliary/`); nothing from earlier experiments is read. Seeds are fixed; across machines or library versions, expect predictions to agree to within a few SGD rather than exactly. Pass `--compare reference.csv` to `main.py` to print the agreement with another submission file.

## Layout

| Path | Purpose |
|---|---|
| `main.py`, `main.ipynb` | Final pipeline (script and narrated notebook) |
| `modules/` | Cleaning, features, honest anchor, models, stage runner (what `main.py` uses) |
| `archived_modules/`, `baseline_main.py`, `baseline_main.ipynb` | Earlier single-LightGBM pipeline (validation RMSE 451.87), see `exploration/Aayush/progress-report.md` |
| `report/final/` | Paper, figures and the claim-verification tooling (`report/final/README.md`) |
| `exploration/Aayush/expt-2/`, `exploration/Aayush/expt-4/`, `exploration/Aayush/expt-5/` | Experiment history and ablations behind the report |
