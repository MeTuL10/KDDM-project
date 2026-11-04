# CS5228 HDB monthly-rent prediction

## Team 2 — Residual Drift

| Member | Student ID |
|---|---|
| Aayush Majumdar | A0352046L |
| Metul Prabhu | A0352394Y |
| Punyaa | A0328213M |
| Beina Zhang | A0352621M |

Final model: residual blend of LightGBM, XGBoost and CatBoost around the official quarterly town × flat-type median rent, with block-level hierarchical target encodings and an honest (forecast-time) anchor for test quarters and no COE or stock series. Public leaderboard 432.32714.

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
| `exploration/Aayush/` | Earlier single-LightGBM and median-anchor experiments, including `aayush_exploration.ipynb`, `initial-eda-inference.md`, `median_anchor_experiment.py`, archived modules, and the baseline script/notebook (validation RMSE 451.87) |
| `exploration/Aayush/expt-2/`, `exploration/Aayush/expt-4/`, `exploration/Aayush/expt-5/` | Experiment history and ablations behind the report |
| `exploration/metul/` | Experiments 1–17: shared leakage-safe preprocessing, AutoGluon LightGBM--CatBoost ensembles, feature engineering/selection, deep tabular and time-series models, gating, segmentation, and residual-stack studies. Start with `findings_so_far.md`, `preprocessing.py`, and the experiment notebooks. |
| `report/metul/`, `report/beina/`, `report/punyaa/` | Team-member evidence reports supporting the final paper's EDA, processing, and model claims |
