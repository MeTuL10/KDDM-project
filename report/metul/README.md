# Report compilation plan

The research-paper material is being compiled into three independent Markdown sections so that each can be reviewed before it is merged into the IEEE LaTeX template in `report/cs5228-template.tex`.

| Document | Intended paper content | Status |
| --- | --- | --- |
| [`eda.md`](eda.md) | Dataset description, target drift, structural and geographic patterns, association stability, nonlinear dependence, category coverage, external-data observations, and residual diagnostics | Complete draft |
| [`data_processing.md`](data_processing.md) | Preprocessing pipeline, leakage controls, feature-engineering variants, feature-set comparisons, and validation RMSE attributable to data-processing choices | Complete draft |
| [`model_experiments.md`](model_experiments.md) | Algorithms from Exp. 2–16, comparable validation results, deep-learning/segmentation/MoE/residual-stack experiments, and final best model | Complete draft |

All headline model rankings will use the common chronological split where available: rows before May 2024 for fitting and rows from May 2024 onward for validation. Results from random cross-validation, alternate cutoffs, segment-month targets, or modified-test proxies will be labelled separately rather than placed in the same ranking.

Figures used by the three drafts are under [`figures/`](figures/). They can be regenerated from any working directory with the corresponding `generate_*_figures.py` script in this folder.
