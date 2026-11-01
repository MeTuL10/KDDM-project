"""
End-to-end stages of the final model: honest-anchor V2 (LightGBM + XGBoost + CatBoost residual blend).

Stages (main.py and main.ipynb call exactly these, in this order):
  1. build_features()        raw CSVs -> cleaned -> V2 features (published benchmark) for train and test
  2. apply_honest_anchor()   test rows after the last labelled quarter get the frozen-and-drifted benchmark
  3. fit_blend_weights()     out-of-time fit -> blend weights            (models)
  4. train_five_fold()       5-fold residual models -> test predictions  (models)
  5. write_submission()      Id, Predicted csv
Nothing here reads files produced by earlier experiments; every input is under data/.
"""

import numpy as np
import pandas as pd

import config
import cleaning
import features
import anchor
import models


def build_features():
    clean_train, clean_test = cleaning.prepare_cleaned_datasets()
    train, test = features.build_feature_tables(clean_train, clean_test)
    assert train['approval_quarter'].max() == config.CUTOFF_QUARTER, 'train extends past the cutoff quarter'
    return train, test


def apply_honest_anchor(test_published: pd.DataFrame) -> pd.DataFrame:
    return anchor.make_honest(test_published, config.CUTOFF_QUARTER)


def write_submission(predictions, path) -> pd.DataFrame:
    assert len(predictions) == 50000 and np.isfinite(predictions).all()
    sub = pd.DataFrame({'Id': np.arange(len(predictions)), 'Predicted': predictions})
    sub.to_csv(path, index=False)
    return sub


def run(submission_path):
    """Run all stages; returns (submission frame, info dict)."""
    train, test_pub = build_features()
    test = apply_honest_anchor(test_pub)
    weights, oot_rmse = models.fit_blend_weights(train)
    out = models.train_five_fold(train, test, weights)
    sub = write_submission(out['blend'], submission_path)
    return sub, dict(weights=weights, oot_rmse=oot_rmse, oof_rmse=out['oof_rmse'], test_preds=out['test_preds'])
