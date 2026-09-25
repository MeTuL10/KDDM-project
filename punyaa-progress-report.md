# Progress Report on My HDB Rental Prediction Algorithm

**Student:** Punyaa

**Course:** CS5228 Knowledge Discovery and Data Mining

**Task:** HDB Monthly Rent Prediction

**Date:** 25 September 2026

## 1. Scope of this analysis

This report is derived entirely from the rental-prediction implementation in my workspace, specifically `scripts/train_rental_model.py`. The findings below come from reading that implementation, inspecting the supplied datasets, and executing its feature-building functions.

The repository also contains an older root-level `train_rental_model.py`. I treat the version under `scripts/` as the current implementation because it replaces the older target-history calculation with a causal, month-aware method. The distinction is important for preventing target leakage.

The task is supervised regression. The model receives information about an HDB rental transaction and returns a numerical estimate of `MONTHLY_RENT`. The training set contains 150,000 labelled transactions from January 2021 through March 2025. The test set contains 50,000 unlabelled transactions from March 2025 through July 2026. Because most test records occur after the training period, the task also has a forecasting component.

## 2. Algorithm used

The implementation uses `CatBoostRegressor`, a gradient-boosted decision-tree algorithm with specialised support for categorical variables. Two CatBoost models are trained using different random seeds, and their predictions are averaged.

Gradient boosting builds trees sequentially. The first prediction is a simple baseline. Every later tree learns an adjustment intended to reduce the remaining error. For transaction \(x\), the model can be represented as:

\[
\hat{y}(x)=b+\eta\sum_{m=1}^{M}h_m(x),
\]

where \(b\) is the initial estimate, \(h_m(x)\) is the numerical output of tree \(m\), \(M\) is the selected number of trees, and \(\eta\) is the learning rate.

The model is not a neural network. It therefore has no input, hidden, or output layers. Its structural size is described by tree depth and boosting iterations.

## 3. Model parameters

| Parameter | Implemented value | Function in the algorithm |
|---|---:|---|
| Loss function | RMSE | Trains the model to reduce squared rental error |
| Maximum iterations | 1,400 | Upper limit on the number of trees |
| Learning rate | 0.045 | Scales the contribution of each tree |
| Tree depth | 9 | Allows nine binary decisions along each symmetric tree path |
| L2 leaf regularisation | 8.0 | Penalises large leaf values |
| Random strength | 0.4 | Adds randomness when evaluating candidate splits |
| Bagging temperature | 0.5 | Controls Bayesian bootstrap sampling |
| Early-stopping patience | 150 | Stops if validation RMSE does not improve |
| Random seeds | 42 and 2026 | Produces two related but non-identical models |
| Threads | All available | Parallelises training |

CatBoost normally uses symmetric or oblivious trees: all nodes at the same level apply the same split rule. A complete depth-nine binary tree can have up to \(2^9=512\) terminal leaves. The final model can contain fewer than 1,400 trees because early stopping selects the best validation iteration.

For final training, the implementation takes the best iteration from each validation model, adds a 10% allowance, and caps the result at 1,400. Predictions from the two seed-specific models are averaged:

\[
\hat{y}_{\text{ensemble}}(x)
=\frac{\hat{y}_{42}(x)+\hat{y}_{2026}(x)}{2}.
\]

The final prediction is clipped to the range SGD 300 to SGD 9,000. This prevents extreme values outside the intended rental range from entering the submission.

## 4. How one rental value is calculated

Each tree asks a sequence of learned questions about the transaction. Numerical questions can test whether floor area, flat age, distance, date, or a historical rent statistic is above a threshold. Categorical questions can separate groups of towns, property identifiers, flat types, models, subzones, or regions.

After nine decision levels, the record reaches a leaf containing a numerical adjustment. CatBoost adds the adjustments from all retained trees to the baseline value. This process is performed independently for both random seeds, after which the two values are averaged and clipped.

The prediction is therefore not a lookup value or the rent of the nearest training record. It is the accumulated result of many small rules and interactions. For example, floor area may have one effect for a newer four-room property near an MRT station and a different effect for an older property in another subzone.

## 5. Data cleaning and base features

All text columns are stripped and converted to lowercase. Flat-type spellings such as `4 room` and `4-room` are normalised to the same value. Dates are parsed at monthly resolution.

The base feature builder produces 34 columns for all 200,000 train and test records. Inspection of the resulting frame found 19 numerical columns, 15 categorical columns, and no missing values after the implemented filling logic.

### 5.1 Time and age variables

The transaction date is converted into:

- year;
- month;
- quarter;
- a continuous month index;
- sine of month;
- cosine of month; and
- flat age at the time of approval.

The cyclic month variables are:

\[
\operatorname{MonthSin}=\sin(2\pi m/12), \qquad
\operatorname{MonthCos}=\cos(2\pi m/12).
\]

They place December and January close together in the feature space, which an ordinary integer month does not do.

### 5.2 Composite property categories

The code creates five interaction identifiers:

- property: town, block, and street;
- property and flat type;
- street and flat type;
- town and flat type; and
- flat model and flat type.

These variables allow the model to learn increasingly broad rental patterns. A specific property-and-flat-type group can capture block-level behaviour, while a town-and-flat-type group provides a fallback market segment when property history is limited.

### 5.3 Location variables

The HDB block table is joined using town, block, and street. This adds:

- postal code;
- latitude and longitude;
- maximum floor;
- completion year;
- subzone;
- planning area; and
- region.

The Haversine formula calculates the great-circle distance from each property to the nearest MRT station, school, and shopping mall. The calculations are processed in chunks of 4,000 rental records to limit memory usage.

### 5.4 Economic variables

COE records are converted to monthly keys. Currency symbols and commas are removed before averaging quota premium and prevailing quota premium values for each month. These two variables are joined to the rental transactions as broad indicators of the economic environment.

## 6. Historical target features

The current implementation adds 29 historical columns to the 34 base columns, giving CatBoost 63 input features in total.

Historical statistics are created for five category levels:

| Category level | Smoothing strength |
|---|---:|
| Property and flat type | 6 |
| Property | 8 |
| Street and flat type | 15 |
| Town and flat type | 30 |
| Flat model and flat type | 30 |

For group \(g\), the smoothed historical mean has the form:

\[
\operatorname{HistMean}_g=
\frac{S_g+\alpha\mu}{N_g+\alpha},
\]

where \(S_g\) is the sum of earlier rents in the group, \(N_g\) is its earlier observation count, \(\mu\) is the previous global mean, and \(\alpha\) is the smoothing strength.

Large groups rely mainly on their own history. Rare groups are pulled towards the earlier global market mean. Stronger smoothing is used for broader categories because they contain heterogeneous properties and should not dominate individual predictions too strongly.

For property-and-flat-type, street-and-flat-type, and town-and-flat-type groups, the code also creates rent means and counts over the preceding 6, 12, and 24 months. These windows represent recent, medium-term, and longer-term local market conditions.

The training implementation is causal: a row receives statistics from strictly earlier calendar months. Records from its own month are excluded even if they appear earlier in file order. Validation rows receive statistics constructed only from the earlier fitting set. Test rows receive statistics constructed from the complete labelled training set.

This time restriction is essential. Without it, a transaction's own rent or future rent information could influence its input features.

## 7. Categorical encoding

The code does not manually apply one-hot encoding. All object columns are converted to strings, collected in a categorical-feature list, and passed to CatBoost through `cat_features`.

CatBoost internally builds ordered categorical statistics and category combinations. Ordered construction is intended to reduce the leakage associated with calculating an unrestricted target mean from the entire training dataset.

The full encoding design is therefore:

1. native CatBoost processing for the 15 categorical columns;
2. manually engineered causal historical means and counts; and
3. no explicit pandas or scikit-learn one-hot transformation.

The manual historical variables are ordinary numerical inputs once constructed. They complement CatBoost's internal categorical handling by explicitly representing time direction and recent rental history.

## 8. Training and validation procedure

The final six months of the training period are reserved for validation. With training data ending in March 2025, the default validation period begins in October 2024.

The procedure is:

1. combine train and test predictors for identical non-target preprocessing;
2. build date, location, economic, and composite features;
3. divide labelled records chronologically;
4. build causal history variables for fitting records;
5. map fitting-history variables to validation records;
6. train two CatBoost models using different seeds;
7. stop each model when validation RMSE stops improving;
8. average validation predictions and calculate RMSE;
9. rebuild causal history features using all labelled records;
10. train the two final models using their selected iteration counts;
11. average and clip test predictions; and
12. create an `Id,Predicted` submission file.

Combining train and test predictors in the first step does not expose test labels because the test dataset has none, and target-derived variables are added only after train and test are separated. However, this is still a transductive preprocessing design because the feature frame is aware of the set of test categories. A stricter deployment pipeline could fit every learned preprocessing choice on training data alone, although the present non-target transformations are mainly deterministic cleaning and external joins.

## 9. How performance is scored

The evaluation metric is root mean squared error:

\[
\operatorname{RMSE}=
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}.
\]

If the actual rent is SGD 3,000 and the model predicts SGD 3,400, the error is SGD 400 and its squared contribution is 160,000. Squaring causes large mistakes to influence the score more heavily than several small mistakes.

Lower RMSE is better. The script prints one RMSE value after averaging the two validation models. I do not report a numerical CatBoost RMSE here because the repository does not store a verified output from the current causal implementation, and CatBoost was unavailable in the environment used for this analysis. Reporting a number from another pipeline or an older run would not describe this implementation reliably.

The submission produced from this modelling work received a **Kaggle leaderboard RMSE of 454.38880**. This is the externally reported competition score for the hidden Kaggle evaluation labels. It should not be presented as the local six-month validation RMSE: the two scores use different records, and the Kaggle target values are not available in the repository for independent recomputation.

## 10. External and synthetic data

The model uses the following auxiliary files:

- `sg-hdb-block.csv`;
- `sg-mrt-stations.csv`;
- `sg-schools.csv`;
- `sg-shopping-malls.csv`; and
- `sg-coe-prices.csv`.

The stock-price file is present in the workspace but is not read by this training script. The separate official HDB rental-register file is also not used by this CatBoost pipeline.

No synthetic training rows are created. The code does not use SMOTE, GANs, oversampling, interpolation, or simulation. Historical averages, distances, category interactions, and cyclic dates are engineered features calculated from real records; they do not constitute a synthetic dataset.

## 11. Strengths of the implementation

The main strength is its treatment of time. Chronological validation and strictly earlier-month target histories align the training procedure with the future test period. This is more defensible than a random split for a changing rental market.

The hierarchy of property, street, town, flat type, and model statistics is also appropriate. It gives the model detailed local information while smoothing rare categories towards broader market behaviour.

CatBoost is a suitable model for this feature set because many columns are categorical and have high cardinality. It avoids constructing a very wide one-hot matrix and can learn interactions between categories and numerical attributes.

The two-seed average is a modest form of ensembling. It does not provide the diversity of unrelated algorithms, but it can reduce sensitivity to CatBoost's stochastic sampling and split selection.

Finally, the program includes early stopping, final refitting, prediction clipping, missing-value handling, and automatic submission creation, giving it a complete training-to-output workflow.

## 12. Limitations and implementation risks

First, the current causal model has no stored local validation result. The available Kaggle RMSE is 454.38880, but reproducibility would improve if each run also wrote its parameters, validation dates, best iteration, local RMSE, monthly bias, submission filename, and leaderboard score to a structured experiment file.

Second, `FURNISHED` and `FEE` are retained even though they are constant in the supplied data. CatBoost should normally ignore non-informative variables, but removing them would make the input design clearer.

Third, the script includes school distance without an ablation result demonstrating that it improves future-period RMSE. Every auxiliary feature should be retained only after a consistent chronological comparison.

Fourth, the model uses COE prices as an economic proxy but does not include a rental-specific macroeconomic index. COE values may correlate with time without explaining rental changes causally.

Fifth, CatBoost trees cannot naturally extrapolate a continuous upward rental trend beyond the target values learned during training. Historical mean variables can describe recent levels, but test dates extend substantially beyond March 2025.

Sixth, only one six-month holdout is used for the final check. Results may depend on unusual conditions in those months. Several rolling-origin evaluations would provide a more reliable performance range.

Finally, the root-level version of `train_rental_model.py` uses a historical-mean function in which fitting rows contribute their own labels. That older file should not be used for reported results. The causal implementation under `scripts/` is the appropriate version for this analysis.

## 13. Recommended next work

1. Install CatBoost in a reproducible environment and record the causal model's validation output.
2. Add rolling-origin validation with several historical cut-off dates.
3. Remove constant variables and perform feature-group ablations.
4. Compare the two-seed ensemble with a single seed to quantify its actual benefit.
5. Record feature importance and segment errors by month, town, flat type, and rent band.
6. Add automated tests proving that a training row never uses a rent from its own or a later month.
7. Save every experiment's parameters and metrics in JSON or CSV.
8. Investigate a leakage-safe rental-market trend feature for longer-horizon extrapolation.

## 14. Conclusion

My implemented algorithm is a two-seed CatBoost regression ensemble operating on 63 engineered features. It uses depth-nine symmetric trees, a learning rate of 0.045, up to 1,400 iterations, L2 regularisation, stochastic sampling, and early stopping. Predictions from seeds 42 and 2026 are averaged and clipped to a defined rental range.

The feature design combines transaction attributes, calendar cycles, block metadata, distances to amenities, COE indicators, composite property categories, and smoothed rent history from strictly earlier months. CatBoost handles categorical columns natively, while the code supplies causal target-history variables as numerical features. No explicit one-hot encoding or synthetic dataset is used.

The resulting competition submission achieved a Kaggle leaderboard RMSE of **454.38880**. This provides external evidence that the model generalises to the hidden competition rows, while the absence of a stored local validation log remains a reproducibility limitation.

The strongest aspect of the approach is its attempt to model local rental history without exposing current or future labels. Its main unresolved issue is long-horizon market extrapolation. Before final submission, the causal implementation should be rerun in a pinned environment, evaluated across multiple chronological windows, and accompanied by a stored experiment record.
