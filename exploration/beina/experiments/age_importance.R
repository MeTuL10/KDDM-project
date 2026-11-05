src <- readLines("model_training.R")
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
suppressMessages(eval(parse(text = src[1:stop_at])))
best <- readRDS("model_best_params.rds")$best
split <- prepare_split(oot_fit_idx, oot_val_idx)
y <- split$apply$monthly_rent; bench <- split$apply$median_rent_benchmark
age_cols <- c("flat_age_at_approval", "lease_commence_date", "year_completed")
all_te <- te_cols
for (use_te in c(TRUE, FALSE)) {
  feature_cols <- c(num_cols, if (use_te) all_te, cat_cols)
  rounds <- if (use_te) 117 else 317
  model <- lgb.train(c(best$lgb$params, list(objective = "regression", num_threads = n_threads, verbosity = -1, seed = 5228)),
                     lgb.Dataset(tree_matrix(split$fit), label = split$fit$residual, categorical_feature = cat_cols),
                     rounds, verbose = -1)
  pred <- function(d) bench + predict(model, tree_matrix(d))
  base <- rmse(y, pred(split$apply))
  inc <- mean(sapply(1:3, function(r) {
    d <- copy(split$apply); i <- sample(nrow(d))
    for (col in age_cols) set(d, j = col, value = d[[col]][i])
    rmse(y, pred(d)) - base
  }))
  # Refit without the age columns at all: does anything else make up for them?
  feature_cols <- setdiff(feature_cols, age_cols)
  model2 <- lgb.train(c(best$lgb$params, list(objective = "regression", num_threads = n_threads, verbosity = -1, seed = 5228)),
                      lgb.Dataset(tree_matrix(split$fit), label = split$fit$residual, categorical_feature = cat_cols),
                      rounds, verbose = -1)
  refit <- rmse(y, bench + predict(model2, tree_matrix(split$apply)))
  cat(sprintf("%s target encodings: OOT %.2f | shuffle age +%.2f | refit without age %.2f (%+.2f)\n",
              if (use_te) "WITH" else "WITHOUT", base, inc, refit, refit - base))
}
