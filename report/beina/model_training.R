library(dplyr)
library(data.table)
library(glmnet)
library(lightgbm)
library(xgboost)
library(catboost)

# Builds train_feature (cleaned train + HDB, COE, stock and median rent data)
source("data_cleaning.R")

set.seed(5228)
n_threads <- 4
# SMOKE=1 runs a tiny configuration to check the pipeline end to end
smoke <- Sys.getenv("SMOKE") == "1"
n_trials <- if (smoke) c(lgb = 1, xgb = 1, cat = 1) else c(lgb = 12, xgb = 10, cat = 6)
max_rounds <- if (smoke) 50 else 5000

# ---- 1. Model frame ----
# Models predict the residual over the official median rent benchmark
# (rent - benchmark), which removes the market-wide rent trend that trees
# cannot extrapolate (PROGRESS_REPORT section 3.1).
df <- as.data.table(train_feature)
df[, `:=`(address = paste(block, street, sep = "_"),
          street_key = paste(town, street, sep = "_"),
          town_model = paste(town, flat_model, sep = "_"),
          residual = monthly_rent - median_rent_benchmark)]
df[, address_room := paste(address, room_count, sep = "_")]

cat_cols <- c("town", "flat_model", "flat_model_modified", "subzone", "planning_area", "region")
for (col in cat_cols) set(df, j = col, value = factor(df[[col]]))

id_cols <- c("rent_approval_date", "approval_quarter", "block", "street", "address",
             "address_room", "street_key", "town_model", "monthly_rent", "residual")
# COE and stock columns are kept in train_feature for EDA but not used by the
# models: their permutation importance is zero or negative and dropping them
# leaves the honest OOT RMSE unchanged (model_experiment.md).
macro_cols <- grep("^coe_|^stock_", names(df), value = TRUE)
num_cols <- setdiff(names(df), c(id_cols, cat_cols, macro_cols))

# ---- 2. Validation splits ----
# OOT holdout: fit on <= 2024-06, evaluate on 2024-07 .. 2025-03, mirroring
# the forward gap to the test period. Hyperparameters are tuned on an inner
# time split (fit <= 2023-12, tune on 2024-01 .. 2024-06) so the OOT months
# are never used for model selection.
oot_fit_idx <- which(df$rent_approval_date <= "2024-06")
oot_val_idx <- which(df$rent_approval_date >= "2024-07")
tune_fit_idx <- which(df$rent_approval_date <= "2023-12")
tune_val_idx <- which(df$rent_approval_date >= "2024-01" & df$rent_approval_date <= "2024-06")

# 5-fold CV stratified on rent deciles
rent_bin <- cut(df$monthly_rent, unique(quantile(df$monthly_rent, 0:10 / 10)), include.lowest = TRUE)
cv_fold <- integer(nrow(df))
for (b in levels(rent_bin)) {
  idx <- which(rent_bin == b)
  cv_fold[idx] <- sample(rep_len(1:5, length(idx)))
}

# ---- 3. In-fold hierarchical Bayesian target encoding ----
# Each key is shrunk towards its parent's encoding:
#   enc = (n * mean_residual + M * parent_enc) / (n + M)
# Encodings are learned only from the fitting rows of a split; fitting rows
# themselves get inner 5-fold out-of-fold encodings (PROGRESS_REPORT 5.2).
te_spec <- list(
  list(name = "te_town",         key = "town",         parent = NA,              m = 30),
  list(name = "te_street",       key = "street_key",   parent = "te_town",       m = 15),
  list(name = "te_address",      key = "address",      parent = "te_street",     m = 10),
  list(name = "te_address_room", key = "address_room", parent = "te_address",    m = 5),
  list(name = "te_subzone",      key = "subzone",      parent = "te_town",       m = 20),
  list(name = "te_flat_model",   key = "flat_model",   parent = NA,              m = 20),
  list(name = "te_town_model",   key = "town_model",   parent = "te_flat_model", m = 10)
)
te_cols <- vapply(te_spec, `[[`, character(1), "name")

encode_targets <- function(fit, apply) {
  out <- data.table(.row = seq_len(nrow(apply)))
  global_mean <- mean(fit$residual)
  for (s in te_spec) {
    stats <- fit[, .(s_sum = sum(residual), s_n = .N), by = c(s$key)]
    matched <- stats[apply[, s$key, with = FALSE], on = s$key]
    prior <- if (is.na(s$parent)) global_mean else out[[s$parent]]
    set(out, j = s$name, value = (fcoalesce(matched$s_sum, 0) + s$m * prior) /
          (fcoalesce(matched$s_n, 0L) + s$m))
  }
  out[, .row := NULL]
}

prepare_split <- function(fit_idx, apply_idx, k = 5) {
  fit <- df[fit_idx]
  apply <- df[apply_idx]
  inner <- sample(rep_len(1:k, nrow(fit)))
  fit_te <- data.table()
  for (i in 1:k) {
    enc <- encode_targets(fit[inner != i], fit[inner == i])
    enc[, .row := which(inner == i)]
    fit_te <- rbind(fit_te, enc)
  }
  setorder(fit_te, .row)
  fit_te[, .row := NULL]
  list(fit = cbind(fit, fit_te), apply = cbind(apply, encode_targets(fit, apply)))
}

# ---- 4. Model wrappers ----
# Each wrapper fits on `fit` and returns residual predictions for `valid`.
# With early_stop = TRUE, `valid` also drives early stopping (tuning only).
feature_cols <- c(num_cols, te_cols, cat_cols)

tree_matrix <- function(d) {
  x <- d[, ..feature_cols]
  for (col in cat_cols) set(x, j = col, value = as.integer(x[[col]]))
  as.matrix(x)
}

fit_lgb <- function(fit, valid, params, nrounds, early_stop = FALSE) {
  dtrain <- lgb.Dataset(tree_matrix(fit), label = fit$residual,
                        categorical_feature = cat_cols)
  p <- c(params, list(objective = "regression", metric = "rmse",
                      num_threads = n_threads, verbosity = -1, seed = 5228))
  if (early_stop) {
    dvalid <- lgb.Dataset.create.valid(dtrain, tree_matrix(valid), label = valid$residual)
    model <- lgb.train(p, dtrain, nrounds, valids = list(valid = dvalid),
                       early_stopping_rounds = 100, verbose = -1)
    nrounds <- model$best_iter
  } else {
    model <- lgb.train(p, dtrain, nrounds, verbose = -1)
  }
  list(pred = predict(model, tree_matrix(valid), num_iteration = nrounds), best_iter = nrounds)
}

fit_xgb <- function(fit, valid, params, nrounds, early_stop = FALSE) {
  dtrain <- xgb.DMatrix(tree_matrix(fit), label = fit$residual, nthread = n_threads)
  dvalid <- xgb.DMatrix(tree_matrix(valid), label = valid$residual, nthread = n_threads)
  p <- c(params, list(objective = "reg:squarederror", eval_metric = "rmse",
                      tree_method = "hist", nthread = n_threads, seed = 5228))
  if (early_stop) {
    model <- xgb.train(p, dtrain, nrounds, evals = list(valid = dvalid),
                       early_stopping_rounds = 100, verbose = 0)
    nrounds <- which.min(attributes(model)$evaluation_log$valid_rmse)
  } else {
    model <- xgb.train(p, dtrain, nrounds, verbose = 0)
  }
  list(pred = predict(model, dvalid, iterationrange = c(1, nrounds)), best_iter = nrounds)
}

cat_pool <- function(d) {
  catboost.load_pool(as.data.frame(d[, ..feature_cols]), label = d$residual)
}

fit_cat <- function(fit, valid, params, nrounds, early_stop = FALSE) {
  p <- c(params, list(iterations = nrounds, loss_function = "RMSE",
                      thread_count = n_threads, logging_level = "Silent", random_seed = 5228))
  valid_pool <- cat_pool(valid)
  if (early_stop) {
    p <- c(p, list(od_type = "Iter", od_wait = 100, use_best_model = TRUE))
    model <- catboost.train(cat_pool(fit), valid_pool, params = p)
  } else {
    model <- catboost.train(cat_pool(fit), params = p)
  }
  list(pred = catboost.predict(model, valid_pool), best_iter = model$tree_count)
}

# Ridge uses numeric + target-encoded features and one-hot low-cardinality
# categories; its penalty is tuned by glmnet's internal 5-fold CV.
ridge_matrix <- function(d) {
  model.matrix(~ ., data = d[, c(num_cols, te_cols, "town", "flat_model", "region"), with = FALSE])[, -1]
}

fit_ridge <- function(fit, valid, ...) {
  cv <- cv.glmnet(ridge_matrix(fit), fit$residual, alpha = 0, nfolds = 5)
  list(pred = predict(cv, ridge_matrix(valid), s = "lambda.min")[, 1], best_iter = NA)
}

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))

# ---- 5. Hyperparameter search ----
# Random search on the inner time split; each draw early-stops on 2024-01 ..
# 2024-06 and is scored by rent RMSE (benchmark + predicted residual).
search_space <- list(
  lgb = function() list(learning_rate = 0.05,
                        num_leaves = sample(c(31, 63, 127, 255), 1),
                        max_depth = sample(c(-1, 8, 12), 1),
                        min_data_in_leaf = sample(c(20, 50, 100, 200), 1),
                        feature_fraction = runif(1, 0.5, 0.9),
                        bagging_fraction = runif(1, 0.6, 0.95), bagging_freq = 1,
                        lambda_l2 = 10^runif(1, -1, 1.5)),
  xgb = function() list(eta = 0.05,
                        max_depth = sample(c(6, 8, 10), 1),
                        min_child_weight = sample(c(1, 5, 20, 50), 1),
                        subsample = runif(1, 0.6, 0.95),
                        colsample_bytree = runif(1, 0.5, 0.9),
                        lambda = 10^runif(1, -1, 1.5)),
  cat = function() list(learning_rate = 0.08,
                        depth = sample(c(6, 8), 1),
                        l2_leaf_reg = 10^runif(1, 0, 1),
                        random_strength = runif(1, 0.5, 2),
                        bagging_temperature = runif(1, 0, 1))
)
fitters <- list(ridge = fit_ridge, lgb = fit_lgb, xgb = fit_xgb, cat = fit_cat)

tune_split <- prepare_split(tune_fit_idx, tune_val_idx)
tune_actual <- tune_split$apply$monthly_rent
tune_bench <- tune_split$apply$median_rent_benchmark

tuning_log <- list()
best <- list()
tune_pred <- list(ridge = tune_bench + fit_ridge(tune_split$fit, tune_split$apply)$pred)

for (m in names(search_space)) {
  for (trial in seq_len(n_trials[[m]])) {
    params <- search_space[[m]]()
    started <- Sys.time()
    res <- fitters[[m]](tune_split$fit, tune_split$apply, params, max_rounds, early_stop = TRUE)
    score <- rmse(tune_actual, tune_bench + res$pred)
    secs <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    cat(sprintf("[tune] %-3s trial %2d  rmse %.2f  rounds %4d  %.0fs\n",
                m, trial, score, res$best_iter, secs))
    tuning_log[[length(tuning_log) + 1]] <- data.table(
      model = m, trial = trial, rmse = score, best_iter = res$best_iter,
      params = paste(names(params), signif(unlist(params), 4), sep = "=", collapse = ", "))
    if (is.null(best[[m]]) || score < best[[m]]$rmse) {
      best[[m]] <- list(params = params, best_iter = res$best_iter, rmse = score)
      tune_pred[[m]] <- tune_bench + res$pred
    }
  }
}
tuning_log <- rbindlist(tuning_log)

# Final fits train on more rows than the tuning fit, so they get 10% more
# rounds than early stopping chose.
final_rounds <- function(m) max(1, round(best[[m]]$best_iter * 1.1))

# ---- 6. Blend weights ----
# Non-negative weights summing to 1 (softmax parametrisation), minimising RMSE.
fit_blend_weights <- function(pred_list, actual) {
  p <- do.call(cbind, pred_list)
  loss <- function(theta) rmse(actual, p %*% (exp(theta) / sum(exp(theta))))
  theta <- optim(rep(0, ncol(p)), loss, method = "BFGS")$par
  setNames(exp(theta) / sum(exp(theta)), names(pred_list))
}

# ---- 7. Out-of-time holdout evaluation ----
oot_split <- prepare_split(oot_fit_idx, oot_val_idx)
oot_actual <- oot_split$apply$monthly_rent
oot_bench <- oot_split$apply$median_rent_benchmark
oot_pred <- list(ridge = oot_bench + fit_ridge(oot_split$fit, oot_split$apply)$pred)
for (m in names(search_space)) {
  res <- fitters[[m]](oot_split$fit, oot_split$apply, best[[m]]$params, final_rounds(m))
  oot_pred[[m]] <- oot_bench + res$pred
  cat(sprintf("[oot] %-5s rmse %.2f\n", m, rmse(oot_actual, oot_pred[[m]])))
}
# Blend weights come from the tuning window, never from the OOT months
oot_weights <- fit_blend_weights(tune_pred, tune_actual)
oot_pred$blend <- as.vector(do.call(cbind, oot_pred[names(oot_weights)]) %*% oot_weights)

# ---- 8. 5-fold cross-validation ----
oof_pred <- lapply(c(fitters, list(blend = NULL)), function(x) numeric(nrow(df)))
for (k in 1:5) {
  split <- prepare_split(which(cv_fold != k), which(cv_fold == k))
  bench <- split$apply$median_rent_benchmark
  idx <- which(cv_fold == k)
  oof_pred$ridge[idx] <- bench + fit_ridge(split$fit, split$apply)$pred
  for (m in names(search_space)) {
    oof_pred[[m]][idx] <- bench + fitters[[m]](split$fit, split$apply, best[[m]]$params,
                                               final_rounds(m))$pred
  }
  cat(sprintf("[cv] fold %d done\n", k))
}
# Weights fitted on the OOF predictions themselves (as in the report's SLSQP
# stacking), so the CV blend score is slightly optimistic.
cv_weights <- fit_blend_weights(oof_pred[names(fitters)], df$monthly_rent)
oof_pred$blend <- as.vector(do.call(cbind, oof_pred[names(cv_weights)]) %*% cv_weights)

# ---- 9. Results ----
model_names <- c(benchmark = "Median rent benchmark only", ridge = "Ridge",
                 lgb = "LightGBM", xgb = "XGBoost", cat = "CatBoost", blend = "Weighted blend")
oot_pred$benchmark <- oot_bench
oof_pred$benchmark <- df$median_rent_benchmark
tune_pred$benchmark <- tune_bench

results <- data.table(
  model = unname(model_names),
  tune_rmse = sapply(names(model_names), function(m)
    if (m == "blend") NA_real_ else rmse(tune_actual, tune_pred[[m]])),
  oot_rmse = sapply(names(model_names), function(m) rmse(oot_actual, oot_pred[[m]])),
  cv5_oof_rmse = sapply(names(model_names), function(m) rmse(df$monthly_rent, oof_pred[[m]]))
)

print(results, digits = 6)
cat("\nOOT blend weights:\n"); print(round(oot_weights, 3))
cat("CV blend weights:\n"); print(round(cv_weights, 3))
cat("\nBest parameters:\n")
for (m in names(best)) {
  cat(sprintf("%-4s rounds=%d  %s\n", m, final_rounds(m),
              paste(names(best[[m]]$params), signif(unlist(best[[m]]$params), 4),
                    sep = "=", collapse = ", ")))
}

if (!smoke) {
  fwrite(results, "model_results.csv")
  fwrite(tuning_log, "model_tuning_log.csv")
  saveRDS(list(best = best, oot_weights = oot_weights, cv_weights = cv_weights),
          "model_best_params.rds")
}
