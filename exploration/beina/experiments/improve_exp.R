# One improvement idea per run on the honest OOT holdout
# (run with BENCHMARK_CUTOFF=2024-Q2 BENCHMARK_GROWTH_QUARTERS=4).
#   VARIANT=base    current pipeline
#   VARIANT=ratio   target = rent / benchmark - 1 instead of rent - benchmark
#   VARIANT=recency rows weighted 0.5^(months before end of fit window / 24)
#                   in target encodings and in all model fits
#   VARIANT=nomacro drop COE, stock and time columns
# SEED changes the inner target-encoding folds and the model seeds.
variant <- Sys.getenv("VARIANT", "base")
model_seed <- as.integer(Sys.getenv("SEED", "1"))
out_dir <- Sys.getenv("OUTDIR")

src <- readLines("model_training.R")
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
src <- gsub("seed = 5228", "seed = model_seed", src[1:stop_at], fixed = TRUE)
suppressMessages(eval(parse(text = src)))
best <- readRDS("model_best_params.rds")$best
set.seed(model_seed)

# ---- Variant setup ----
if (variant == "ratio") df[, residual := monthly_rent / median_rent_benchmark - 1]
to_rent <- function(pred, d) {
  if (variant == "ratio") d$median_rent_benchmark * (1 + pred) else d$median_rent_benchmark + pred
}
if (variant == "nomacro") {
  drop_cols <- c(grep("^coe_|^stock_", num_cols, value = TRUE),
                 "time", "approval_year", "approval_month", "month_sin", "month_cos")
  num_cols <- setdiff(num_cols, drop_cols)
}
feature_cols <- c(num_cols, te_cols, cat_cols)
half_life <- 24
set_weights <- function(fit_idx) {
  end_time <- max(df$time[fit_idx])
  df[, w := if (variant == "recency") 0.5^((end_time - time) / half_life) else 1]
}

# Weighted versions of the target encoding and model wrappers (w = 1
# reproduces model_training.R exactly)
encode_targets <- function(fit, apply) {
  out <- data.table(.row = seq_len(nrow(apply)))
  global_mean <- weighted.mean(fit$residual, fit$w)
  for (s in te_spec) {
    stats <- fit[, .(s_sum = sum(w * residual), s_n = sum(w)), by = c(s$key)]
    matched <- stats[apply[, s$key, with = FALSE], on = s$key]
    prior <- if (is.na(s$parent)) global_mean else out[[s$parent]]
    set(out, j = s$name, value = (fcoalesce(matched$s_sum, 0) + s$m * prior) /
          (fcoalesce(matched$s_n, 0) + s$m))
  }
  out[, .row := NULL]
}

fit_lgb <- function(fit, valid, params, nrounds, early_stop = FALSE) {
  dtrain <- lgb.Dataset(tree_matrix(fit), label = fit$residual, weight = fit$w,
                        categorical_feature = cat_cols)
  p <- c(params, list(objective = "regression", metric = "rmse",
                      num_threads = n_threads, verbosity = -1, seed = model_seed))
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
  dtrain <- xgb.DMatrix(tree_matrix(fit), label = fit$residual, weight = fit$w, nthread = n_threads)
  dvalid <- xgb.DMatrix(tree_matrix(valid), label = valid$residual, nthread = n_threads)
  p <- c(params, list(objective = "reg:squarederror", eval_metric = "rmse",
                      tree_method = "hist", nthread = n_threads, seed = model_seed))
  if (early_stop) {
    model <- xgb.train(p, dtrain, nrounds, evals = list(valid = dvalid),
                       early_stopping_rounds = 100, verbose = 0)
    nrounds <- which.min(attributes(model)$evaluation_log$valid_rmse)
  } else {
    model <- xgb.train(p, dtrain, nrounds, verbose = 0)
  }
  list(pred = predict(model, dvalid, iterationrange = c(1, nrounds)), best_iter = nrounds)
}

fit_cat <- function(fit, valid, params, nrounds, early_stop = FALSE) {
  p <- c(params, list(iterations = nrounds, loss_function = "RMSE",
                      thread_count = n_threads, logging_level = "Silent", random_seed = model_seed))
  fit_pool <- catboost.load_pool(as.data.frame(fit[, ..feature_cols]), label = fit$residual,
                                 weight = fit$w)
  valid_pool <- cat_pool(valid)
  if (early_stop) {
    p <- c(p, list(od_type = "Iter", od_wait = 100, use_best_model = TRUE))
    model <- catboost.train(fit_pool, valid_pool, params = p)
  } else {
    model <- catboost.train(fit_pool, params = p)
  }
  list(pred = catboost.predict(model, valid_pool), best_iter = model$tree_count)
}

fit_ridge <- function(fit, valid, ...) {
  cv <- cv.glmnet(ridge_matrix(fit), fit$residual, weights = fit$w, alpha = 0, nfolds = 5)
  list(pred = predict(cv, ridge_matrix(valid), s = "lambda.min")[, 1], best_iter = NA)
}

blend_weights <- function(pred_list, actual) {
  p <- do.call(cbind, pred_list)
  loss <- function(theta) rmse(actual, p %*% (exp(theta) / sum(exp(theta))))
  theta <- optim(rep(0, ncol(p)), loss, method = "BFGS")$par
  setNames(exp(theta) / sum(exp(theta)), names(pred_list))
}

# ---- Tune rounds + blend weights on the tuning window, then score OOT ----
run_split <- function(fit_idx, apply_idx) {
  set_weights(fit_idx)
  prepare_split(fit_idx, apply_idx)
}
tune <- run_split(tune_fit_idx, tune_val_idx)
oot <- run_split(oot_fit_idx, oot_val_idx)
fitters <- list(lgb = fit_lgb, xgb = fit_xgb, cat = fit_cat)

tune_pred <- list(ridge = to_rent(fit_ridge(tune$fit, tune$apply)$pred, tune$apply))
oot_pred <- list(ridge = to_rent(fit_ridge(oot$fit, oot$apply)$pred, oot$apply))
rounds <- c()
for (m in names(fitters)) {
  tuned <- fitters[[m]](tune$fit, tune$apply, best[[m]]$params, 5000, early_stop = TRUE)
  tune_pred[[m]] <- to_rent(tuned$pred, tune$apply)
  rounds[m] <- max(1, round(tuned$best_iter * 1.1))
  oot_pred[[m]] <- to_rent(fitters[[m]](oot$fit, oot$apply, best[[m]]$params, rounds[[m]])$pred, oot$apply)
}
w_blend <- blend_weights(tune_pred, tune$apply$monthly_rent)
oot_pred$blend <- as.vector(do.call(cbind, oot_pred[names(w_blend)]) %*% w_blend)

y <- oot$apply$monthly_rent
res <- data.table(variant = variant, seed = model_seed, model = names(oot_pred),
                  oot_rmse = sapply(oot_pred, function(p) rmse(y, p)),
                  rounds = c(ridge = NA, rounds, blend = NA)[names(oot_pred)])
print(res, digits = 6)
fwrite(res, file.path(out_dir, sprintf("improve_%s_seed%d.csv", variant, model_seed)))
saveRDS(list(pred = oot_pred, y = y), file.path(out_dir, sprintf("improve_%s_seed%d.rds", variant, model_seed)))
