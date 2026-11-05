library(dplyr)
library(data.table)

# Implements Wang, Chan, Wang & Chang (2016), "Predicting Public Housing
# Prices Using Delayed Neural Networks" (Reference Paper1.pdf):
#   Part 1 - static feed-forward ANN (6-h-1, tansig hidden, linear output)
#            on six unit characteristics, trained with Levenberg-Marquardt,
#            several restarts, best run chosen by lowest validation MSE.
#   Part 2 - delayed (time-series) neural network forecasting a quarterly
#            housing price index from its own lagged values.
# Part 1 is evaluated on the same OOT holdout and 5-fold CV as
# model_training.R; Part 2 forecasts the median rent benchmark index.

# Builds train_feature and the median rent tables
source("data_cleaning.R")

set.seed(5228)
# SMOKE=1 runs a tiny configuration to check the pipeline end to end
smoke <- Sys.getenv("SMOKE") == "1"
n_restarts <- if (smoke) 1 else 2
hidden_sizes <- if (smoke) 8 else c(8, 10, 15)
max_epochs <- if (smoke) 5 else 150
n_cores <- 4

# ---- 1. Model frame & splits (identical to model_training.R) ----
df <- as.data.table(train_feature)

rent_bin <- cut(df$monthly_rent, unique(quantile(df$monthly_rent, 0:10 / 10)), include.lowest = TRUE)
cv_fold <- integer(nrow(df))
for (b in levels(rent_bin)) {
  idx <- which(rent_bin == b)
  cv_fold[idx] <- sample(rep_len(1:5, length(idx)))
}

oot_fit_idx <- which(df$rent_approval_date <= "2024-06")
oot_val_idx <- which(df$rent_approval_date >= "2024-07")
tune_fit_idx <- which(df$rent_approval_date <= "2023-12")
tune_val_idx <- which(df$rent_approval_date >= "2024-01" & df$rent_approval_date <= "2024-06")

# Hidden sizes and evaluation splits run in parallel; a parallel-safe RNG is
# switched on only after the folds are drawn so they match model_training.R.
RNGkind("L'Ecuyer-CMRG")
set.seed(5228)

# The paper's six intrinsic inputs. The data has no unit storey, so the
# block's max_floor stands in for "floor / level"; the nearest school is the
# nearer of the nearest primary and secondary school.
df[, `:=`(elapsed_lease = approval_year - lease_commence_date,
          dist_nearest_school = pmin(dist_nearest_primary_school, dist_nearest_secondary_school),
          residual = monthly_rent - median_rent_benchmark)]
paper_inputs <- c("room_count", "max_floor", "floor_area_sqm", "elapsed_lease",
                  "dist_nearest_school", "dist_nearest_mrt")

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))

# ---- 2. Levenberg-Marquardt MLP (MATLAB fitnet / trainlm defaults) ----
# Inputs and target are mapped to [-1, 1] (mapminmax); mu starts at 1e-3,
# is divided by 10 after a successful step and multiplied by 10 otherwise;
# training stops after 6 consecutive validation-error increases (max_fail).
minmax_fit <- function(x) list(lo = apply(x, 2, min), hi = apply(x, 2, max))
minmax_apply <- function(x, s) {
  sweep(sweep(x, 2, s$lo), 2, pmax(s$hi - s$lo, 1e-12) / 2, "/") - 1
}

unpack <- function(theta, h, d) {
  list(W1 = matrix(theta[1:(h * d)], h, d), b1 = theta[h * d + 1:h],
       w2 = theta[h * d + h + 1:h], b2 = theta[h * d + 2 * h + 1])
}

mlp_forward <- function(p, X) {
  z <- tanh(sweep(X %*% t(p$W1), 2, p$b1, "+"))
  list(z = z, yhat = drop(z %*% p$w2) + p$b2)
}

# Jacobian of the network output w.r.t. theta, columns ordered as unpack()
mlp_jacobian <- function(p, X, z) {
  g <- sweep(1 - z^2, 2, p$w2, "*")
  cbind(do.call(cbind, lapply(seq_len(ncol(X)), function(k) g * X[, k])), g, z, 1)
}

nguyen_widrow <- function(h, d) {
  beta <- 0.7 * h^(1 / d)
  W1 <- matrix(runif(h * d, -1, 1), h, d)
  W1 <- beta * W1 / sqrt(rowSums(W1^2))
  c(W1, runif(h, -beta, beta), runif(h, -0.5, 0.5), runif(1, -0.5, 0.5))
}

train_lm <- function(X, y, Xv, yv, h, max_epochs, max_fail = 6) {
  d <- ncol(X)
  sse <- function(theta, X, y) sum((y - mlp_forward(unpack(theta, h, d), X)$yhat)^2)
  theta <- nguyen_widrow(h, d)
  mu <- 1e-3
  best_theta <- theta
  best_val <- sse(theta, Xv, yv)
  fail <- 0
  for (epoch in seq_len(max_epochs)) {
    p <- unpack(theta, h, d)
    f <- mlp_forward(p, X)
    e <- y - f$yhat
    J <- mlp_jacobian(p, X, f$z)
    JtJ <- crossprod(J)
    Jte <- crossprod(J, e)
    improved <- FALSE
    while (mu <= 1e10) {
      step <- tryCatch(solve(JtJ + mu * diag(ncol(J)), Jte), error = function(err) NULL)
      if (!is.null(step)) {
        candidate <- theta + drop(step)
        if (sse(candidate, X, y) < sum(e^2)) {
          theta <- candidate
          mu <- mu * 0.1
          improved <- TRUE
          break
        }
      }
      mu <- mu * 10
    }
    if (!improved) break
    val <- sse(theta, Xv, yv)
    if (val < best_val) {
      best_val <- val
      best_theta <- theta
      fail <- 0
    } else {
      fail <- fail + 1
      if (fail >= max_fail) break
    }
  }
  list(theta = best_theta, h = h, val_mse = best_val / length(yv), epochs = epoch)
}

# Fits a 6-h-1 network on `fit` and predicts monthly rent for `apply`.
# target = "monthly_rent" is the paper as written; target = "residual" trains
# on rent - benchmark like the other models and adds the benchmark back.
# Within `fit`, a random 25% is held out for early stopping, matching the
# paper's 60:20 train:validation division (the `apply` rows are the test set).
fit_paper_ann <- function(fit, apply, target, h, restarts = n_restarts) {
  X <- as.matrix(fit[, ..paper_inputs])
  sx <- minmax_fit(X)
  Xs <- minmax_apply(X, sx)
  y <- fit[[target]]
  sy <- range(y)
  ys <- (y - sy[1]) / (sy[2] - sy[1]) * 2 - 1
  is_val <- runif(nrow(fit)) < 0.25
  runs <- lapply(seq_len(restarts), function(r) {
    train_lm(Xs[!is_val, ], ys[!is_val], Xs[is_val, ], ys[is_val], h, max_epochs)
  })
  best <- runs[[which.min(vapply(runs, `[[`, numeric(1), "val_mse"))]]
  Xa <- minmax_apply(as.matrix(apply[, ..paper_inputs]), sx)
  pred <- mlp_forward(unpack(best$theta, h, ncol(X)), Xa)$yhat
  pred <- (pred + 1) / 2 * (sy[2] - sy[1]) + sy[1]
  if (target == "residual") pred <- pred + apply$median_rent_benchmark
  list(pred = pred, epochs = vapply(runs, `[[`, numeric(1), "epochs"))
}

# ---- 3. Part 1: static ANN - architecture search ----
# The paper compares 6-8-1, 6-10-1 and 6-15-1; the hidden size is chosen on
# the 2024-01 .. 2024-06 tuning window so OOT months are never used.
variants <- c(paper_ann = "monthly_rent", paper_ann_residual = "residual")
best_h <- c()
tune_log <- list()
for (v in names(variants)) {
  started <- Sys.time()
  runs <- parallel::mclapply(hidden_sizes, function(h) {
    fit_paper_ann(df[tune_fit_idx], df[tune_val_idx], variants[[v]], h)
  }, mc.cores = length(hidden_sizes))
  scores <- vapply(runs, function(res) rmse(df$monthly_rent[tune_val_idx], res$pred), numeric(1))
  secs <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  for (i in seq_along(hidden_sizes)) {
    cat(sprintf("[tune] %-18s 6-%d-1  rmse %.2f  epochs %s\n", v, hidden_sizes[i],
                scores[i], paste(runs[[i]]$epochs, collapse = "/")))
  }
  cat(sprintf("[tune] %s done in %.0fs\n", v, secs))
  tune_log[[length(tune_log) + 1]] <- data.table(variant = v, hidden = hidden_sizes, rmse = scores)
  best_h[v] <- hidden_sizes[which.min(scores)]
}
tune_log <- rbindlist(tune_log)

# ---- 4. Part 1: OOT holdout and 5-fold CV ----
paper_results <- list()
# Split 0 is the OOT holdout, splits 1-5 are the CV folds
eval_splits <- c(list(list(fit = oot_fit_idx, apply = oot_val_idx)),
                 lapply(1:5, function(k) list(fit = which(cv_fold != k), apply = which(cv_fold == k))))
for (v in names(variants)) {
  started <- Sys.time()
  preds <- parallel::mclapply(eval_splits, function(s) {
    fit_paper_ann(df[s$fit], df[s$apply], variants[[v]], best_h[[v]])$pred
  }, mc.cores = n_cores)
  oot <- list(pred = preds[[1]])
  oof <- numeric(nrow(df))
  for (k in 1:5) oof[eval_splits[[k + 1]]$apply] <- preds[[k + 1]]
  paper_results[[v]] <- data.table(
    model = sprintf("%s (6-%d-1)", c(paper_ann = "Paper ANN, rent target",
                                     paper_ann_residual = "Paper ANN, residual target")[[v]],
                    best_h[[v]]),
    tune_rmse = min(tune_log[variant == v]$rmse),
    oot_rmse = rmse(df$monthly_rent[oot_val_idx], oot$pred),
    cv5_oof_rmse = rmse(df$monthly_rent, oof))
  cat(sprintf("[eval] %-18s oot %.2f  cv %.2f  %.0fs\n", v, paper_results[[v]]$oot_rmse,
              paper_results[[v]]$cv5_oof_rmse,
              as.numeric(difftime(Sys.time(), started, units = "secs"))))
}
paper_results <- rbindlist(paper_results)

# ---- 5. Part 2: delayed neural network for the rent index ----
# Quarterly rent index = geometric mean of the national median rents of
# 3-room .. executive flats (the only types published in every quarter since
# 2005-Q2). The paper's exogenous macro series are not available before
# 2021, so this is the autoregressive (NAR) form: the index change in
# quarter t is predicted from the changes in t-1 and t-2 (MATLAB's default
# 1:2 delays, 10 hidden neurons). Log changes are modelled instead of the
# raw level, because a tanh network cannot extrapolate above the record
# rents of 2023-24.
rent_index <- feature_nat_median_rent |>
  filter(room_count >= 3) |>
  group_by(quarter) |>
  summarise(log_index = mean(log(nat_median_rent_benchmark)), .groups = "drop") |>
  arrange(quarter)

fit_delayed_nn <- function(series, delays = 2, h = 10, restarts = if (smoke) 2 else 10) {
  lagged <- embed(series, delays + 1)
  y <- lagged[, 1]
  X <- lagged[, -1, drop = FALSE]
  sx <- minmax_fit(X)
  sy <- range(y)
  Xs <- minmax_apply(X, sx)
  ys <- (y - sy[1]) / (sy[2] - sy[1]) * 2 - 1
  runs <- lapply(seq_len(restarts), function(r) {
    # Paper: random 70:15:15 division; the test 15% only reports fit quality
    part <- sample(c("train", "val", "test"), length(y), replace = TRUE, prob = c(0.70, 0.15, 0.15))
    run <- train_lm(Xs[part == "train", , drop = FALSE], ys[part == "train"],
                    Xs[part == "val", , drop = FALSE], ys[part == "val"], h, max_epochs = 1000)
    run$part <- part
    run
  })
  best <- runs[[which.min(vapply(runs, `[[`, numeric(1), "val_mse"))]]
  predict_one <- function(lags) {
    x <- minmax_apply(matrix(lags, nrow = 1), sx)
    (mlp_forward(unpack(best$theta, h, delays), x)$yhat + 1) / 2 * (sy[2] - sy[1]) + sy[1]
  }
  fitted <- vapply(seq_len(nrow(X)), function(i) predict_one(X[i, ]), numeric(1))
  list(predict_one = predict_one, delays = delays,
       r_value = cor(fitted, y), test_mse = mean((fitted - y)[best$part == "test"]^2))
}

# Closed-loop multi-step forecast: predictions are fed back as lags
forecast_delayed_nn <- function(model, series, steps) {
  history <- series
  for (s in seq_len(steps)) {
    lags <- rev(tail(history, model$delays))
    history <- c(history, model$predict_one(lags))
  }
  tail(history, steps)
}

# OOT check: train on index changes up to 2024-Q2, forecast 2024-Q3 .. 2025-Q1
oot_quarters <- c("2024-Q3", "2024-Q4", "2025-Q1")
index_hist <- rent_index[rent_index$quarter <= "2024-Q2", ]
dnn <- fit_delayed_nn(diff(index_hist$log_index))
dnn_growth <- cumsum(forecast_delayed_nn(dnn, diff(index_hist$log_index), length(oot_quarters)))
names(dnn_growth) <- oot_quarters

# Benchmark forecasts for each OOT row, all anchored on information up to
# 2024-Q2: the town/flat-type benchmark of 2024-Q2 is scaled by
#   - delayed NN:   the forecast index growth
#   - naive:        no growth (last value carried forward)
#   - linear trend: a town/flat-type linear trend over the last 8 quarters
#                   (the extrapolation PROGRESS_REPORT 3.1 uses for test)
anchor <- feature_median_rent |>
  filter(quarter == "2024-Q2") |>
  select(town, room_count, anchor_bench = median_rent_benchmark)
nat_anchor <- feature_nat_median_rent |>
  filter(quarter == "2024-Q2") |>
  select(room_count, nat_anchor = nat_median_rent_benchmark)
quarter_num <- function(q) as.integer(substr(q, 1, 4)) * 4 + as.integer(substr(q, 7, 7))
trend <- feature_median_rent |>
  filter(quarter >= "2022-Q3", quarter <= "2024-Q2") |>
  group_by(town, room_count) |>
  filter(n() >= 4) |>
  summarise(slope = coef(lm(median_rent_benchmark ~ quarter_num(quarter)))[2], .groups = "drop")

oot_rows <- df[oot_val_idx, .(town = as.character(town), room_count, approval_quarter,
                              monthly_rent, median_rent_benchmark)] |>
  left_join(anchor, by = c("town", "room_count")) |>
  left_join(nat_anchor, by = "room_count") |>
  left_join(nat_anchor |> transmute(room_count = room_count - 1, nat_next = nat_anchor),
            by = "room_count") |>
  left_join(trend, by = c("town", "room_count")) |>
  mutate(anchor_bench = coalesce(anchor_bench, nat_anchor, nat_next),
         steps = quarter_num(approval_quarter) - quarter_num("2024-Q2"),
         bench_naive = anchor_bench,
         bench_trend = anchor_bench + coalesce(slope, 0) * steps,
         bench_dnn = anchor_bench * exp(dnn_growth[approval_quarter]))

actual_growth <- rent_index$log_index[match(oot_quarters, rent_index$quarter)] -
  rent_index$log_index[rent_index$quarter == "2024-Q2"]
index_results <- data.table(
  quarter = oot_quarters,
  actual_index_change_pct = round(100 * (exp(actual_growth) - 1), 2),
  delayed_nn_change_pct = round(100 * (exp(dnn_growth) - 1), 2))

benchmark_results <- data.table(
  benchmark_source = c("Official median rent (known)", "Delayed NN forecast (paper part 2)",
                       "Town linear trend forecast", "Naive: last quarter carried forward"),
  oot_rmse_benchmark_only = c(rmse(oot_rows$monthly_rent, oot_rows$median_rent_benchmark),
                              rmse(oot_rows$monthly_rent, oot_rows$bench_dnn),
                              rmse(oot_rows$monthly_rent, oot_rows$bench_trend),
                              rmse(oot_rows$monthly_rent, oot_rows$bench_naive)))

# Forecast for 2026-Q3, the test quarter missing from sg-hdb-median-rent.csv
dnn_full <- fit_delayed_nn(diff(rent_index$log_index))
growth_2026q3 <- forecast_delayed_nn(dnn_full, diff(rent_index$log_index), 1)

# ---- 6. Results ----
other_models <- if (file.exists("model_results.csv")) fread("model_results.csv") else NULL
comparison <- rbind(other_models, paper_results)

cat("\nPart 1 - per-flat rent prediction (RMSE, SGD):\n")
print(comparison, digits = 6)
cat("\nPart 1 - architecture search on the tuning window:\n")
print(tune_log, digits = 6)
cat(sprintf("\nPart 2 - delayed NN (NAR, delays 1:2, 10 hidden): R = %.3f, test MSE = %.2e\n",
            dnn$r_value, dnn$test_mse))
print(index_results)
cat("\nPart 2 - benchmark-only rent RMSE on OOT with forecast benchmarks:\n")
print(benchmark_results, digits = 6)
cat(sprintf("\nDelayed NN forecast index change 2026-Q2 -> 2026-Q3: %+.2f%%\n",
            100 * (exp(growth_2026q3) - 1)))

if (!smoke) {
  fwrite(comparison, "paper_ann_comparison.csv")
  fwrite(benchmark_results, "paper_dnn_benchmark_forecast.csv")
}
