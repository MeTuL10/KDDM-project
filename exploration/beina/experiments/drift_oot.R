# Honest OOT: medians truncated at BENCHMARK_CUTOFF, drift over BENCHMARK_GROWTH_QUARTERS
src <- readLines("model_training.R")
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
suppressMessages(eval(parse(text = src[1:stop_at])))
saved <- readRDS("model_best_params.rds")
best <- saved$best; w <- saved$oot_weights
rounds <- function(m) max(1, round(best[[m]]$best_iter * 1.1))
split <- prepare_split(oot_fit_idx, oot_val_idx)
y <- split$apply$monthly_rent; bench <- split$apply$median_rent_benchmark
preds <- list(ridge = bench + fit_ridge(split$fit, split$apply)$pred)
for (m in c("lgb", "xgb", "cat")) {
  f <- list(lgb = fit_lgb, xgb = fit_xgb, cat = fit_cat)[[m]]
  preds[[m]] <- bench + f(split$fit, split$apply, best[[m]]$params, rounds(m))$pred
}
preds$blend <- as.vector(do.call(cbind, preds[names(w)]) %*% w)
g <- Sys.getenv("BENCHMARK_GROWTH_QUARTERS")
res <- data.table(growth_quarters = g, drift_pct = round(100 * (exp(benchmark_growth) - 1), 3),
                  model = c("benchmark", names(preds)),
                  rmse = c(rmse(y, bench), sapply(preds, function(p) rmse(y, p))))
q <- split$apply$approval_quarter
bias <- data.table(growth_quarters = g, q, err = y - preds$blend)[, .(blend_rmse = sqrt(mean(err^2)), mean_error = mean(err)), by = .(growth_quarters, q)]
fwrite(res, file.path(Sys.getenv("OUTDIR"), paste0("drift_oot_", g, ".csv")))
fwrite(bias, file.path(Sys.getenv("OUTDIR"), paste0("drift_bias_", g, ".csv")))
print(res); print(bias)
