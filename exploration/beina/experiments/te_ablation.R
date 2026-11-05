# OOT effect of the in-fold target encodings: same models, same tuned
# hyperparameters, with and without the 7 te_* columns.
src <- readLines("model_training.R")
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
suppressMessages(eval(parse(text = src[1:stop_at])))
best <- readRDS("model_best_params.rds")$best
fitters <- list(lgb = fit_lgb, xgb = fit_xgb, cat = fit_cat)
all_te_cols <- te_cols

blend_weights <- function(pred_list, actual) {
  p <- do.call(cbind, pred_list)
  loss <- function(theta) rmse(actual, p %*% (exp(theta) / sum(exp(theta))))
  theta <- optim(rep(0, ncol(p)), loss, method = "BFGS")$par
  setNames(exp(theta) / sum(exp(theta)), names(pred_list))
}

tune_split <- prepare_split(tune_fit_idx, tune_val_idx)
oot_split <- prepare_split(oot_fit_idx, oot_val_idx)
tune_actual <- tune_split$apply$monthly_rent
oot_actual <- oot_split$apply$monthly_rent

out <- list()
for (use_te in c(TRUE, FALSE)) {
  te_cols <- if (use_te) all_te_cols else character(0)
  feature_cols <- c(num_cols, te_cols, cat_cols)
  tune_pred <- list(ridge = tune_split$apply$median_rent_benchmark +
                      fit_ridge(tune_split$fit, tune_split$apply)$pred)
  oot_pred <- list(ridge = oot_split$apply$median_rent_benchmark +
                     fit_ridge(oot_split$fit, oot_split$apply)$pred)
  rounds <- c(ridge = NA)
  for (m in names(fitters)) {
    tuned <- fitters[[m]](tune_split$fit, tune_split$apply, best[[m]]$params, 5000, early_stop = TRUE)
    tune_pred[[m]] <- tune_split$apply$median_rent_benchmark + tuned$pred
    rounds[m] <- max(1, round(tuned$best_iter * 1.1))
    oot_pred[[m]] <- oot_split$apply$median_rent_benchmark +
      fitters[[m]](oot_split$fit, oot_split$apply, best[[m]]$params, rounds[[m]])$pred
    cat(sprintf("[%s] %-5s oot %.2f\n", if (use_te) "with TE" else "no TE", m, rmse(oot_actual, oot_pred[[m]])))
  }
  w <- blend_weights(tune_pred, tune_actual)
  oot_pred$blend <- as.vector(do.call(cbind, oot_pred[names(w)]) %*% w)
  out[[length(out) + 1]] <- data.table(
    target_encoding = use_te, model = names(oot_pred),
    oot_rmse = sapply(oot_pred, function(p) rmse(oot_actual, p)),
    rounds = c(rounds[names(fitters)], ridge = NA, blend = NA)[names(oot_pred)],
    blend_weight = c(w, blend = NA)[names(oot_pred)])
}
res <- rbindlist(out)
wide <- dcast(res, model ~ target_encoding, value.var = "oot_rmse")
setnames(wide, c("FALSE", "TRUE"), c("no_te", "with_te"))
wide[, change := no_te - with_te]
print(res, digits = 6)
print(wide[order(with_te)], digits = 6)
fwrite(res, file.path(Sys.getenv("OUTDIR"), "te_ablation.csv"))
