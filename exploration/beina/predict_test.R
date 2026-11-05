# Predicts test.csv with the best model from model_training.R: the weighted
# blend of Ridge, LightGBM, XGBoost and CatBoost, each predicting
# rent - median rent benchmark. All models are refit on the full train set
# using the tuned hyperparameters saved in model_best_params.rds.

# Reuse sections 1-4 of model_training.R (data, target encoding, wrappers);
# this also sources data_cleaning.R, which builds train_feature and test_feature.
src <- readLines("model_training.R")
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
eval(parse(text = src[1:stop_at]))

saved <- readRDS("model_best_params.rds")
best <- saved$best
# Weights fitted on the 5-fold out-of-fold predictions over all train months
weights <- saved$cv_weights

# ---- 1. Test frame with the same keys and factor levels as train ----
test_dt <- as.data.table(test_feature)
test_dt[, `:=`(address = paste(block, street, sep = "_"),
               street_key = paste(town, street, sep = "_"),
               town_model = paste(town, flat_model, sep = "_"),
               monthly_rent = NA_real_,
               residual = 0)]
test_dt[, address_room := paste(address, room_count, sep = "_")]
for (col in cat_cols) set(test_dt, j = col, value = factor(test_dt[[col]], levels = levels(df[[col]])))
stopifnot(!anyNA(test_dt[, ..cat_cols]))

n_train <- nrow(df)
df <- rbind(df, test_dt[, names(df), with = FALSE])
train_idx <- seq_len(n_train)
test_idx <- n_train + seq_len(nrow(test_dt))

# ---- 2. Fit on all train rows, predict test ----
# Target encodings are learned from train only (train rows get inner
# out-of-fold encodings, test rows the full-train encodings).
split <- prepare_split(train_idx, test_idx)
bench <- split$apply$median_rent_benchmark
final_rounds <- function(m) max(1, round(best[[m]]$best_iter * 1.1))
fitters <- list(lgb = fit_lgb, xgb = fit_xgb, cat = fit_cat)

preds <- list(ridge = bench + fit_ridge(split$fit, split$apply)$pred)
for (m in c("lgb", "xgb", "cat")) {
  preds[[m]] <- bench + fitters[[m]](split$fit, split$apply, best[[m]]$params, final_rounds(m))$pred
  cat(sprintf("[fit] %s done (%d rounds)\n", m, final_rounds(m)))
}
blend <- as.vector(do.call(cbind, preds[names(weights)]) %*% weights)

# ---- 3. Export in the submission_0918.csv format ----
submission <- data.frame(id = seq_along(blend) - 1, predicted = blend)
out_file <- Sys.getenv("SUBMISSION_FILE", sprintf("submission_%s.csv", format(Sys.Date(), "%m%d")))
write.csv(submission, out_file, row.names = FALSE)

cat("\nBlend weights:\n"); print(round(weights, 3))
cat("\nPrediction summary:\n"); print(summary(blend))
cat("Correlation between models:\n"); print(round(cor(do.call(cbind, preds)), 3))
cat("\nWrote", out_file, "\n")
