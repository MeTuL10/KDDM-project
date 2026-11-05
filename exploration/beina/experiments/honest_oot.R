# OOT evaluation as a real forecast made at the end of 2024-06: the median
# rent file is truncated at 2024-Q2 and holdout quarters carry that value forward.
suppressMessages(source("data_cleaning.R"))
cutoff <- "2024-Q2"
feature_median_rent <- feature_median_rent |> filter(quarter <= cutoff)
feature_nat_median_rent <- feature_median_rent |>
  group_by(quarter, room_count) |>
  summarise(nat_median_rent_benchmark = median(median_rent_benchmark), .groups = "drop") |>
  complete(quarter, room_count) |>
  group_by(room_count) |> arrange(quarter, .by_group = TRUE) |>
  fill(nat_median_rent_benchmark, .direction = "downup") |> ungroup()
feature_nat_next <- feature_nat_median_rent |>
  transmute(quarter, room_count = room_count - 1, nat_next_benchmark = nat_median_rent_benchmark)
last_benchmark_quarter <- cutoff
official <- train_feature
train_feature <- train_feature |>
  select(-median_rent_benchmark, -nat_median_rent_benchmark, -median_rent_ratio, -area_to_benchmark_ratio) |>
  add_median_rent("Carried-forward")
stopifnot(identical(names(train_feature), names(official)))

src <- readLines("model_training.R")
src <- src[!grepl('^source\\("data_cleaning.R"\\)', src)]
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
suppressMessages(eval(parse(text = src[1:stop_at])))
saved <- readRDS("model_best_params.rds")
best <- saved$best; w <- saved$oot_weights
rounds <- function(m) max(1, round(best[[m]]$best_iter * 1.1))

cat("benchmark changed on fit rows:", sum(official$median_rent_benchmark[oot_fit_idx] != df$median_rent_benchmark[oot_fit_idx]),
    " on OOT rows:", sum(official$median_rent_benchmark[oot_val_idx] != df$median_rent_benchmark[oot_val_idx]), "\n")
split <- prepare_split(oot_fit_idx, oot_val_idx)
y <- split$apply$monthly_rent; bench <- split$apply$median_rent_benchmark
preds <- list(ridge = bench + fit_ridge(split$fit, split$apply)$pred)
for (m in c("lgb", "xgb", "cat")) {
  f <- list(lgb = fit_lgb, xgb = fit_xgb, cat = fit_cat)[[m]]
  preds[[m]] <- bench + f(split$fit, split$apply, best[[m]]$params, rounds(m))$pred
}
preds$blend <- as.vector(do.call(cbind, preds[names(w)]) %*% w)
res <- data.table(model = c("benchmark only", names(preds)),
                  honest_oot_rmse = c(rmse(y, bench), sapply(preds, function(p) rmse(y, p))))
print(res, digits = 6)
# Error by quarter: carry-forward gets staler further from the cutoff
q <- split$apply$approval_quarter
print(data.table(q, err = y - preds$blend)[, .(rows = .N, blend_rmse = sqrt(mean(err^2)), mean_error = mean(err)), by = q][order(q)], digits = 5)
