src <- readLines("model_training.R")
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
suppressMessages(eval(parse(text = src[1:stop_at])))
best <- readRDS("model_best_params.rds")$best
rounds <- function(m) max(1, round(best[[m]]$best_iter * 1.1))
split <- prepare_split(oot_fit_idx, oot_val_idx)
fit <- split$fit; oot <- split$apply
y <- oot$monthly_rent; bench <- oot$median_rent_benchmark

# Fit the three tree models on the OOT fitting window
lgb_m <- lgb.train(c(best$lgb$params, list(objective = "regression", num_threads = n_threads, verbosity = -1, seed = 5228)),
                   lgb.Dataset(tree_matrix(fit), label = fit$residual, categorical_feature = cat_cols),
                   rounds("lgb"), verbose = -1)
xgb_m <- xgb.train(c(best$xgb$params, list(objective = "reg:squarederror", tree_method = "hist", nthread = n_threads, seed = 5228)),
                   xgb.DMatrix(tree_matrix(fit), label = fit$residual), rounds("xgb"), verbose = 0)
cat_m <- catboost.train(cat_pool(fit), params = c(best$cat$params, list(iterations = rounds("cat"), loss_function = "RMSE",
                        thread_count = n_threads, logging_level = "Silent", random_seed = 5228)))

pred_cat <- function(d) bench + catboost.predict(cat_m, cat_pool(d))
base_rmse <- rmse(y, pred_cat(oot))
cat("CatBoost OOT rmse:", base_rmse, "\n")

# Permutation importance on OOT (CatBoost, best single model), 3 shuffles each
perm <- function(cols) {
  mean(sapply(1:3, function(r) {
    d <- copy(oot); perm_idx <- sample(nrow(d))
    for (col in cols) set(d, j = col, value = d[[col]][perm_idx])
    rmse(y, pred_cat(d)) - base_rmse
  }))
}
single <- rbindlist(lapply(feature_cols, function(f) data.table(feature = f, rmse_increase = perm(f))))
groups <- list(
  `target encodings (all 7)` = te_cols,
  `location TE: address/street/subzone` = c("te_address", "te_address_room", "te_street", "te_subzone"),
  `flat size: floor_area, sqm_per_room, room_count` = c("floor_area_sqm", "sqm_per_room", "room_count"),
  `benchmark-derived` = c("median_rent_benchmark", "nat_median_rent_benchmark", "median_rent_ratio", "area_to_benchmark_ratio"),
  `flat age / lease` = c("flat_age_at_approval", "lease_commence_date", "year_completed"),
  `MRT (counts, distance, score)` = c(grep("^mrt_", feature_cols, value = TRUE), "dist_nearest_mrt", "dist_nearest_planned_mrt"),
  `malls` = c(grep("^shopping_", feature_cols, value = TRUE), "dist_nearest_mall", "mall_proximity_score"),
  `schools` = c(grep("^school_", feature_cols, value = TRUE), "dist_nearest_primary_school", "dist_nearest_secondary_school"),
  `CBD distance + interactions` = c("dist_to_cbd", "cbd_room_interaction", "cbd_area_interaction"),
  `building height` = c("max_floor", "floor_area_interaction"),
  `flat model` = c("flat_model", "flat_model_modified", "te_flat_model", "te_town_model"),
  `town / region / planning area` = c("town", "region", "planning_area", "subzone", "te_town"),
  `time (time, year, month, sin/cos)` = c("time", "approval_year", "approval_month", "month_sin", "month_cos"),
  `COE` = grep("^coe_", feature_cols, value = TRUE),
  `stocks` = grep("^stock_", feature_cols, value = TRUE))
grouped <- rbindlist(lapply(names(groups), function(g) data.table(group = g, n_features = length(groups[[g]]), rmse_increase = perm(groups[[g]]))))

# Built-in gain importance, normalised to %
lgb_gain <- lgb.importance(lgb_m)[, .(feature = Feature, lgb_gain_pct = 100 * Gain / sum(Gain))]
xgb_gain <- as.data.table(xgb.importance(model = xgb_m))[, .(feature = Feature, xgb_gain_pct = 100 * Gain / sum(Gain))]
cat_imp <- catboost.get_feature_importance(cat_m)
cat_gain <- data.table(feature = rownames(cat_imp), cat_pvc_pct = cat_imp[, 1])
tab <- Reduce(function(a, b) merge(a, b, by = "feature", all = TRUE), list(single, lgb_gain, xgb_gain, cat_gain))
for (col in names(tab)[-1]) set(tab, which(is.na(tab[[col]])), col, 0)
setorder(tab, -rmse_increase)
options(width = 200)
print(tab[1:25], digits = 3)
print(grouped[order(-rmse_increase)], digits = 3)
fwrite(tab, file.path(Sys.getenv("OUTDIR"), "feature_importance.csv"))
fwrite(grouped, file.path(Sys.getenv("OUTDIR"), "feature_importance_groups.csv"))
