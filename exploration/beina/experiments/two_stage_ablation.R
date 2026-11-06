# Reuse sections 1-4 of model_training.R (data, splits, target encoding, wrappers)
src <- readLines("model_training.R")
stop_at <- grep("^# ---- 5. Hyperparameter search", src) - 1
suppressMessages(eval(parse(text = src[1:stop_at])))
lgb_params <- readRDS("model_best_params.rds")$best$lgb$params

macro_cols <- c("coe_qp_mean", "coe_qp_spread", "coe_bids_mean", grep("^stock_", names(df), value = TRUE))
time_cols <- c("time", "approval_year", "approval_month", "month_sin", "month_cos")
median_cols <- c("median_rent_benchmark", "nat_median_rent_benchmark", "median_rent_ratio", "area_to_benchmark_ratio")

monthly <- df[, c(list(avg_rent = mean(monthly_rent)), lapply(.SD, first)),
              by = rent_approval_date, .SDcols = c("time", macro_cols)][order(rent_approval_date)]

# Stage 1: national monthly average rent, fitted on fit months, forecast for later months
stage1 <- function(fit_months, method) {
  m <- copy(monthly)
  fit <- m$rent_approval_date %in% fit_months
  if (method == "oracle") {
    m[, pred := avg_rent]
  } else if (method == "trend") {
    m[, pred := predict(lm(avg_rent ~ time, data = m[fit]), m)]
  } else {
    x <- scale(as.matrix(m[, c("time", macro_cols), with = FALSE]),
               center = colMeans(as.matrix(m[fit, c("time", macro_cols), with = FALSE])),
               scale = apply(as.matrix(m[fit, c("time", macro_cols), with = FALSE]), 2, sd))
    cv <- cv.glmnet(x[fit, ], m$avg_rent[fit], alpha = 0, nfolds = 5)
    m[, pred := predict(cv, x, s = "lambda.min")[, 1]]
  }
  setNames(m$pred, m$rent_approval_date)
}

stage2_base <- setdiff(num_cols, c(time_cols, macro_cols, median_cols))
variants <- list(
  current          = list(base = "official", stage1 = NA,       feats = num_cols),
  ts_trend         = list(base = "national", stage1 = "trend",  feats = stage2_base),
  ts_macro         = list(base = "national", stage1 = "macro",  feats = stage2_base),
  ts_macro_median  = list(base = "national", stage1 = "macro",  feats = c(stage2_base, median_cols)),
  ts_oracle        = list(base = "national", stage1 = "oracle", feats = stage2_base),
  ts_oracle_median = list(base = "national", stage1 = "oracle", feats = c(stage2_base, median_cols))
)

run_split <- function(v, fit_idx, apply_idx, nrounds, early_stop) {
  fit_months <- unique(df$rent_approval_date[fit_idx])
  base <- if (v$base == "official") df$median_rent_benchmark else
    unname(stage1(fit_months, v$stage1)[df$rent_approval_date])
  df[, residual := monthly_rent - base]
  feature_cols <<- c(v$feats, te_cols, cat_cols)
  split <- prepare_split(fit_idx, apply_idx)
  res <- fit_lgb(split$fit, split$apply, lgb_params, nrounds, early_stop = early_stop)
  list(pred = base[apply_idx] + res$pred, base = base[apply_idx], best_iter = res$best_iter)
}

oot_months <- sort(unique(df$rent_approval_date[oot_val_idx]))
actual_avg <- monthly[rent_approval_date %in% oot_months, avg_rent]
out <- list()
for (name in names(variants)) {
  v <- variants[[name]]
  tuned <- run_split(v, tune_fit_idx, tune_val_idx, 5000, TRUE)
  oot <- run_split(v, oot_fit_idx, oot_val_idx, max(1, round(tuned$best_iter * 1.1)), FALSE)
  s1_err <- if (v$base == "national") {
    f <- stage1(unique(df$rent_approval_date[oot_fit_idx]), v$stage1)[oot_months]
    rmse(actual_avg, f)
  } else NA_real_
  out[[name]] <- data.table(variant = name,
                            stage1_monthly_avg_rmse = s1_err,
                            base_only_oot_rmse = rmse(df$monthly_rent[oot_val_idx], oot$base),
                            lgb_oot_rmse = rmse(df$monthly_rent[oot_val_idx], oot$pred),
                            rounds = round(tuned$best_iter * 1.1))
  print(out[[name]])
}
res <- rbindlist(out)
print(res, digits = 6)
fwrite(res, file.path(Sys.getenv("OUTDIR"), "two_stage_ablation.csv"))
