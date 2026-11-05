library(dplyr)
library(readr)
library(tidyr)

# Experiment with Data/complete.csv: HDB rental approvals 2021-01 .. 2026-09
# (month, town, block, street, flat type, rent), the source of train.csv and
# test.csv, so it contains the real rents of the test rows. It is used here
# only to
#   1. fill gaps in sg-hdb-median-rent.csv with medians computed from it, and
#   2. score test predictions against the real test rents.
# Models still train on train.csv only. The predictions are for analysis and
# are not Kaggle submissions; the "test-quarter medians" variant uses
# medians computed from the test rents themselves.

out_dir <- "complete_experiment"
dir.create(out_dir, showWarnings = FALSE)
# HDB publishes a median only for groups of about 20+ rentals; gaps are filled
# from complete.csv when the group has at least this many rentals
min_rentals <- 5

normalize_keys <- function(df) {
  df |>
    mutate(town = gsub("\\s+", " ", trimws(tolower(town))),
           block = trimws(tolower(block)),
           room_count = as.numeric(ifelse(substr(tolower(flat_type), 1, 1) == "e", "6",
                                          substr(flat_type, 1, 1))),
           key = paste(rent_approval_date, town, block, room_count, sep = "|"))
}

complete <- read_csv("Data/complete.csv", show_col_types = FALSE) |>
  normalize_keys() |>
  mutate(quarter = paste0(substr(rent_approval_date, 1, 4), "-Q",
                          ceiling(as.integer(substr(rent_approval_date, 6, 7)) / 3)))
cat("complete.csv:", nrow(complete), "rentals,", min(complete$rent_approval_date), "to",
    max(complete$rent_approval_date), "\n")

# ---- 1. Fill sg-hdb-median-rent.csv ----
complete_median <- complete |>
  group_by(quarter, town, room_count) |>
  summarise(n_rentals = n(), complete_median = median(monthly_rent), .groups = "drop")

official <- read_csv("Data/auxiliary/sg-hdb-median-rent.csv", na = c("", "na", "NA", "-"),
                     show_col_types = FALSE) |>
  mutate(town_key = gsub("\\s+", " ", trimws(tolower(town))),
         room_count = as.numeric(ifelse(flat_type == "EXEC", "6", substr(flat_type, 1, 1))))

agree <- official |>
  inner_join(complete_median, by = c("quarter", "town_key" = "town", "room_count")) |>
  filter(!is.na(median_rent))
cat(sprintf("Official vs complete.csv medians: %d pairs, %.1f%% identical, mean |diff| S$%.1f\n",
            nrow(agree), 100 * mean(agree$median_rent == agree$complete_median),
            mean(abs(agree$median_rent - agree$complete_median))))

flat_type_code <- c("1-RM", "2-RM", "3-RM", "4-RM", "5-RM", "EXEC")
filled <- official |>
  left_join(complete_median, by = c("quarter", "town_key" = "town", "room_count")) |>
  mutate(source = case_when(!is.na(median_rent) ~ "official",
                            !is.na(complete_median) & n_rentals >= min_rentals ~ "complete.csv",
                            TRUE ~ "missing"),
         median_rent = ifelse(source == "complete.csv", complete_median, median_rent))
# Quarter / town combinations absent from the official file (2026-Q3, tengah)
extra <- complete_median |>
  filter(n_rentals >= min_rentals) |>
  anti_join(official, by = c("quarter", "town" = "town_key", "room_count")) |>
  transmute(quarter, town = toupper(town), flat_type = flat_type_code[room_count],
            median_rent = complete_median, source = "complete.csv (not in official file)")
filled_out <- bind_rows(select(filled, quarter, town, flat_type, median_rent, source), extra) |>
  arrange(quarter, town, flat_type)
filled_file <- file.path(out_dir, "sg-hdb-median-rent-filled.csv")
write_csv(select(filled_out, -source), filled_file, na = "na")

cat("\nMedian rent values from 2021-Q1 onward, by source:\n")
print(filled_out |> filter(quarter >= "2021-Q1") |> count(source) |> mutate(pct = round(100 * n / sum(n), 1)))
cat("Still missing from 2021-Q1, by flat type:\n")
print(filled_out |> filter(quarter >= "2021-Q1", source == "missing") |> count(flat_type))

# ---- 2. Train on train.csv, predict test with the filled medians ----
# predict_test.R is reused unchanged; settings are passed as environment
# variables so the main submission files are not touched.
run_prediction <- function(submission_file, cutoff = NULL) {
  log_file <- file.path(out_dir, sub("\\.csv$", ".log", basename(submission_file)))
  env <- c(paste0("MEDIAN_RENT_FILE=", filled_file),
           paste0("SUBMISSION_FILE=", submission_file),
           if (!is.null(cutoff)) paste0("BENCHMARK_CUTOFF=", cutoff))
  status <- system2("Rscript", "predict_test.R", env = env, stdout = log_file, stderr = log_file)
  if (status != 0) stop("predict_test.R failed, see ", log_file)
  cat("Wrote", submission_file, "\n")
}
# Honest: filled medians up to the last train quarter (2025-Q1), then the
# usual carry-forward with 4-quarter drift
# SKIP_PREDICTIONS=1 re-scores existing prediction files without retraining
if (Sys.getenv("SKIP_PREDICTIONS") != "1") {
  run_prediction("submission_1027_full.csv")
  # Oracle benchmark: filled medians for every test quarter (2025-Q1 .. 2026-Q3)
  run_prediction(file.path(out_dir, "submission_test_quarter_medians.csv"), cutoff = "2026-Q3")
}

# ---- 3. Match test rows to their real rents ----
# complete.csv has no floor area, flat model or lease year, so test rows are
# matched to it on month, town, block, flat type and street (street names are
# abbreviated in complete.csv: "ANG MO KIO AVE 4"). Rows sharing that key form
# a group, and each group is one of:
#   exact      all candidate rents are equal (incl. a single candidate)
#   set_known  after removing the rents of train rows with the same key
#              (2025-03 is in both files), exactly as many candidates remain as
#              there are test rows: the group's rents are known as a set
#   extra      more candidates than test rows (records not sampled into
#              train/test), so which rents belong to the test rows is unknown
# Test rows in a set_known group are identical in every feature, so any
# pairing of rows and rents is equivalent: matched_rent assigns the group's
# rents to its rows in id order, and a model gives every row of the group
# the same prediction, which makes the squared error exact. Extra groups
# get matched_rent = NA, with best / worst case bounds in the scores.
library(data.table)
street_abbreviations <- c(avenue = "ave", street = "st", road = "rd", crescent = "cres", drive = "dr",
                          central = "ctrl", north = "nth", south = "sth", upper = "upp", bukit = "bt",
                          jalan = "jln", kampong = "kg", lorong = "lor", tanjong = "tg", terrace = "ter",
                          place = "pl", close = "cl", commonwealth = "c'wealth", heights = "hts",
                          gardens = "gdns", park = "pk", square = "sq", saint = "st.", market = "mkt")
abbreviate_street <- function(x) {
  streets <- unique(x)
  abbreviated <- vapply(strsplit(gsub("\\s+", " ", trimws(tolower(streets))), " "), function(words) {
    hit <- words %in% names(street_abbreviations)
    words[hit] <- street_abbreviations[words[hit]]
    paste(words, collapse = " ")
  }, character(1))
  abbreviated[match(x, streets)]
}
read_rentals <- function(file, street_col) {
  d <- read_csv(file, show_col_types = FALSE)
  colnames(d) <- tolower(colnames(d))
  d <- normalize_keys(d)
  d$group_key <- paste(d$key, abbreviate_street(d[[street_col]]), sep = "|")
  as.data.table(d)
}
complete_dt <- as.data.table(complete)[, group_key := paste(key, abbreviate_street(street_name), sep = "|")]
train_dt <- read_rentals("Data/train.csv", "street")
# Read as text so the original columns are written back unchanged
test_raw <- read_csv("Data/test.csv", col_types = cols(.default = col_character()))
test_dt <- read_rentals("Data/test.csv", "street")[, id := .I - 1L]
stopifnot(all(test_dt$group_key %in% complete_dt$group_key))

# Candidate rents per key, minus each train rent once (multiset difference)
cand <- complete_dt[, .(group_key, rent = monthly_rent)][, occ := rowid(group_key, rent)]
used <- train_dt[group_key %in% test_dt$group_key, .(group_key, rent = monthly_rent)][, occ := rowid(group_key, rent)]
remaining <- cand[!used, on = .(group_key, rent, occ)]
groups <- test_dt[, .(n_test = .N, features_identical = uniqueN(.SD) == 1, month = rent_approval_date[1]),
                  by = group_key, .SDcols = c("floor_area_sqm", "flat_model", "lease_commence_date")]
groups <- merge(groups, cand[, .(all_unique = uniqueN(rent)), by = group_key], by = "group_key")
groups <- merge(groups, remaining[, .(n_remaining = .N, remaining_unique = uniqueN(rent)), by = group_key],
                by = "group_key", all.x = TRUE)[is.na(n_remaining), `:=`(n_remaining = 0L, remaining_unique = 0L)]
stopifnot(sum(groups$n_test) == nrow(test_dt))
groups[, case := fifelse(all_unique == 1 | (n_remaining == n_test & remaining_unique == 1), "exact",
                         fifelse(n_remaining == n_test, "set_known", "extra"))]

cat("\nTest rows by matching case:\n")
print(groups[, .(groups = .N, rows = sum(n_test), features_identical = mean(features_identical)), by = case][
  , pct_rows := round(100 * rows / sum(rows), 2)][order(case)])

# matched_rent: exact -> the common rent; set_known -> the group's rents in
# ascending order, assigned to its rows in id order; extra -> NA
assigned <- remaining[groups[case == "set_known", .(group_key)], on = "group_key"][order(group_key, rent)][
  , rank := seq_len(.N), by = group_key]
exact_rent <- cand[groups[case == "exact", .(group_key)], on = "group_key"][, .(exact_rent = rent[1]), by = group_key]
test_dt[, rank := rowid(group_key)]
test_dt <- merge(test_dt, groups[, .(group_key, case)], by = "group_key")
test_dt <- merge(test_dt, assigned[, .(group_key, rank, set_rent = rent)], by = c("group_key", "rank"), all.x = TRUE)
test_dt <- merge(test_dt, exact_rent, by = "group_key", all.x = TRUE)
test_dt[, matched_rent := fcoalesce(as.numeric(exact_rent), as.numeric(set_rent))]
setorder(test_dt, id)
stopifnot(nrow(test_dt) == nrow(test_raw))

test_match <- test_raw
test_match$matched_rent <- test_dt$matched_rent
write_csv(test_match, "test_match.csv", na = "")
cat(sprintf("\nWrote test_match.csv: %d rows, matched_rent filled for %d (%.2f%%)\n",
            nrow(test_match), sum(!is.na(test_match$matched_rent)), 100 * mean(!is.na(test_match$matched_rent))))

# ---- 4. Score submissions ----
submissions <- tribble(
  ~file, ~description, ~kaggle_public,
  "submission_0918.csv", "earlier submission (0918)", NA,
  file.path(out_dir, "submission_1027_published_medians.csv"),
  "blend, published test-quarter medians (leaky), with COE/stocks", 433.36655,
  "submission_1027.csv", "final blend: cutoff 2025-Q1 + drift, no COE/stocks", NA,
  "submission_1027_full.csv", "final blend + medians filled from complete.csv, cutoff 2025-Q1 + drift", NA,
  file.path(out_dir, "submission_test_quarter_medians.csv"),
  "final blend + filled medians for all test quarters (oracle benchmark)", NA
) |> filter(file.exists(file))

quarter_of <- function(d) paste0(substr(d, 1, 4), "-Q", ceiling(as.integer(substr(d, 6, 7)) / 3))
extra_rents <- remaining[groups[case == "extra", .(group_key, n_test, n_remaining)], on = "group_key"]
score <- function(file) {
  pred <- fread(file)
  setnames(pred, tolower(names(pred)))
  d <- merge(test_dt[, .(id, group_key, case, matched_rent, month = rent_approval_date)],
             pred[, .(id, predicted)], by = "id")
  # Pairing within set_known groups only matters if predictions differ
  unequal <- d[case == "set_known", .(spread = diff(range(predicted))), by = group_key][spread > 1e-6, .N]
  known <- d[!is.na(matched_rent)]
  # Extra groups: the n_test candidates closest to / farthest from the
  # group prediction give the smallest / largest possible squared error
  ex <- merge(extra_rents, d[case == "extra", .(p = mean(predicted)), by = group_key], by = "group_key")
  ex[, sq := (rent - p)^2]
  setorder(ex, group_key, sq)
  ex[, r := seq_len(.N), by = group_key]
  sse_extra_lo <- ex[r <= n_test, sum(sq)]
  sse_extra_hi <- ex[r > n_remaining - n_test, sum(sq)]
  sse_known <- known[, sum((matched_rent - predicted)^2)]
  list(summary = tibble(
         known_rows = nrow(known),
         rmse_known = sqrt(sse_known / nrow(known)),
         mean_error_known = known[, mean(matched_rent - predicted)],
         rmse_all_lower = sqrt((sse_known + sse_extra_lo) / nrow(d)),
         rmse_all_upper = sqrt((sse_known + sse_extra_hi) / nrow(d)),
         set_known_groups_with_unequal_predictions = unequal),
       by_quarter = as_tibble(known[, .(rows = .N, rmse = sqrt(mean((matched_rent - predicted)^2)),
                                        mean_error = mean(matched_rent - predicted)), by = .(quarter = quarter_of(month))]))
}

scores <- lapply(submissions$file, score)
results <- bind_cols(submissions, bind_rows(lapply(scores, `[[`, "summary")))
by_quarter <- bind_rows(lapply(seq_along(scores), function(i)
  mutate(scores[[i]]$by_quarter, file = basename(submissions$file[i])))) |> arrange(quarter)

options(width = 220)
cat("\nRMSE on test rows with a matched rent, and bounds over all test rows:\n")
print(as.data.frame(results |> select(-file) |> mutate(across(where(is.numeric), ~ round(.x, 2)))))
cat("\nBy quarter, matched rows (rmse / mean error = actual - predicted):\n")
print(as.data.frame(by_quarter |> mutate(cell = sprintf("%.1f / %+.1f", rmse, mean_error)) |>
  select(quarter, rows, file, cell) |> pivot_wider(names_from = file, values_from = cell)))
write_csv(results, file.path(out_dir, "test_rmse.csv"))
write_csv(by_quarter, file.path(out_dir, "test_rmse_by_quarter.csv"))
