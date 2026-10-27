library(dplyr)
library(tidyr)
library(ggplot2)

# Exploratory data analysis for exploratory_data_analysis.md: distributions,
# correlations, time trends and outliers of the cleaned train features.
# Figures are written to eda/, summary numbers to eda/eda_stats.txt.
source("data_cleaning.R")
dir.create("eda", showWarnings = FALSE)
stats_file <- "eda/eda_stats.txt"
sink(stats_file)
options(width = 160)

# Palette (validated default from the dataviz skill)
surface <- "#fcfcfb"
ink <- "#0b0b0b"
ink_2 <- "#52514e"
grid_col <- "#e6e5e1"
series <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
theme_eda <- theme_minimal(base_size = 11) +
  theme(plot.background = element_rect(fill = surface, color = NA),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = grid_col, linewidth = 0.3),
        plot.title = element_text(color = ink, face = "bold"),
        plot.subtitle = element_text(color = ink_2),
        axis.text = element_text(color = ink_2),
        axis.title = element_text(color = ink_2),
        strip.text = element_text(color = ink, hjust = 0))
save_plot <- function(p, file, w = 10, h = 6) ggsave(file.path("eda", file), p, width = w, height = h, dpi = 150)

df <- train_feature |>
  mutate(month = as.Date(paste0(rent_approval_date, "-01")),
         flat_type = factor(ifelse(room_count == 6, "executive", paste0(room_count, "-room")),
                            levels = c("1-room", "2-room", "3-room", "4-room", "5-room", "executive")),
         residual = monthly_rent - median_rent_benchmark,
         rent_per_sqm = monthly_rent / floor_area_sqm)

macro_cols <- grep("^coe_|^stock_", names(df), value = TRUE)
num_cols <- names(df)[sapply(df, is.numeric)] |>
  setdiff(c("monthly_rent", "residual", "rent_per_sqm"))
row_num_cols <- setdiff(num_cols, macro_cols)

# ---- 1. Overview & raw missing values ----
cat("==== 1. Overview ====\n")
cat("train rows", nrow(train), " test rows", nrow(test), " feature columns", ncol(train_feature), "\n")
cat("train months", min(train$RENT_APPROVAL_DATE), "-", max(train$RENT_APPROVAL_DATE),
    " test months", min(test$RENT_APPROVAL_DATE), "-", max(test$RENT_APPROVAL_DATE), "\n")
cat("raw missing values (train):", sum(is.na(train)), " (test):", sum(is.na(test)), "\n")
cat("FURNISHED values:", paste(unique(train$FURNISHED), collapse = ","),
    " FEE values:", paste(unique(train$FEE), collapse = ","), "\n")
cat("FLAT_TYPE spellings:", paste(sort(unique(train$FLAT_TYPE)), collapse = ", "), "\n")
cat("rows per flat type:\n"); print(table(df$flat_type))
cat("rent ending in 00:", round(100 * mean(df$monthly_rent %% 100 == 0), 1), "%  in 50:",
    round(100 * mean(df$monthly_rent %% 50 == 0), 1), "%\n")
cat("\nNumeric summary:\n")
print(as.data.frame(t(sapply(df[c("monthly_rent", row_num_cols)], function(x)
  round(c(min = min(x), q25 = quantile(x, .25, names = FALSE), median = median(x),
          mean = mean(x), q75 = quantile(x, .75, names = FALSE), max = max(x), sd = sd(x)), 2)))))

# ---- 2. Histograms ----
p <- ggplot(df, aes(monthly_rent)) +
  geom_histogram(binwidth = 100, fill = series[1], color = surface, linewidth = 0.2) +
  scale_x_continuous(labels = scales::label_comma()) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Distribution of monthly rent", subtitle = "Train data, S$100 bins",
       x = "Monthly rent (SGD)", y = "Rentals") + theme_eda
save_plot(p, "hist_monthly_rent.png", 9, 4.5)

p <- df |>
  select(all_of(row_num_cols)) |>
  pivot_longer(everything()) |>
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = series[1], color = surface, linewidth = 0.1) +
  facet_wrap(~ name, scales = "free", ncol = 6) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Distributions of numeric features", subtitle = "Train data, 30 bins per panel",
       x = NULL, y = "Rentals") +
  theme_eda + theme(axis.text = element_text(size = 6), strip.text = element_text(size = 7))
save_plot(p, "hist_numeric_features.png", 14, 11)

cat("\nSkewness (|skew| > 1 listed):\n")
skew <- sapply(df[c("monthly_rent", row_num_cols)], function(x) mean((x - mean(x))^3) / sd(x)^3)
print(round(sort(skew[abs(skew) > 1], decreasing = TRUE), 2))

# ---- 3. Correlation ----
cat("\n==== 3. Correlation ====\n")
cor_tab <- tibble(feature = num_cols,
                  pearson_rent = sapply(num_cols, function(f) cor(df[[f]], df$monthly_rent)),
                  spearman_rent = sapply(num_cols, function(f) cor(df[[f]], df$monthly_rent, method = "spearman")),
                  pearson_residual = sapply(num_cols, function(f) cor(df[[f]], df$residual))) |>
  arrange(desc(abs(pearson_rent)))
print(as.data.frame(mutate(cor_tab, across(where(is.numeric), ~ round(.x, 3)))))

p <- cor_tab |>
  select(feature, `Monthly rent` = pearson_rent, `Rent - median benchmark` = pearson_residual) |>
  pivot_longer(-feature, names_to = "target", values_to = "r") |>
  mutate(feature = factor(feature, levels = rev(cor_tab$feature))) |>
  ggplot(aes(r, feature, color = target)) +
  geom_vline(xintercept = 0, color = ink_2, linewidth = 0.3) +
  geom_point(size = 2.4) +
  scale_color_manual(values = series[1:2], name = NULL) +
  labs(title = "Pearson correlation of each numeric feature with the target",
       subtitle = "Sorted by correlation with monthly rent; train data",
       x = "Pearson r", y = NULL) +
  theme_eda + theme(legend.position = "top", axis.text.y = element_text(size = 7))
save_plot(p, "corr_with_target.png", 9, 10)

cm <- cor(df[row_num_cols])
ord <- hclust(as.dist(1 - abs(cm)))$order
cm_long <- as.data.frame(as.table(cm[ord, ord]))
names(cm_long) <- c("a", "b", "r")
p <- ggplot(cm_long, aes(a, b, fill = r)) +
  geom_tile(color = surface, linewidth = 0.2) +
  scale_fill_gradient2(low = "#2a78d6", mid = "#f0efec", high = "#e34948", midpoint = 0,
                       limits = c(-1, 1), name = "Pearson r") +
  labs(title = "Correlation between numeric features",
       subtitle = "Clustered by |r|; red = positive, blue = negative; train data", x = NULL, y = NULL) +
  theme_eda + theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
                    axis.text.y = element_text(size = 7), panel.grid = element_blank())
save_plot(p, "corr_heatmap.png", 11, 10)

cat("\nFeature pairs with |r| >= 0.9:\n")
high <- which(abs(cm) >= 0.9 & upper.tri(cm), arr.ind = TRUE)
print(data.frame(a = rownames(cm)[high[, 1]], b = colnames(cm)[high[, 2]], r = round(cm[high], 3)) |> arrange(desc(abs(r))))

# ---- 4. Time trends ----
cat("\n==== 4. Time trends ====\n")
monthly <- df |>
  group_by(month) |>
  summarise(rentals = n(), median_rent = median(monthly_rent), mean_rent = mean(monthly_rent),
            mean_benchmark = mean(median_rent_benchmark), mean_residual = mean(residual),
            rent_per_sqm = median(rent_per_sqm), .groups = "drop")
yearly <- df |> group_by(approval_year) |>
  summarise(rentals = n(), median_rent = median(monthly_rent), mean_rent = round(mean(monthly_rent)),
            median_rent_per_sqm = round(median(rent_per_sqm), 1), mean_residual = round(mean(residual), 1))
print(as.data.frame(yearly))
cat("mean rent first 3 months:", round(mean(head(monthly$mean_rent, 3))),
    " last 3 months:", round(mean(tail(monthly$mean_rent, 3))), "\n")
by_type_year <- df |> group_by(flat_type, approval_year) |> summarise(median = median(monthly_rent), .groups = "drop") |>
  pivot_wider(names_from = approval_year, values_from = median)
print(as.data.frame(by_type_year))

by_type <- df |> filter(flat_type != "1-room") |>
  group_by(month, flat_type) |> summarise(median_rent = median(monthly_rent), .groups = "drop")
labels <- by_type |> group_by(flat_type) |> slice_max(month, n = 1)
p <- ggplot(by_type, aes(month, median_rent, color = flat_type)) +
  geom_line(linewidth = 0.8) +
  geom_text(data = labels, aes(label = flat_type), hjust = -0.1, size = 3.2, show.legend = FALSE, color = ink_2) +
  scale_color_manual(values = series[2:6], name = "Flat type") +
  scale_x_date(date_breaks = "6 months", date_labels = "%Y-%m", expand = expansion(mult = c(0.01, 0.1))) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Median monthly rent by flat type",
       subtitle = "Train data by rent approval month; 1-room omitted (42 rentals)",
       x = "Rent approval month", y = "Median monthly rent (SGD)") +
  theme_eda + theme(legend.position = "top")
save_plot(p, "trend_rent_by_flat_type.png", 10, 5.5)

p <- ggplot(monthly, aes(month, rentals)) +
  geom_col(fill = series[1], width = 25) +
  scale_x_date(date_breaks = "6 months", date_labels = "%Y-%m") +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Number of rentals per month", subtitle = "Train data", x = "Rent approval month", y = "Rentals") +
  theme_eda
save_plot(p, "trend_rental_volume.png", 10, 4)

p <- monthly |>
  select(month, `Mean monthly rent` = mean_rent, `Mean median-rent benchmark` = mean_benchmark) |>
  pivot_longer(-month) |>
  ggplot(aes(month, value, color = name)) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = series[1:2], name = NULL) +
  scale_x_date(date_breaks = "6 months", date_labels = "%Y-%m") +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Mean monthly rent and the town x flat-type median rent benchmark",
       subtitle = "Train data; the benchmark follows the market level. Mean gap: +S$25-47 in 2021-22, about -S$75 from 2023",
       x = "Rent approval month", y = "SGD") +
  theme_eda + theme(legend.position = "top")
save_plot(p, "trend_rent_vs_benchmark.png", 10, 5)

# COE and stocks indexed to 100 at 2021-01 so they share one axis
macro <- df |> distinct(month, across(all_of(macro_cols))) |> arrange(month)
macro_idx <- macro |>
  mutate(across(all_of(macro_cols), ~ 100 * .x / first(.x))) |>
  select(month, coe_qp_mean, starts_with("stock_")) |>
  pivot_longer(-month) |>
  mutate(name = gsub("^stock_|_adjclose$", "", name), name = gsub("_", " ", name),
         group = ifelse(name == "coe qp mean", "COE quota premium (cat. A & B)", "Stocks (top-volume per sector)"))
rent_idx <- monthly |> transmute(month, value = 100 * mean_rent / first(mean_rent))
p <- ggplot(macro_idx, aes(month, value, group = name)) +
  geom_line(color = "#b9b8b1", linewidth = 0.4) +
  geom_line(data = filter(macro_idx, name == "coe qp mean"), color = series[2], linewidth = 0.9) +
  geom_line(data = rent_idx, aes(group = NULL), color = series[1], linewidth = 1) +
  annotate("text", x = max(rent_idx$month), y = tail(rent_idx$value, 1), label = "Mean rent",
           hjust = -0.1, color = ink_2, size = 3.2) +
  annotate("text", x = max(macro$month), y = tail(filter(macro_idx, name == "coe qp mean")$value, 1),
           label = "COE", hjust = -0.1, color = ink_2, size = 3.2) +
  scale_x_date(date_breaks = "6 months", date_labels = "%Y-%m", expand = expansion(mult = c(0.01, 0.08))) +
  labs(title = "Rent, COE premium and stock prices indexed to January 2021 = 100",
       subtitle = "Grey lines: the 8 stock series used as features; train months only",
       x = "Rent approval month", y = "Index (2021-01 = 100)") +
  theme_eda
save_plot(p, "trend_macro_indexed.png", 10, 5.5)
cat("\nCorrelation of monthly mean rent with monthly macro series (levels / month-on-month changes):\n")
mm <- monthly |> left_join(macro, by = "month")
print(round(sapply(macro_cols, function(c) c(level = cor(mm$mean_rent, mm[[c]]),
                                              change = cor(diff(mm$mean_rent), diff(mm[[c]])))), 3))

# ---- 5. Outliers ----
cat("\n==== 5. Outliers ====\n")
iqr_flag <- function(x, k = 1.5) {
  q <- quantile(x, c(.25, .75)); x < q[1] - k * diff(q) | x > q[2] + k * diff(q)
}
out <- df |>
  group_by(flat_type) |> mutate(out_type = iqr_flag(monthly_rent)) |>
  group_by(flat_type, approval_quarter) |> mutate(out_type_quarter = iqr_flag(monthly_rent)) |>
  ungroup() |>
  mutate(out_global = iqr_flag(monthly_rent),
         out_residual = abs(residual - median(residual)) > 3.5 * 1.4826 * mad(residual),
         out_area = iqr_flag(sqm_per_room, 3))
cat("IQR (1.5x) outliers of monthly_rent - all rows:", sum(out$out_global),
    sprintf("(%.2f%%)", 100 * mean(out$out_global)), "\n")
cat("  within flat type:", sum(out$out_type), sprintf("(%.2f%%)", 100 * mean(out$out_type)), "\n")
cat("  within flat type x quarter:", sum(out$out_type_quarter), sprintf("(%.2f%%)", 100 * mean(out$out_type_quarter)), "\n")
cat("Robust |z| > 3.5 on rent - benchmark:", sum(out$out_residual), sprintf("(%.2f%%)", 100 * mean(out$out_residual)), "\n")
cat("3x IQR outliers of sqm_per_room:", sum(out$out_area), "\n")
cat("\nWithin flat type x quarter outliers by flat type:\n")
print(as.data.frame(out |> group_by(flat_type) |> summarise(rows = n(), outliers = sum(out_type_quarter),
                                                            pct = round(100 * mean(out_type_quarter), 2),
                                                            min_rent = min(monthly_rent), max_rent = max(monthly_rent))))
cat("\n10 highest rents:\n")
print(as.data.frame(df |> slice_max(monthly_rent, n = 10, with_ties = FALSE) |>
  select(rent_approval_date, town, block, street, flat_type, flat_model, floor_area_sqm, monthly_rent, median_rent_benchmark)))
cat("\n10 lowest rents:\n")
print(as.data.frame(df |> slice_min(monthly_rent, n = 10, with_ties = FALSE) |>
  select(rent_approval_date, town, block, street, flat_type, flat_model, floor_area_sqm, monthly_rent, median_rent_benchmark)))
cat("\nfloor_area_sqm by flat type:\n")
print(as.data.frame(df |> group_by(flat_type) |> summarise(min = min(floor_area_sqm), median = median(floor_area_sqm), max = max(floor_area_sqm))))
cat("\nAuxiliary data anomalies: hdb blocks with -1 sentinels = 11; median-rent entry error bukit merah 5-room 2022-Q1 = 300\n")

p <- ggplot(df, aes(flat_type, monthly_rent)) +
  geom_boxplot(fill = "#cde2fb", color = series[1], outlier.color = series[2], outlier.size = 0.6,
               outlier.alpha = 0.5, width = 0.6) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Monthly rent by flat type", subtitle = "Train data; orange points are beyond 1.5 x IQR within the flat type",
       x = NULL, y = "Monthly rent (SGD)") + theme_eda
save_plot(p, "outlier_box_flat_type.png", 9, 5)

p <- ggplot(df, aes(residual)) +
  geom_histogram(binwidth = 100, fill = series[1], color = surface, linewidth = 0.2) +
  geom_vline(xintercept = median(df$residual) + c(-1, 1) * 3.5 * 1.4826 * mad(df$residual),
             color = series[2], linetype = "dashed") +
  scale_x_continuous(labels = scales::label_comma()) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Rent minus town x flat-type median benchmark",
       subtitle = "Dashed lines: robust |z| = 3.5 thresholds; train data",
       x = "Monthly rent - median rent benchmark (SGD)", y = "Rentals") + theme_eda
save_plot(p, "outlier_residual_hist.png", 9, 4.5)

sink()
