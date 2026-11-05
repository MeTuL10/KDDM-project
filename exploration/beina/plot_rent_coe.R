library(dplyr)
library(tidyr)
library(ggplot2)

# Builds train_feature (cleaned train + COE and other auxiliary data)
source("data_cleaning.R")

# Monthly rent is summarised as the median with its interquartile band;
# coe_qp_mean is already one value per month.
monthly <- train_feature |>
  group_by(rent_approval_date) |>
  summarise(rent_median = median(monthly_rent),
            rent_q25 = quantile(monthly_rent, 0.25),
            rent_q75 = quantile(monthly_rent, 0.75),
            coe_qp_mean = first(coe_qp_mean),
            .groups = "drop") |>
  mutate(month = as.Date(paste0(rent_approval_date, "-01")))

# Two measures on different scales get their own panels on a shared time
# axis rather than a dual y-axis.
panel_levels <- c("Monthly rent (SGD), median and IQR",
                  "COE quota premium, mean of categories A & B (SGD)")
plot_data <- bind_rows(
  monthly |> transmute(month, panel = panel_levels[1], value = rent_median,
                       lower = rent_q25, upper = rent_q75),
  monthly |> transmute(month, panel = panel_levels[2], value = coe_qp_mean,
                       lower = NA_real_, upper = NA_real_)
) |>
  mutate(panel = factor(panel, levels = panel_levels))

series_color <- "#2a78d6"
text_secondary <- "#52514e"

p <- ggplot(plot_data, aes(month, value)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = series_color, alpha = 0.15,
              na.rm = TRUE) +
  geom_line(color = series_color, linewidth = 0.8) +
  facet_wrap(~ panel, ncol = 1, scales = "free_y") +
  scale_x_date(date_breaks = "2 months", date_labels = "%m",
               expand = expansion(mult = 0.01)) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Monthly rent and COE premium by rent approval month",
       subtitle = "Train data, January 2021 to March 2025",
       x = "Rent approval month", y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.background = element_rect(fill = "#fcfcfb", color = NA),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "#e6e5e1", linewidth = 0.3),
        strip.text = element_text(hjust = 0, face = "bold", color = "#0b0b0b"),
        plot.subtitle = element_text(color = text_secondary),
        axis.text = element_text(color = text_secondary),
        axis.title.x = element_text(color = text_secondary, margin = margin(t = 8)))

ggsave("rent_coe_over_time.png", p, width = 10, height = 6.5, dpi = 150)
