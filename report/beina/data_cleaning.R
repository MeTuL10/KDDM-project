library(readr)
library(sf)
library(data.table)
library(tidyr)
library(dplyr)
library(lubridate)


# ---- 0. Clean train & test ----
# Read all the data
train <- read_delim("Data/train.csv", delim = ",")
test <- read_delim("Data/test.csv", delim = ",")

coe <- read_delim("Data/auxiliary/sg-coe-prices.csv", delim = ",")
hdb_block <- read_delim("Data/auxiliary/sg-hdb-block.csv", delim = ",")
mrt <- read_delim("Data/auxiliary/sg-mrt-stations.csv", delim = ",")
schools <- read_delim("Data/auxiliary/sg-schools.csv", delim = ",")
shopping_mall <- read_delim("Data/auxiliary/sg-shopping-malls.csv", delim = ",")
stock_price <- read_delim("Data/auxiliary/sg-stock-prices.csv", delim = ",")
median_rent <- read_delim("Data/auxiliary/sg-hdb-median-rent.csv", delim = ",",
                          na = c("", "na", "NA", "-"))
colnames(median_rent) <- tolower(colnames(median_rent))

# "time" is anchored at 0 on the earliest month across train+test so the
# index stays consistent when test (future months) is scored against train.
ref_date <- min(as.Date(paste0(c(train$RENT_APPROVAL_DATE, test$RENT_APPROVAL_DATE), "-01")))

clean_flat_data <- function(df, ref_date) {
  colnames(df) <- tolower(colnames(df))

  df |>
    select(-furnished, -fee) |>
    mutate(
      room_count = as.numeric(ifelse(substr(flat_type, 1, 1) == "e",
                                      "6", substr(flat_type, 1, 1))),
      approval_year = as.integer(substr(rent_approval_date, 1, 4)),
      approval_month = as.integer(substr(rent_approval_date, 6, 7)),
      approval_quarter = paste0(approval_year, "-Q", ceiling(approval_month / 3)),
      time = interval(ref_date, as.Date(paste0(rent_approval_date, "-01"))) %/% months(1),
      flat_model_modified = case_when(
        flat_model %in% c("model a2", "model a", "standard", "simplified", "new generation", "improved") ~ "old_model",
        flat_model %in% c("maisonette", "improved maisonette", "model a maisonette", "premium maisonette") ~ "maisonette",
        flat_model %in% c("apartment", "premium apartment", "premium apartment loft") ~ "apartment",
        flat_model %in% c("type s1", "type s2", "dbss") ~ "dbss",
        flat_model %in% c("3gen", "adjoined flat", "terrace") ~ "special",
        flat_model == "2 room" ~ "compact",
        .default = NA_character_
      )
    )
}

train_cleaned <- clean_flat_data(train, ref_date)
test_cleaned  <- clean_flat_data(test, ref_date)

# ---- Auxiliary data cleaning & spatial features ----

## ---- 1. COE data ----
colnames(coe) <- tolower(colnames(coe))
feature_coe <- coe |>
  mutate(quota_premium_num = as.numeric(gsub("[$,]", "", quota_premium)),
         bids_received_num = as.numeric(gsub(",", "", bids_received)),
         month_num = sprintf("%02d", match(month, tolower(month.abb))),
         yearmonth = paste0(year, "-", month_num)) |>
  filter(category %in% c("a", "b")) |>
  group_by(yearmonth) |>
  summarise(coe_qp_mean = mean(quota_premium_num),
            coe_qp_spread = max(quota_premium_num) - min(quota_premium_num),
            coe_bids_mean = mean(bids_received_num)) |>
  ungroup()

## ---- 2. Stock price data ----
colnames(stock_price) <- tolower(colnames(stock_price))

# No sector field exists in the raw data, so each company is assigned a
# sector by hand from public knowledge of its primary line of business.
stock_sector_map <- tribble(
  ~name,                                   ~sector,
  "DBS Group",                             "financial",
  "OCBC Bank",                             "financial",
  "UOB",                                   "financial",
  "Singapore Exchange",                    "financial",
  "iFAST Corporation",                     "financial",
  "MoneyHero",                             "financial",
  "SWI Capital Holding",                   "financial",

  "Ascendas Reit",                         "real_estate",
  "CapitaLand Ascott Trust",               "real_estate",
  "CapitaLand Investment Limited",         "real_estate",
  "CapitaLand Mall Trust",                 "real_estate",
  "City Developments",                     "real_estate",
  "Frasers Logistics & Industrial Trust",  "real_estate",
  "Hotel Properties Limited",              "real_estate",
  "Keppel REIT",                           "real_estate",
  "Mapletree Commercial Trust",            "real_estate",
  "Mapletree Industrial Trust",            "real_estate",
  "Mapletree Logistics Trust",             "real_estate",
  "Singapore Land",                        "real_estate",
  "Suntec REIT",                           "real_estate",
  "UOL Group",                             "real_estate",
  "Yanlord",                               "real_estate",

  "Singtel",                               "telecom",
  "NetLink Trust",                         "telecom",
  "Pacific Century Regional Developments", "telecom",

  "BOC Aviation",                          "transport",
  "ComfortDelGro",                         "transport",
  "HPH Trust (Hutchison Port)",            "transport",
  "SATS",                                  "transport",
  "SIA Engineering Company",               "transport",
  "Singapore Airlines",                    "transport",
  "Singapore Post",                        "transport",
  "Penguin International Limited",         "transport",

  "Golden Agri-Resources",                 "industrials",
  "Jardine Cycle & Carriage",              "industrials",
  "Keppel",                                "industrials",
  "Kenon Holdings",                        "industrials",
  "Olam",                                  "industrials",
  "Sembcorp",                              "industrials",
  "Seatrium",                              "industrials",
  "ST Engineering",                        "industrials",
  "Wilmar International",                  "industrials",
  "SMJ International",                     "industrials",
  "GCL Global Holdings Ltd",               "industrials",

  "CMON",                                  "consumer",
  "Genting Singapore",                     "consumer",
  "Haw Par",                                "consumer",
  "LightInTheBox Holding",                 "consumer",
  "Sheng Siong Group",                     "consumer",
  "Super Hi International",                "consumer",

  "CytoMed Therapeutics",                  "healthcare",
  "EUDA Health",                           "healthcare",
  "Mobile-health Network Solutions",       "healthcare",
  "Riverstone Holdings",                   "healthcare",
  "Wave Life Sciences",                    "healthcare",

  "Antalpha Platform",                     "technology",
  "BeLive Holdings",                       "technology",
  "Bgin Blockchain",                       "technology",
  "BitFuFu",                               "technology",
  "Bitdeer Technologies Group",            "technology",
  "Enigmatig",                             "technology",
  "Flex",                                  "technology",
  "Genius Group",                          "technology",
  "Grab Holdings",                         "technology",
  "Guardforce AI",                         "technology",
  "IGG Inc",                               "technology",
  "Karooooo",                              "technology",
  "Kulicke and Soffa Industries",          "technology",
  "Ohmyhome",                              "technology",
  "Orangekloud Technology",                "technology",
  "Phaos Technology Holdings",             "technology",
  "Primech Holdings",                      "technology",
  "Ryde Group",                            "technology",
  "SIMPPLE",                               "technology",
  "Sea Limited",                           "technology",
  "Venture Corporation",                   "technology",
  "Webuy Global",                          "technology"
)

stock_price <- stock_price |>
  left_join(stock_sector_map, by = "name")

# Pick the company with the largest average trading volume in each sector
top_stock_per_sector <- stock_price |>
  group_by(sector, name) |>
  summarise(avg_volume = mean(volume, na.rm = TRUE), .groups = "drop") |>
  group_by(sector) |>
  slice_max(avg_volume, n = 1, with_ties = FALSE) |>
  ungroup()

feature_stock <- stock_price |>
  semi_join(top_stock_per_sector, by = c("sector", "name")) |>
  mutate(yearmonth = format(as.Date(date), "%Y-%m"),
         name_clean = gsub("[^a-z0-9]+", "_", tolower(name))) |> 
  group_by(yearmonth, name_clean) |>
  summarise(adjclose_mean = mean(adjusted_close, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(id_cols = "yearmonth", names_from = "name_clean",
              values_from = "adjclose_mean",
              names_glue = "stock_{name_clean}_adjclose")

## ---- 3. School data ----
colnames(schools) <- tolower(colnames(schools))
schools_cleaned <- schools |>
  mutate(type_code_modified = case_when(
    type_code == "government school" ~ "gov sch",
    type_code == "government-aided sch" ~ "gov_aid sch",
    .default = "other"),
    nature_modified = ifelse(natrue_code == "co-ed school", "mix_gender", "single_gender"),
    mainlevel_modified = case_when(
      mainlevel_code == "primary" ~ "primary",
      mainlevel_code %in% c("secondary (s1-s4)", "secondary (s1-s5)") ~ "secondary",
      mainlevel_code %in% c("junior college", "centralised institute") ~ "post_secondary",
      mainlevel_code %in% c("mixed level (s1-jc2)", "mixed level (s1-s5, jc1-jc2)") ~ "s_to_jc",
      mainlevel_code == "mixed level (p1-s4)" ~ "p_to_s",
      TRUE ~ "Other"))

feature_schools <- schools_cleaned |>
  group_by(region, planning_area, postal_code,
           type_code_modified, nature_modified, session_code, mainlevel_modified) |>
  summarise(n_schools = n_distinct(name)) |>
  ungroup()

## ---- 4. HDB Block data ----
colnames(hdb_block) <- tolower(colnames(hdb_block))

# 11 blocks store -1 as a sentinel for unknown max_floor / year_completed.
# max_floor is filled from the median of the same street (then town);
# year_completed is filled per rental row from lease_commence_date in section 7.
hdb_block <- hdb_block |>
  mutate(max_floor = na_if(max_floor, -1),
         year_completed = na_if(year_completed, -1)) |>
  group_by(block, street) |>
  mutate(max_floor = coalesce(max_floor, median(max_floor, na.rm = TRUE))) |>
  group_by(town) |>
  mutate(max_floor = coalesce(max_floor, median(max_floor, na.rm = TRUE))) |>
  ungroup()

colnames(mrt) <- tolower(colnames(mrt))
colnames(shopping_mall) <- tolower(colnames(shopping_mall))

# Filter open vs. planned MRT stations
mrt_open <- mrt[tolower(mrt$status) == "open",]
mrt_planned <- mrt[tolower(mrt$status) == "planned",]

# A school on a combined "mixed level (p1-s4)" track serves both primary and
# secondary students, so it belongs in both the primary and secondary sets.
primary_schools <- schools_cleaned[schools_cleaned$mainlevel_modified %in% c("primary", "p_to_s"),]
secondary_schools <- schools_cleaned[schools_cleaned$mainlevel_modified %in% c("secondary", "p_to_s"),]

# Convert to Spatial sf objects (WGS84 / EPSG:4326)
mrt_sf <- st_as_sf(mrt_open, coords = c("longitude", "latitude"), crs = 4326)
mrt_planned_sf <- st_as_sf(mrt_planned, coords = c("longitude", "latitude"), crs = 4326)
hdb_sf <- st_as_sf(hdb_block, coords = c("longitude", "latitude"), crs = 4326)
sm_sf <- st_as_sf(shopping_mall, coords = c("longitude", "latitude"), crs = 4326)
schools_sf <- st_as_sf(schools_cleaned, coords = c("longitude", "latitude"), crs = 4326)
primary_sf <- st_as_sf(primary_schools, coords = c("longitude", "latitude"), crs = 4326)
secondary_sf <- st_as_sf(secondary_schools, coords = c("longitude", "latitude"), crs = 4326)
cbd_sf <- st_as_sf(data.frame(longitude = 103.8510, latitude = 1.2840),
                   coords = c("longitude", "latitude"), crs = 4326)

# Transform coordinates to SVY21 (EPSG:3414 - Singapore Projected Coordinate System)
# This allows precise meter-based linear distance calculations
mrt_sf_3414 <- st_transform(mrt_sf, crs = 3414)
mrt_planned_sf_3414 <- st_transform(mrt_planned_sf, crs = 3414)
hdb_sf_3414 <- st_transform(hdb_sf, crs = 3414)
sm_sf_3414 <- st_transform(sm_sf, crs = 3414)
school_sf_3414 <- st_transform(schools_sf, crs = 3414)
primary_sf_3414 <- st_transform(primary_sf, crs = 3414)
secondary_sf_3414 <- st_transform(secondary_sf, crs = 3414)
cbd_sf_3414 <- st_transform(cbd_sf, crs = 3414)


## ---- 5. Calculate Distance Matrix & Distinct Station Counts ----

# Compute pairwise linear distances in meters (HDB x MRT)
dist_mrt_open <- st_distance(hdb_sf_3414, mrt_sf_3414)
dist_sm <- st_distance(hdb_sf_3414, sm_sf_3414)
dist_schools <- st_distance(hdb_sf_3414, school_sf_3414)
dist_mrt_planned <- st_distance(hdb_sf_3414, mrt_planned_sf_3414)
dist_primary <- st_distance(hdb_sf_3414, primary_sf_3414)
dist_secondary <- st_distance(hdb_sf_3414, secondary_sf_3414)
dist_cbd <- st_distance(hdb_sf_3414, cbd_sf_3414)

# Nearest distance (km) from each HDB block to any point in a distance matrix
nearest_distance_km <- function(dist_mat) {
  apply(dist_mat, 1, function(row) min(as.numeric(row)) / 1000)
}

# Function to count distinct MRT codes within a specified radius (in meters)
count_distinct_mrt <- function(dist_mat, radius, station_codes) {
  apply(dist_mat, 1, function(row) {
    in_range_codes <- station_codes[as.numeric(row) <= radius]
    return(length(unique(in_range_codes)))
  })
}

count_distinct_sm <- function(dist_mat, radius, shopping_mall_name){
  apply(dist_mat, 1, function(row) {
    in_range_codes <- shopping_mall_name[as.numeric(row) <= radius]
    return(length(unique(in_range_codes)))
  })
}

count_distinct_school <- function(dist_mat, radius, school_name){
  apply(dist_mat, 1, function(row) {
    in_range_codes <- school_name[as.numeric(row) <= radius]
    return(length(unique(in_range_codes)))
  })
}

setDT(hdb_block)
# Add count features to HDB data frame
hdb_block[, mrt_count_500m  := count_distinct_mrt(dist_mrt_open, 500,  mrt_open$code)]
hdb_block[, mrt_count_1000m := count_distinct_mrt(dist_mrt_open, 1000, mrt_open$code)]
hdb_block[, mrt_count_1500m := count_distinct_mrt(dist_mrt_open, 1500, mrt_open$code)]
hdb_block[, shopping_count_500m  := count_distinct_sm(dist_sm, 500, shopping_mall$name)]
hdb_block[, shopping_count_1000m  := count_distinct_sm(dist_sm, 1000, shopping_mall$name)]
hdb_block[, shopping_count_1500m  := count_distinct_sm(dist_sm, 1500, shopping_mall$name)]
hdb_block[, school_count_1km  := count_distinct_school(dist_schools, 1000, schools_cleaned$name)]
hdb_block[, school_count_3km  := count_distinct_school(dist_schools, 3000, schools_cleaned$name)]
hdb_block[, school_count_5km  := count_distinct_school(dist_schools, 5000, schools_cleaned$name)]

# Distance (km) from each unique town/block/street to the nearest POI
hdb_block[, dist_nearest_mrt := nearest_distance_km(dist_mrt_open)]
hdb_block[, dist_nearest_planned_mrt := nearest_distance_km(dist_mrt_planned)]
hdb_block[, dist_nearest_mall := nearest_distance_km(dist_sm)]
hdb_block[, dist_nearest_primary_school := nearest_distance_km(dist_primary)]
hdb_block[, dist_nearest_secondary_school := nearest_distance_km(dist_secondary)]
hdb_block[, dist_to_cbd := nearest_distance_km(dist_cbd)]

# Inverse-distance decay: higher score the closer the nearest POI is. The
# 0.1 km (100 m) floor caps the score at 10 and avoids a divide-by-near-zero
# blow-up for blocks sitting right next to a station/mall.
hdb_block[, mrt_proximity_score := 1 / (dist_nearest_mrt + 0.1)]
hdb_block[, mall_proximity_score := 1 / (dist_nearest_mall + 0.1)]
hdb_block[, school_proximity_score := 1 / (dist_nearest_primary_school + 0.1)]


## ---- 6. Merge with HDB block, COE and stock data ----
# About 20% of rental rows spell STREET in title case ("Ang Mo Kio Avenue 3")
# while the HDB look-up is all lowercase, so the join keys are normalised first.
# After normalisation every town/block/street in train and test matches
# exactly one HDB block, so no fallback imputation is needed.
normalize_address <- function(df) {
  df |>
    mutate(across(c(town, block, street),
                  ~ gsub("\\s+", " ", trimws(tolower(.x)))))
}

hdb_block <- normalize_address(as_tibble(hdb_block))

merge_auxiliary <- function(cleaned, label) {
  merged <- normalize_address(cleaned) |>
    left_join(hdb_block, by = c("town", "block", "street")) |>
    left_join(feature_coe, by = c("rent_approval_date" = "yearmonth")) |>
    left_join(feature_stock, by = c("rent_approval_date" = "yearmonth"))

  # Sanity checks: the joins must not duplicate rows, and every row should find
  # its HDB block and its month's COE / stock values.
  stopifnot(nrow(merged) == nrow(cleaned))
  cat(label, "rows missing HDB block match:", sum(is.na(merged$latitude)), "\n")
  cat(label, "rows missing COE match:      ", sum(is.na(merged$coe_qp_mean)), "\n")
  cat(label, "rows missing stock match:    ",
      sum(!complete.cases(select(merged, starts_with("stock_")))), "\n")
  merged
}

## ---- 7. Derived features ----
# Month is encoded on the unit circle so December and January sit next to
# each other instead of 11 units apart.
# Blocks usually complete about a year before the lease starts (median lag
# of year_completed - lease_commence_date is -1), so that lag fills the gaps.
add_derived_features <- function(df) {
  df |>
    mutate(year_completed = coalesce(year_completed, lease_commence_date - 1),
           flat_age_at_approval = approval_year - year_completed,
           month_sin = sin(2 * pi * approval_month / 12),
           month_cos = cos(2 * pi * approval_month / 12),
           sqm_per_room = floor_area_sqm / room_count,
           cbd_room_interaction = dist_to_cbd * room_count,
           cbd_area_interaction = dist_to_cbd * floor_area_sqm,
           floor_area_interaction = max_floor * floor_area_sqm) |>
    select(-flat_type, -postal_code, -latitude, -longitude)
}

## ---- 8. Merge HDB median rent benchmark ----
# Official quarterly median rent per (town, flat type); "na" and "-" mark
# quarters with too few rentals. Town names carry
# trailing spaces in places ("QUEENSTOWN "), and trimming creates duplicate
# keys, so values are averaged per key. EXEC maps to room_count 6, matching
# clean_flat_data().
#
# Each quarter's median is an aggregate of that quarter's rents, so only
# quarters up to the last train quarter are used: test quarters must not see
# medians computed from the rents being predicted. BENCHMARK_CUTOFF can move
# the cutoff earlier to evaluate the same forecast setup on the OOT holdout.
benchmark_cutoff <- Sys.getenv("BENCHMARK_CUTOFF", max(train_cleaned$approval_quarter))

feature_median_rent <- median_rent |>
  mutate(town = gsub("\\s+", " ", trimws(tolower(town))),
         room_count = as.numeric(ifelse(flat_type == "EXEC", "6", substr(flat_type, 1, 1)))) |>
  group_by(quarter, town, room_count) |>
  summarise(median_rent_benchmark = mean(median_rent, na.rm = TRUE), .groups = "drop") |>
  filter(!is.na(median_rent_benchmark)) |>
  # Drop entry errors such as bukit merah 5-room 2022-Q1 = 300 by removing
  # values below half the cross-town median for that flat type and quarter.
  group_by(quarter, room_count) |>
  filter(median_rent_benchmark >= 0.5 * median(median_rent_benchmark)) |>
  ungroup() |>
  filter(quarter <= benchmark_cutoff)

# National benchmark: median across towns for the same flat type and quarter.
# Quarters where no town published a value borrow the adjacent quarter.
feature_nat_median_rent <- feature_median_rent |>
  group_by(quarter, room_count) |>
  summarise(nat_median_rent_benchmark = median(median_rent_benchmark), .groups = "drop") |>
  complete(quarter, room_count) |>
  group_by(room_count) |>
  arrange(quarter, .by_group = TRUE) |>
  fill(nat_median_rent_benchmark, .direction = "downup") |>
  ungroup()

# 1-room has no published medians since 2021 and 2-room is sparse, so rows
# without a town benchmark fall back to the national median of their flat
# type, then to that of the next larger flat type.
feature_nat_next <- feature_nat_median_rent |>
  transmute(quarter, room_count = room_count - 1,
            nat_next_benchmark = nat_median_rent_benchmark)

# Quarters after the cutoff carry the cutoff quarter's benchmark forward,
# grown by the recent drift of the national rent index (geometric mean of the
# 3-room .. executive national medians): the average quarterly log growth over
# the last BENCHMARK_GROWTH_QUARTERS quarters up to the cutoff.
quarter_number <- function(q) as.integer(substr(q, 1, 4)) * 4 + as.integer(substr(q, 7, 7))
growth_quarters <- as.integer(Sys.getenv("BENCHMARK_GROWTH_QUARTERS", "4"))

rent_index <- feature_nat_median_rent |>
  filter(room_count >= 3) |>
  group_by(quarter) |>
  summarise(log_index = mean(log(nat_median_rent_benchmark)), .groups = "drop") |>
  arrange(quarter)
benchmark_growth <- if (growth_quarters > 0) {
  recent <- tail(rent_index$log_index, growth_quarters + 1)
  (last(recent) - first(recent)) / growth_quarters
} else 0
cat(sprintf("Benchmark cutoff %s, drift %+.2f%% per quarter after it\n",
            benchmark_cutoff, 100 * (exp(benchmark_growth) - 1)))

add_median_rent <- function(df, label) {
  df <- df |>
    mutate(benchmark_quarter = pmin(approval_quarter, benchmark_cutoff),
           drift = exp(benchmark_growth *
                         pmax(0, quarter_number(approval_quarter) - quarter_number(benchmark_cutoff)))) |>
    left_join(feature_median_rent, by = c("benchmark_quarter" = "quarter", "town", "room_count")) |>
    left_join(feature_nat_median_rent, by = c("benchmark_quarter" = "quarter", "room_count")) |>
    left_join(feature_nat_next, by = c("benchmark_quarter" = "quarter", "room_count")) |>
    mutate(nat_median_rent_benchmark = coalesce(nat_median_rent_benchmark, nat_next_benchmark) * drift,
           median_rent_benchmark = coalesce(median_rent_benchmark * drift, nat_median_rent_benchmark),
           median_rent_ratio = median_rent_benchmark / nat_median_rent_benchmark,
           area_to_benchmark_ratio = floor_area_sqm / median_rent_benchmark) |>
    select(-nat_next_benchmark, -benchmark_quarter, -drift)
  cat(label, "rows missing median rent benchmark:", sum(is.na(df$median_rent_benchmark)), "\n")
  df
}

## ---- 9. Build train and test feature tables ----
build_features <- function(cleaned, label) {
  merge_auxiliary(cleaned, label) |>
    add_derived_features() |>
    add_median_rent(label)
}

train_feature <- build_features(train_cleaned, "Train")
test_feature <- build_features(test_cleaned, "Test")
