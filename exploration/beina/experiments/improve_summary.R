library(data.table)
d <- Sys.getenv("OUTDIR")
res <- rbindlist(lapply(list.files(d, "^improve_.*_seed[0-9]+\\.csv$", full.names = TRUE), fread))
summ <- res[, .(mean = mean(oot_rmse), min = min(oot_rmse), max = max(oot_rmse)), by = .(variant, model)]
base <- summ[variant == "base", .(model, base_mean = mean)]
summ <- merge(summ, base, by = "model")[, change_vs_base := mean - base_mean]
options(width = 160)
print(dcast(summ, model ~ variant, value.var = "mean"), digits = 6)
print(summ[model == "blend", .(variant, mean, min, max, change_vs_base)], digits = 5)
print(res[model == "blend", .(variant, seed, oot_rmse)][order(variant, seed)], digits = 6)
# Idea 1: average the base runs' predictions across seeds
runs <- lapply(1:3, function(s) readRDS(file.path(d, sprintf("improve_base_seed%d.rds", s))))
y <- runs[[1]]$y
bag <- sapply(names(runs[[1]]$pred), function(m) sqrt(mean((y - rowMeans(sapply(runs, function(r) r$pred[[m]])))^2)))
print(data.table(model = names(bag), bagged_3_seeds = bag, single_seed_mean = base[match(names(bag), model), base_mean])[, change := bagged_3_seeds - single_seed_mean][], digits = 6)
# Rounds chosen
print(res[model != "ridge" & model != "blend", .(rounds = round(mean(rounds))), by = .(variant, model)][order(model, variant)])
