# ==============================================================================
# trace plot exploration: all 24 combinations (3 datasets x 8 prior combos)
# purpose: exploratory only — identify convergence quality and select
#          one representative combination for the appendix figure
# outputs: one pdf per dataset (8 panels per pdf, 4 params per panel)
#          plus a convergence summary csv
# ==============================================================================

library(tidyverse)
library(patchwork)

# ------------------------------------------------------------------------------
# 0. paths (reuse from main script)
# ------------------------------------------------------------------------------
RESULTS_DIR <- "C:/Users/TopSoarer/Desktop/MSP data-2026/4-30-real-repcs-infer/results"
OUTPUT_DIR  <- "C:/Users/TopSoarer/Desktop/MSP data-2026/4-30-real-repcs-infer/figures/trace_exploration"
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

DATASETS   <- c("ElangaNdille_2012", "Fustec_KhonKaen", "Fustec_RoiEt")
HALF_LIVES <- c(21, 42, 82, 120)
B0_VALUES  <- c(0.012, 0.044)
PARAM_KEYS <- c("log_a1", "log_a2", "beta_sin", "beta_cos")

PARAM_LABELS <- c(
  log_a1   = expression(log~A[1]),
  log_a2   = expression(log~A[2]),
  beta_sin = expression(beta[sin]),
  beta_cos = expression(beta[cos])
)

make_tag <- function(dataset, t12, b0) {
  b0_str <- gsub("\\.", "p", as.character(b0))
  sprintf("%s__t12_%d__b0_%s", dataset, t12, b0_str)
}

load_trace <- function(dataset, t12, b0) {
  tag  <- make_tag(dataset, t12, b0)
  path <- file.path(RESULTS_DIR, dataset,
                    sprintf("trace_%s.csv", tag))
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}

load_summary <- function(dataset, t12, b0) {
  tag  <- make_tag(dataset, t12, b0)
  path <- file.path(RESULTS_DIR, dataset,
                    sprintf("summary_%s.csv", tag))
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}

# ------------------------------------------------------------------------------
# 1. base theme for trace plots
# ------------------------------------------------------------------------------
trace_theme <- function() {
  theme_bw(base_size = 9) +
    theme(
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(color = "grey92", linewidth = 0.3),
      axis.title        = element_text(size = 8),
      strip.background  = element_rect(fill = "grey90"),
      strip.text        = element_text(size = 8, face = "bold"),
      legend.position   = "none"
    )
}

# ------------------------------------------------------------------------------
# 2. single trace panel: one parameter, all 32 walkers
# ------------------------------------------------------------------------------
build_trace_panel <- function(trace_df, param, param_label,
                              n_walkers = 32, thin = 5) {
  if (is.null(trace_df)) return(ggplot() + theme_void())

  # thin for plotting speed (every 5th step)
  df <- trace_df %>%
    filter(step %% thin == 0) %>%
    select(step, walker, value = all_of(param)) %>%
    mutate(walker_f = factor(walker))

  # compute running mean across all walkers for each step
  run_mean <- df %>%
    group_by(step) %>%
    summarise(mean_val = mean(value), .groups = "drop")

  ggplot() +
    geom_line(data = df,
              aes(x = step, y = value, group = walker_f),
              color = "#378ADD", alpha = 0.18, linewidth = 0.25) +
    geom_line(data = run_mean,
              aes(x = step, y = mean_val),
              color = "#C0392B", linewidth = 0.8) +
    labs(x = "step", y = param_label) +
    trace_theme()
}

# ------------------------------------------------------------------------------
# 3. single acf panel: one parameter, pooled across walkers
# ------------------------------------------------------------------------------
build_acf_panel <- function(trace_df, param, param_label,
                             max_lag = 50) {
  if (is.null(trace_df)) return(ggplot() + theme_void())

  # compute acf for each walker then average
  walkers  <- unique(trace_df$walker)
  acf_list <- map(walkers, function(w) {
    x   <- trace_df %>% filter(walker == w) %>%
      arrange(step) %>% pull(.data[[param]])
    acf_obj <- acf(x, lag.max = max_lag, plot = FALSE)
    tibble(lag = acf_obj$lag[, 1, 1], acf = acf_obj$acf[, 1, 1])
  })
  acf_avg <- bind_rows(acf_list) %>%
    group_by(lag) %>%
    summarise(acf = mean(acf), .groups = "drop")

  # 95% confidence band for white noise
  ci <- qnorm(0.975) / sqrt(max(trace_df$step) + 1)

  ggplot(acf_avg, aes(x = lag, y = acf)) +
    geom_hline(yintercept = 0,   color = "grey50", linewidth = 0.5) +
    geom_hline(yintercept =  ci, color = "#E74C3C",
               linetype = "dashed", linewidth = 0.5) +
    geom_hline(yintercept = -ci, color = "#E74C3C",
               linetype = "dashed", linewidth = 0.5) +
    geom_segment(aes(x = lag, xend = lag, y = 0, yend = acf),
                 color = "#2C3E50", linewidth = 0.5) +
    geom_point(size = 0.8, color = "#2C3E50") +
    labs(x = "lag", y = bquote(.(param_label) ~ "ACF")) +
    trace_theme()
}

# ------------------------------------------------------------------------------
# 4. one 4x2 figure per combination: 4 params x (trace | acf)
# ------------------------------------------------------------------------------
build_combo_figure <- function(dataset, t12, b0) {
  trace_df <- load_trace(dataset, t12, b0)
  if (is.null(trace_df)) {
    message("  trace file not found: ", make_tag(dataset, t12, b0))
    return(NULL)
  }

  # load acceptance and tau from summary
  summ        <- load_summary(dataset, t12, b0)
  accept_str  <- "n/a"
  tau_str     <- "n/a"
  if (!is.null(summ)) {
    acc_row <- summ %>% filter(parameter == "acceptance_fraction")
    if (nrow(acc_row) > 0)
      accept_str <- sprintf("%.3f", acc_row$post_mean[1])
    tau_rows <- summ %>%
      filter(str_starts(parameter, "tau_"))
    if (nrow(tau_rows) > 0)
      tau_str <- paste(sprintf("%.1f", tau_rows$post_mean), collapse = " / ")
  }

  panels <- map(PARAM_KEYS, function(pk) {
    lbl <- PARAM_LABELS[[pk]]
    list(
      trace = build_trace_panel(trace_df, pk, lbl),
      acf   = build_acf_panel(  trace_df, pk, lbl)
    )
  })

  fig <- (
    (panels[[1]]$trace | panels[[1]]$acf) /
    (panels[[2]]$trace | panels[[2]]$acf) /
    (panels[[3]]$trace | panels[[3]]$acf) /
    (panels[[4]]$trace | panels[[4]]$acf)
  ) +
    plot_annotation(
      title    = sprintf("%s  |  t½ = %d d  |  B₀ = %.3f",
                         gsub("_", " ", dataset), t12, b0),
      subtitle = sprintf("acceptance = %s   |   τ (log_a1/log_a2/β_sin/β_cos) = %s",
                         accept_str, tau_str),
      theme    = theme(
        plot.title    = element_text(hjust = 0.5, face = "bold", size = 11),
        plot.subtitle = element_text(hjust = 0.5, size = 9, color = "grey30")
      )
    )
  fig
}

# ------------------------------------------------------------------------------
# 5. convergence summary table across all 24 combinations
# ------------------------------------------------------------------------------
message("building convergence summary ...")

conv_rows <- map_dfr(DATASETS, function(ds) {
  map_dfr(HALF_LIVES, function(t12) {
    map_dfr(B0_VALUES, function(b0) {
      summ <- load_summary(ds, t12, b0)
      if (is.null(summ)) return(tibble())

      acc_row  <- summ %>% filter(parameter == "acceptance_fraction")
      tau_rows <- summ %>% filter(str_starts(parameter, "tau_"))

      acceptance <- if (nrow(acc_row)  > 0) acc_row$post_mean[1]  else NA
      tau_vals   <- if (nrow(tau_rows) > 0) tau_rows$post_mean    else rep(NA, 4)
      n_eff      <- if (all(!is.na(tau_vals))) {
        # flat chain = n_walkers * n_samples rows; n_walkers=32
        n_flat <- load_trace(ds, t12, b0) %>%
          { if (is.null(.)) NA else nrow(.) / 32 * 32 }
        round(n_flat / mean(tau_vals, na.rm = TRUE))
      } else NA

      tibble(
        dataset      = ds,
        half_life    = t12,
        b0           = b0,
        acceptance   = round(acceptance, 3),
        tau_log_a1   = round(tau_vals[1], 1),
        tau_log_a2   = round(tau_vals[2], 1),
        tau_beta_sin = round(tau_vals[3], 1),
        tau_beta_cos = round(tau_vals[4], 1),
        n_eff_approx = n_eff,
        converged    = !is.na(acceptance) &
                       acceptance >= 0.20 & acceptance <= 0.50 &
                       !is.na(n_eff) & n_eff >= 400
      )
    })
  })
})

write_csv(conv_rows,
          file.path(OUTPUT_DIR, "convergence_summary_all.csv"))
message("convergence summary saved")

# print to console for quick inspection
print(conv_rows %>%
        select(dataset, half_life, b0, acceptance, n_eff_approx, converged),
      n = 30)

# flag best candidates for appendix figure
best_candidates <- conv_rows %>%
  filter(converged == TRUE) %>%
  arrange(dataset, desc(n_eff_approx))
message("\nbest candidates for appendix figure:")
print(best_candidates %>% select(dataset, half_life, b0, acceptance, n_eff_approx))

# ------------------------------------------------------------------------------
# 6. generate and save one pdf per dataset (8 combo pages each)
# ------------------------------------------------------------------------------
for (ds in DATASETS) {
  message("\ngenerating trace plots for: ", ds)
  out_path <- file.path(OUTPUT_DIR,
                        sprintf("trace_exploration_%s.pdf", ds))
  pdf(out_path, width = 14, height = 10)

  for (t12 in HALF_LIVES) {
    for (b0 in B0_VALUES) {
      message("  t12=", t12, "  b0=", b0)
      fig <- build_combo_figure(ds, t12, b0)
      if (!is.null(fig)) print(fig)
    }
  }
  dev.off()
  message("  saved: ", out_path)
}

message("\n=== exploration complete ===")
message("next step: inspect pdfs and convergence_summary_all.csv,")
message("then select one representative combination for the appendix figure.")
