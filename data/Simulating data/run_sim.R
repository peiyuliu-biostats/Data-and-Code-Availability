# ==============================================================================
# run_sim.R (repeated cs simulation)
# ==============================================================================

# 1. force print every line to log
options(echo = TRUE)

# 2. capture and print all errors
tryCatch({
  
  cat("=== script start ===\n")
  
  # --- get arguments ---
  args <- commandArgs(trailingOnly = TRUE)
  print(paste("received args:", paste(args, collapse=" ")))
  
  if (length(args) == 0) {
    TASK_ID <- 1
    cat("no args, defaulting to task_id = 1\n")
  } else {
    TASK_ID <- as.integer(args[1])
  }
  print(paste("parsed task_id:", TASK_ID))
  
  # --- set parameters ---
  REPS_PER_TASK <- 50
  START_SEED    <- (TASK_ID - 1) * REPS_PER_TASK + 1
  OUTPUT_DIR    <- "sim_results"
  
  if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR)
  
  # --- load packages ---
  cat("loading tidyverse...\n")
  library(tidyverse)
  
  # --- load functions ---
  cat("sourcing helper function...\n")
  if (!file.exists("sim-anpp-function-latentid.R")) {
    stop("error: sim-anpp-function-latentid.R not found!")
  }
  source("sim-anpp-function-latentid.R")
  cat("helper function loaded.\n")
  
  # --- model definitions (true_params etc.) ---
  # ensure this section exists to avoid script ending prematurely
  # must include: true_params, duration_levels, lambda_func_fixed, generate_master_dataset_fixed_cohort
  
  TRUE_PARAMS <- list(
    mu_c = log(0.1) - 0.5 * log(0.15/0.1^2 + 1), 
    sigma_c = sqrt(log(0.15/0.1^2 + 1)),
    gamma = log(2)/21,      
    B0 = 0.3,
    sigma_epsilon = 0.15
  )
  DURATION_LEVELS <- c(12, 18, 24, 36)
  SAMPLE_SIZES    <- c(70, 120, 300, 420, 540)  
  BETA_0 <- log(1/15)
  OMEGA <- 2 * pi / 365
  lambda_func_fixed <- function(t) {
    exp(BETA_0 + 0.5 * sin(OMEGA * t) + 0.2 * cos(OMEGA * t))
  }
  generate_master_dataset_fixed_cohort <- function(rep_id) {
    set.seed(rep_id)
    max_months <- max(DURATION_LEVELS)
    max_n <- max(SAMPLE_SIZES)
    times <- seq(30, by=30, length.out = max_months)
    population_pool <- 1:10000
    fixed_cohort_ids <- sample(population_pool, max_n, replace = FALSE)
    all_data <- list()
    for (i in 1:max_months) {
      t_center <- times[i]
      ids_in_round <- fixed_cohort_ids
      t_obs_vec <- t_center + runif(max_n, -2, 2)
      ab_levels <- numeric(max_n)
      for(k in 1:max_n) {
        ab_levels[k] <- sim_individual_ab(
          t_obs = t_obs_vec[k],
          lambda_func = lambda_func_fixed,
          params = TRUE_PARAMS,
          latent_id = ids_in_round[k]
        )
      }
      all_data[[i]] <- tibble(
        time = t_obs_vec,
        antibody = ab_levels,
        survey_month = i,
        row_in_survey = 1:max_n
      )
    }
    bind_rows(all_data)
  }
  
  # --- run loop ---
  cat("starting loop...\n")
  results_container <- list()
  
  for (i in 1:REPS_PER_TASK) {
    current_rep <- START_SEED + i - 1
    cat(sprintf("processing global rep: %d\n", current_rep)) # force progress output
    
    # core logic
    master_df <- generate_master_dataset_fixed_cohort(rep_id = current_rep)
    
    for (dur in DURATION_LEVELS) {
      for (sz in SAMPLE_SIZES) {
        key <- paste0("Rep", current_rep, "_M", dur, "_N", sz)
        subset_df <- master_df %>%
          filter(survey_month <= dur, row_in_survey <= sz) %>%
          select(time, antibody, survey_month)
        results_container[[key]] <- subset_df
      }
    }
  }
  
  # --- save ---
  outfile <- file.path(OUTPUT_DIR, sprintf("sim_batch_%03d.RData", TASK_ID))
  cat(paste("saving to:", outfile, "\n"))
  save(results_container, file = outfile)
  cat("=== script completed ===\n")
  
}, error = function(e) {
  # if error occurs, print message and exit
  cat(paste("\nfatal error:", e$message, "\n"))
  quit(status = 1)
})