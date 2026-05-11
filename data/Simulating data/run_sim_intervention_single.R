# ==============================================================================
# run_sim_intervention_single.R
# hipergator simulation script for single intervention scenario
# ==============================================================================

# 1. force print logs for debugging
options(echo = TRUE)

tryCatch({
  cat("=== script start ===\n")
  
  # --- 0. get command line arguments (slurm array task id) ---
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    TASK_ID <- 1
    cat("warning: no args provided. defaulting to task id 1.\n")
  } else {
    TASK_ID <- as.integer(args[1])
  }
  
  # --- configure parameters ---
  REPS_PER_TASK <- 50   # 50 reps per task (total 20 tasks = 1000 reps)
  START_SEED    <- (TASK_ID - 1) * REPS_PER_TASK + 1
  OUTPUT_DIR    <- "sim_int_single_results" # modify folder name to differentiate
  
  if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR)
  
  cat(sprintf("task id: %d | processing reps: %d to %d\n", 
              TASK_ID, START_SEED, START_SEED + REPS_PER_TASK - 1))
  
  # ==============================================================================
  # 1. load libraries and functions
  # ==============================================================================
  suppressPackageStartupMessages(library(tidyverse))
  
  if (!file.exists("sim-anpp-function-latentid.R")) {
    stop("critical error: 'sim-anpp-function-latentid.R' not found!")
  }
  source("sim-anpp-function-latentid.R")
  
  # ==============================================================================
  # 2. define constants and parameters
  # ==============================================================================
  # fixed constants
  B0 <- 0.3
  T_INT <- 910          # intervention time (day)
  OMEGA <- 2 * pi / 365
  BETA_0 <- log(1/15)   
  
  # experimental design: keep only [moderate intervention + 14-day half-life]
  # moderate intervention usually refers to efficacy = 60% (beta = log(1-0.60))
  TEST_SCENARIOS <- list(
    scenario_target = list(
      name = "S_Mid_G14", 
      true_beta = log(1 - 0.60), # moderate intervention (60% efficacy)
      true_gamma_hl = 14         # specify 14-day half-life
    )
  )
  
  # sample size list
  SAMPLE_SIZES <- c(15, 70, 120, 300)
  
  # observation time points (offset relative to t_int)
  OFFSETS <- c(0, 15, 30, 45) 
  
  # base parameters
  BASE_ANPP_PARAMS <- list(
    c_mean = 0.2,
    c_var = 0.25,
    B0 = B0,
    sigma_epsilon = 0.15 
  )
  
  # ==============================================================================
  # 3. core generation function (fixed cohort logic)
  # ==============================================================================
  generate_intervention_cohort <- function(rep_id, scenario, n_size) {
    
    set.seed(rep_id) 
    
    # 1. prepare parameters
    params <- BASE_ANPP_PARAMS
    params$gamma <- log(2) / scenario$true_gamma_hl
    # log-normal parameter conversion
    params$sigma_c_sq <- log(params$c_var / params$c_mean^2 + 1)
    params$sigma_c <- sqrt(params$sigma_c_sq)
    params$mu_c <- log(params$c_mean) - params$sigma_c_sq / 2
    
    # 2. define lambda function with intervention
    beta_val <- scenario$true_beta
    lambda_func <- function(t) {
      # seasonal baseline
      base <- BETA_0 + 0.5 * sin(OMEGA * t) + 0.2 * cos(OMEGA * t)
      # intervention effect (step function)
      eff <- ifelse(t > T_INT, beta_val, 0)
      return(exp(base + eff))
    }
    
    # 3. generate fixed cohort
    population_pool <- 1:10000
    cohort_ids <- sample(population_pool, n_size, replace = FALSE)
    
    # 4. generate data (4 time points)
    df_list <- list()
    
    # generate slight jitter for time points
    jitter_matrix <- matrix(runif(n_size * length(OFFSETS), -1, 1), 
                            nrow = n_size, ncol = length(OFFSETS))
    
    for(j in seq_along(OFFSETS)) {
      offset <- OFFSETS[j]
      target_time <- T_INT + offset
      actual_times <- target_time + jitter_matrix[, j]
      
      ab_levels <- numeric(n_size)
      for(k in 1:n_size) {
        ab_levels[k] <- sim_individual_ab(
          t_obs = actual_times[k],
          lambda_func = lambda_func,
          params = params,
          latent_id = cohort_ids[k] 
        )
      }
      
      df_list[[j]] <- tibble(
        time = actual_times,
        antibody = ab_levels,
        offset_label = paste0("T+", offset),
        survey_round = j
      )
    }
    
    bind_rows(df_list)
  }
  
  # ==============================================================================
  # 4. run loop
  # ==============================================================================
  cat("starting simulation loop...\n")
  results_container <- list()
  
  for (i in 1:REPS_PER_TASK) {
    current_rep <- START_SEED + i - 1
    
    if (i %% 10 == 0) cat(sprintf("  processing rep %d / %d (global rep %d)\n", 
                                  i, REPS_PER_TASK, current_rep))
    
    # only one scenario, extract directly
    scen <- TEST_SCENARIOS[[1]]
    
    # 1. generate max sample (n=120)
    max_n <- max(SAMPLE_SIZES)
    full_df <- generate_intervention_cohort(rep_id = current_rep, scenario = scen, n_size = max_n)
    
    # 2. slice and save
    for (n in SAMPLE_SIZES) {
      subset_df <- full_df %>%
        group_by(survey_round) %>%
        slice_head(n = n) %>%
        ungroup() %>%
        select(time, antibody, offset_label) # remove id
      
      # naming convention: scenario_rep_n
      key <- paste0(scen$name, "_Rep", current_rep, "_N", n)
      results_container[[key]] <- subset_df
    }
  }
  
  # ==============================================================================
  # 5. save results
  # ==============================================================================
  outfile <- file.path(OUTPUT_DIR, sprintf("sim_int_single_batch_%03d.RData", TASK_ID))
  cat(paste("saving to:", outfile, "\n"))
  save(results_container, file = outfile)
  
  cat("=== script completed successfully ===\n")
  
}, error = function(e) {
  cat(paste("\nfatal error:", e$message, "\n"))
  quit(status = 1)
})