# ==============================================================================
# run_sim_multisite.R
# hipergator simulation script for multi-site cross-sectional model
# ==============================================================================

options(echo = TRUE)

tryCatch({
  cat("=== script start ===\n")
  
  # --- 0. get command line arguments ---
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    TASK_ID <- 1
    cat("warning: no args provided. defaulting to task id 1.\n")
  } else {
    TASK_ID <- as.integer(args[1])
  }
  
  # --- configure parameters ---
  # target: 5000 reps in total
  # strategy: run 100 tasks, each running 50 reps (100 * 50 = 5000)
  REPS_PER_TASK <- 50   
  START_SEED    <- (TASK_ID - 1) * REPS_PER_TASK + 1
  OUTPUT_DIR    <- "sim_multisite_results"
  
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
  # 2. define constants and site metadata
  # ==============================================================================
  # base biological parameters
  BASE_ANPP_PARAMS <- list(
    c_mean = 0.2, c_var = 0.25, gamma_hl = 20, B0 = 0.3, sigma_epsilon = 0.15
  )
  # automatically calculate log-normal parameters
  BASE_ANPP_PARAMS$gamma <- log(2) / BASE_ANPP_PARAMS$gamma_hl
  BASE_ANPP_PARAMS$sigma_c_sq <- log(BASE_ANPP_PARAMS$c_var / BASE_ANPP_PARAMS$c_mean^2 + 1)
  BASE_ANPP_PARAMS$sigma_c <- sqrt(BASE_ANPP_PARAMS$sigma_c_sq)
  BASE_ANPP_PARAMS$mu_c <- log(BASE_ANPP_PARAMS$c_mean) - BASE_ANPP_PARAMS$sigma_c_sq / 2
  
  # seasonality and coefficients
  OMEGA <- 2 * pi / 365
  BETA_0 <- log(1/15)
  
  # covariate coefficients (true values for inference)
  COEFFS <- list(
    spp_aegypti = 0, spp_albo = -0.2,
    clim_tropical = 0.5, clim_arid = -0.6, clim_temperate = 0,
    urban = 0.4, rural = 0
  )
  
  # site definitions (4 typical sites)
  SITE_META <- tibble(
    site_name = c("Site_A", "Site_B", "Site_C", "Site_D"),
    climate = c("Tropical", "Arid", "Tropical", "Temperate"),
    is_urban = c(1, 0, 0, 1), # 1=urban, 0=rural
    species = c("Ae. aegypti", "Ae. albopictus", "Ae. aegypti", "Ae. aegypti")
  )
  
  # sample size list
  SAMPLE_SIZES <- c(15, 70, 120, 300)
  
  # observation time points (assume all sites are observed on day 365)
  OBS_TIME <- 365
  
  # ==============================================================================
  # 3. core generation function
  # ==============================================================================
  generate_multisite_data <- function(rep_id, n_max) {
    set.seed(rep_id) # ensure randomness is independent for each rep
    
    all_sites_list <- list()
    
    for(i in 1:nrow(SITE_META)) {
      site_info <- SITE_META[i,]
      
      # 1. build lambda function for this site
      # calculate fixed linear predictor
      beta_clim <- switch(site_info$climate, 
                          "Tropical" = COEFFS$clim_tropical,
                          "Arid" = COEFFS$clim_arid,
                          "Temperate" = COEFFS$clim_temperate)
      
      beta_spp <- switch(site_info$species,
                         "Ae. aegypti" = COEFFS$spp_aegypti,
                         "Ae. albopictus" = COEFFS$spp_albo)
      
      beta_urb <- ifelse(site_info$is_urban == 1, COEFFS$urban, COEFFS$rural)
      
      # lambda(t)
      lambda_func <- function(t) {
        log_lambda <- BETA_0 + beta_clim + beta_spp + beta_urb + 
          0.5 * sin(OMEGA * t) + 0.2 * cos(OMEGA * t)
        return(exp(log_lambda))
      }
      
      # 2. generate data for n_max individuals
      # note: pure cross-sectional data, individuals at each site are independent, no latent_id needed for cross-site linking
      # but for function call, we still generate random c internally
      
      # generate n_max individuals
      ab_levels <- numeric(n_max)
      
      # add a slight jitter near the observation time point, consistent with real sampling
      actual_times <- OBS_TIME + runif(n_max, -2, 2)
      
      for(k in 1:n_max) {
        # sim_individual_ab will automatically generate random c if latent_id is not passed
        # here we simulate pure cross-section, let the function generate it internally directly
        ab_levels[k] <- sim_individual_ab(
          t_obs = actual_times[k],
          lambda_func = lambda_func,
          params = BASE_ANPP_PARAMS,
          latent_id = NULL # pure cross-sectional, fresh draw for everyone
        )
      }
      
      # 3. organize data for this site
      all_sites_list[[site_info$site_name]] <- tibble(
        site_name = site_info$site_name,
        climate = site_info$climate,
        is_urban = site_info$is_urban,
        species = site_info$species,
        time = actual_times,
        antibody = ab_levels,
        row_id = 1:n_max # used for subsequent slicing
      )
    }
    
    bind_rows(all_sites_list)
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
    
    # 1. generate max sample (n=120)
    max_n <- max(SAMPLE_SIZES)
    full_df <- generate_multisite_data(rep_id = current_rep, n_max = max_n)
    
    # 2. slice and save (for 15, 70, 120)
    for (n in SAMPLE_SIZES) {
      
      # take the first n rows for each site
      subset_df <- full_df %>%
        group_by(site_name) %>%
        filter(row_id <= n) %>%
        ungroup() %>%
        select(-row_id) # remove auxiliary column
      
      # naming convention: rep_n
      key <- paste0("Rep", current_rep, "_N", n)
      results_container[[key]] <- subset_df
    }
  }
  
  # ==============================================================================
  # 5. save results
  # ==============================================================================
  outfile <- file.path(OUTPUT_DIR, sprintf("sim_multisite_batch_%03d.RData", TASK_ID))
  cat(paste("saving to:", outfile, "\n"))
  save(results_container, file = outfile)
  
  cat("=== script completed successfully ===\n")
  
}, error = function(e) {
  cat(paste("\nfatal error:", e$message, "\n"))
  quit(status = 1)
})