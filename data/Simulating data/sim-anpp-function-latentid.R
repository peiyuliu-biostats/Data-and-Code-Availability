# sim-anpp-function-latentid.R

library(tidyverse)

set.seed(42)

 
#' simulate individual antibody level
#' 
#' @param t_obs        calendar time of observation (days)
#' @param lambda_func  function(t) returning biting rate lambda(t)
#' @param age_at_obs   age of the individual at observation (days)
#' @param params       parameter list
#' @param baseline_mod modifier for baseline antibody (useful for age accumulation)
#' @param latent_id    (optional) integer ID to fix 'C' for this person across time.
sim_individual_ab <- function(t_obs, lambda_func, age_at_obs = NULL, params, baseline_mod = 0, latent_id = NULL) {
  
  # 1. define simulation window (burn-in 3 yrs or ages)
  if (!is.null(age_at_obs)) {
    t_start <- t_obs - age_at_obs 
  } else {
    t_start <- t_obs - (365 * 3)  
  }
  t_end <- t_obs
  
  # 2. sample individual-specific boost magnitude (C)
  # compliance with Assumption (iii): C is constant within an individual.
  
  if (!is.null(latent_id)) {
    # save current random state to avoid messing up other random processes (like bite times)
    if (exists(".Random.seed", envir = .GlobalEnv)) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv)
    } else {
      runif(1) # initialize seed if not exists
      old_seed <- get(".Random.seed", envir = .GlobalEnv)
    }
    
    # use latent_id to seed the random number generator *only* for C.
    # the same person (latent_id) always gets the exact same C value.
    set.seed(latent_id)
    c_individual <- rlnorm(1, meanlog = params$mu_c, sdlog = params$sigma_c)
    
    # restore random state so subsequent bite times remain stochastic
    assign(".Random.seed", old_seed, envir = .GlobalEnv)
    
  } else {
    # if no ID provided (pure random cross-section), draw a fresh C
    c_individual <- rlnorm(1, meanlog = params$mu_c, sdlog = params$sigma_c)
  }
  
  # 3. generate nhpp biting times (thinning algorithm)
  check_pts <- seq(t_start, t_end, length.out = 50)
  lambda_vals <- lambda_func(check_pts)
  lambda_max <- max(lambda_vals) * 1.5 
  if(lambda_max < 1e-6) lambda_max <- 0.01 
  
  duration <- t_end - t_start
  n_expected <- lambda_max * duration
  n_candidate <- rpois(1, n_expected)
  
  if (n_candidate > 0) {
    candidate_times <- sort(runif(n_candidate, t_start, t_end))
    accept_probs <- lambda_func(candidate_times) / lambda_max
    actual_bite_times <- candidate_times[runif(n_candidate) < accept_probs]
  } else {
    actual_bite_times <- numeric(0)
  }
  
  # 4. calculate antibody convolution
  current_b0 <- params$B0 + baseline_mod
  
  if (length(actual_bite_times) == 0) {
    ab_val <- current_b0
  } else {
    # use the fixed 'c_individual' for all bites of this person
    decays <- exp(-params$gamma * (t_end - actual_bite_times))
    ab_val <- c_individual * sum(decays) + current_b0
  }
  
  # 5. add measurement error
  ab_obs <- ab_val + rnorm(1, 0, params$sigma_epsilon)
  return(max(0.001, ab_obs)) 
}

 
