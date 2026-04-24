## m2_cross_tmb_at_if2.R — Evaluate TMB Laplace loglik at IF2 parameters

library(astsa); library(TMB)
data(rec); data(soi)
y <- log(as.numeric(rec)); sv <- as.numeric(soi); TT <- length(y)

## Load TMB DLL (must have run run_m2_tmb.R first)
if(!is.loaded("m2_tvb")) dyn.load(dynlib("m2_tvb"))

## IF2 best params (Deepan Run 1, chain 9)
## b_mean on natural scale = 0.9215 → logit scale = log((1+b)/(1-b))
b_if2 <- 0.9215
b_mean_logit <- log((1 + b_if2) / (1 - b_if2))

p_if2 <- list(
  a = 0.2699,
  b_mean = b_mean_logit,
  c_env = -0.0035,
  phi_E = 0.6751,
  log_sigma_p = -1.3188,
  log_sigma_o = -2.8838,
  log_sigma_E = -1.3374,
  log_sigma_S = -2.1465,
  log_sigma_b = log(0.01),   # fixed
  x = y, e = sv,
  logit_b = rep(b_mean_logit, TT)
)

obj <- MakeADFun(
  data = list(y = y, s = sv),
  parameters = p_if2,
  random = c("x", "e", "logit_b"),
  map = list(log_sigma_b = factor(NA)),
  DLL = "m2_tvb",
  silent = TRUE
)

## TMB integrates out random effects via Laplace at these fixed params
tmb_nll_at_if2 <- obj$fn(obj$par)

cat("=== TMB Laplace loglik at IF2 params ===\n")
cat(sprintf("  TMB loglik at IF2:  %.4f\n", -tmb_nll_at_if2))
cat(sprintf("  TMB loglik at TMB:  %.4f\n", -131.5559))
cat(sprintf("  PF loglik at IF2:   ~-160.3 (Deepan)\n"))
cat(sprintf("\n  Laplace error at IF2 params: %.1f units\n",
            -tmb_nll_at_if2 - (-160.3)))
