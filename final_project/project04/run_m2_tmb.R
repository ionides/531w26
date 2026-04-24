## run_m2_tmb.R — M2: Time-varying b_t, TMB Laplace approximation
## Model: Gompertz + state-space SOI + logit-scale RW on b_t
## The product b_t * X_{t-1} makes Laplace inexact.

library(astsa)
library(TMB)
data(rec); data(soi)
y <- log(as.numeric(rec))
sv <- as.numeric(soi)
TT <- length(y)

## ── TMB template ──
cpp_m2 <- '
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_VECTOR(y);
  DATA_VECTOR(s);

  PARAMETER(a);
  PARAMETER(b_mean);       // logit(b) initial value
  PARAMETER(c_env);
  PARAMETER(phi_E);
  PARAMETER(log_sigma_p);
  PARAMETER(log_sigma_o);
  PARAMETER(log_sigma_E);
  PARAMETER(log_sigma_S);
  PARAMETER(log_sigma_b);

  PARAMETER_VECTOR(x);       // latent recruitment, length T
  PARAMETER_VECTOR(e);       // latent environment, length T
  PARAMETER_VECTOR(logit_b); // logit-scale b_t, length T

  Type sp = exp(log_sigma_p);
  Type so = exp(log_sigma_o);
  Type sE = exp(log_sigma_E);
  Type sS = exp(log_sigma_S);
  Type sb = exp(log_sigma_b);

  int T = y.size();
  Type nll = Type(0.0);

  // --- Environment AR(1) ---
  Type sd_E0 = sE / sqrt(Type(1.0) - phi_E * phi_E);
  nll -= dnorm(e(0), Type(0.0), sd_E0, true);
  for(int t = 1; t < T; t++)
    nll -= dnorm(e(t), phi_E * e(t-1), sE, true);
  for(int t = 0; t < T; t++)
    nll -= dnorm(s(t), e(t), sS, true);

  // --- logit(b_t) random walk ---
  nll -= dnorm(logit_b(0), b_mean, sb, true);
  for(int t = 1; t < T; t++)
    nll -= dnorm(logit_b(t), logit_b(t-1), sb, true);

  // --- Recruitment with time-varying b_t ---
  Type b0 = Type(2.0) / (Type(1.0) + exp(-b_mean)) - Type(1.0);
  Type mu0 = a / (Type(1.0) - b0);
  Type sd0 = sp / sqrt(Type(1.0) - b0 * b0);
  nll -= dnorm(x(0), mu0, sd0, true);
  for(int t = 1; t < T; t++){
    Type bt = Type(2.0) / (Type(1.0) + exp(-logit_b(t))) - Type(1.0);
    nll -= dnorm(x(t), a + bt * x(t-1) + c_env * e(t-1), sp, true);
  }
  for(int t = 0; t < T; t++)
    nll -= dnorm(y(t), x(t), so, true);

  ADREPORT(sp); ADREPORT(so); ADREPORT(sE); ADREPORT(sS); ADREPORT(sb);
  return nll;
}
'

writeLines(cpp_m2, "m2_tvb.cpp")
compile("m2_tvb.cpp")
dyn.load(dynlib("m2_tvb"))

## ── Multi-start optimization ──
set.seed(531)
dat <- list(y = y, s = sv)
ns <- 20
fits <- list()

cat("=== M2 TMB: time-varying b_t ===\n")
cat(sprintf("T = %d, %d random starts\n", TT, ns))

for(i in 1:ns) {
  b_mean_init <- qlogis((runif(1, 0.5, 0.95) + 1) / 2)
  p0 <- list(
    a = runif(1, -0.5, 1),
    b_mean = b_mean_init,
    c_env = runif(1, -0.2, 0.2),
    phi_E = runif(1, 0.3, 0.95),
    log_sigma_p = runif(1, -2, 0),
    log_sigma_o = runif(1, -3, -1),
    log_sigma_E = runif(1, -2, 0),
    log_sigma_S = runif(1, -2, 0),
    log_sigma_b = log(0.01),  # FIXED at 0.01
    x = y + rnorm(TT, 0, 0.1),
    e = sv + rnorm(TT, 0, 0.1),
    logit_b = rep(b_mean_init, TT) + rnorm(TT, 0, 0.05)
  )

  obj <- tryCatch(
    MakeADFun(dat, p0, random = c("x", "e", "logit_b"),
              map = list(log_sigma_b = factor(NA)),  # fix sigma_b
              DLL = "m2_tvb", silent = TRUE),
    error = function(e) NULL
  )
  if(is.null(obj)) next

  ## Lower bounds: log_sigma_o >= log(0.05) = -3.0
  lo <- c(-5, -5, -1, -0.999, -5, -3.0, -5, -5)
  hi <- c( 5,  5,  1,  0.999,  3,  3.0,  3,  3)

  opt <- tryCatch(
    nlminb(obj$par, obj$fn, obj$gr, lower=lo, upper=hi,
           control = list(eval.max = 2000, iter.max = 1000)),
    error = function(e) list(convergence = 99)
  )

  if(opt$convergence == 0) {
    fits[[length(fits) + 1]] <- list(par = opt$par, nll = opt$objective, obj = obj, opt = opt)
    cat(sprintf("  Start %2d: nll = %.2f (converged)\n", i, opt$objective))
  } else {
    cat(sprintf("  Start %2d: failed\n", i))
  }
}

cat(sprintf("\n%d / %d converged\n", length(fits), ns))

## ── Best fit ──
nlls <- sapply(fits, "[[", "nll")
best_idx <- which.min(nlls)
best <- fits[[best_idx]]

cat(sprintf("\nBest TMB loglik: %.4f\n", -best$nll))

## Extract parameters
sr <- tryCatch(sdreport(best$obj), error = function(e) NULL)
if(!is.null(sr)) {
  fe <- summary(sr, "fixed")
  cat("\nFixed effects:\n")
  print(round(fe, 4))

  ## Extract b_t trajectory
  re <- summary(sr, "random")
  logit_b_hat <- re[(2*TT+1):(3*TT), "Estimate"]
  b_hat <- 2 / (1 + exp(-logit_b_hat)) - 1

  cat(sprintf("\nb_t range: [%.4f, %.4f]\n", min(b_hat), max(b_hat)))
  cat(sprintf("b_t mean:  %.4f\n", mean(b_hat)))
  cat(sprintf("sigma_b:   %.4f\n", exp(fe["log_sigma_b", "Estimate"])))
  cat(sprintf("sigma_o:   %.4f\n", exp(fe["log_sigma_o", "Estimate"])))
  cat(sprintf("c_env:     %.4f\n", fe["c_env", "Estimate"]))
}

## ── Save results ──
save(fits, best, file = "m2_tmb_results.RData")
cat("\nSaved to m2_tmb_results.RData\n")
