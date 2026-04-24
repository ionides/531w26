## M2 Kalman Filter — exact marginal log-likelihood
## No TMB, no PF, no approximation.
## Run: Rscript verify_m2_kf.R

library(astsa)
data(rec); data(soi)
y <- log(as.numeric(rec))
s <- as.numeric(soi)
TT <- length(y)

## ============================================================
## Kalman filter for M2 bivariate state-space model
##
##   State: z_t = (X_t, E_t)'
##   Obs:   w_t = (Y_t, S_t)'
##
##   z_t = d + F * z_{t-1} + process noise ~ N(0, Q)
##   w_t = H * z_t + obs noise ~ N(0, R)
## ============================================================

m2_loglik <- function(theta) {
  a     <- theta[1]
  b     <- theta[2]
  c_env <- theta[3]
  phi_E <- theta[4]
  sp    <- exp(theta[5])  # sigma_p
  so    <- exp(theta[6])  # sigma_o
  sE    <- exp(theta[7])  # sigma_E
  sS    <- exp(theta[8])  # sigma_S

  ## If |b| >= 1 or |phi_E| >= 1, return -Inf
  if (abs(b) >= 1 || abs(phi_E) >= 1) return(-1e10)

  ## System matrices
  d <- c(a, 0)                       # intercept
  Fm <- matrix(c(b, c_env, 0, phi_E), 2, 2, byrow = TRUE)  # transition
  ## F = [ b      c_env ]
  ##     [ 0      phi_E ]
  ## Row 1: X_t = a + b*X_{t-1} + c*E_{t-1}
  ## Row 2: E_t = 0*X_{t-1} + phi*E_{t-1}
  Q <- diag(c(sp^2, sE^2))          # process cov
  H <- diag(2)                       # observation: w = H * z + noise
  R <- diag(c(so^2, sS^2))          # observation cov

  ## Initial state: stationary distribution
  mu0 <- solve(diag(2) - Fm) %*% d
  ## Stationary covariance: vec(P0) = (I - F⊗F)^{-1} vec(Q)
  P0 <- matrix(solve(diag(4) - kronecker(Fm, Fm)) %*% as.vector(Q), 2, 2)

  ## Kalman filter
  mu <- mu0
  P  <- P0
  ll <- 0

  for (t in 1:TT) {
    ## Predict
    mu_pred <- d + Fm %*% mu
    P_pred  <- Fm %*% P %*% t(Fm) + Q

    ## Innovation
    w <- c(y[t], s[t])
    v <- w - H %*% mu_pred
    S_innov <- H %*% P_pred %*% t(H) + R

    ## Log-likelihood contribution
    det_S <- det(S_innov)
    if (det_S <= 0) return(-1e10)
    S_inv <- solve(S_innov)
    ll <- ll - 0.5 * (2 * log(2*pi) + log(det_S) + t(v) %*% S_inv %*% v)

    ## Update
    K  <- P_pred %*% t(H) %*% S_inv
    mu <- mu_pred + K %*% v
    P  <- (diag(2) - K %*% H) %*% P_pred
  }
  return(as.numeric(ll))
}

## ============================================================
## Evaluate at TMB MLE and IF2 MLE
## ============================================================

cat("=== Exact Kalman Filter Log-Likelihood for M2 ===\n\n")

## IF2 best (from HW8 start #6)
theta_if2 <- c(
  a = 0.0527, b = 0.9743, c_env = 0.0422, phi_E = 0.6778,
  log_sigma_p = log(0.2516), log_sigma_o = log(0.0729),
  log_sigma_E = log(0.2247), log_sigma_S = log(0.1840)
)
ll_if2 <- m2_loglik(theta_if2)
cat("IF2 MLE:  loglik =", round(ll_if2, 4), "\n")

## TMB MLE (from rendered PDF — update these with your actual TMB values)
theta_tmb <- c(
  a = 0.2810, b = 0.9289, c_env = -0.0069, phi_E = 0.6420,
  log_sigma_p = log(0.2495), log_sigma_o = log(0.0183),
  log_sigma_E = log(0.2941), log_sigma_S = log(0.0719)
)
ll_tmb <- m2_loglik(theta_tmb)
cat("TMB MLE:  loglik =", round(ll_tmb, 4), "\n")

cat("\nDifference (TMB - IF2):", round(ll_tmb - ll_if2, 4), "\n")

## ============================================================
## Optimize via Kalman filter (ground truth MLE)
## ============================================================
cat("\n=== Optimizing via Kalman filter (no approximation) ===\n")

## Try from IF2 starting point
opt1 <- optim(theta_if2, function(th) -m2_loglik(th),
              method = "L-BFGS-B",
              lower = c(-5, -0.999, -1, -0.999, -5, -5, -5, -5),
              upper = c( 5,  0.999,  1,  0.999,  3,  3,  3,  3),
              control = list(maxit = 1000))

cat("\nKF-MLE (unconstrained sigma_o):\n")
cat("  loglik =", round(-opt1$value, 4), "\n")
cat("  a =", round(opt1$par[1], 4), "\n")
cat("  b =", round(opt1$par[2], 4), "\n")
cat("  c =", round(opt1$par[3], 4), "\n")
cat("  phi =", round(opt1$par[4], 4), "\n")
cat("  sigma_p =", round(exp(opt1$par[5]), 4), "\n")
cat("  sigma_o =", round(exp(opt1$par[6]), 4), "\n")
cat("  sigma_E =", round(exp(opt1$par[7]), 4), "\n")
cat("  sigma_S =", round(exp(opt1$par[8]), 4), "\n")

## Also optimize with sigma_o >= 0.05 to match IF2 regime
opt2 <- optim(theta_if2, function(th) -m2_loglik(th),
              method = "L-BFGS-B",
              lower = c(-5, -0.999, -1, -0.999, -5, log(0.05), -5, -5),
              upper = c( 5,  0.999,  1,  0.999,  3,  3,  3,  3),
              control = list(maxit = 1000))

cat("\nKF-MLE (sigma_o >= 0.05):\n")
cat("  loglik =", round(-opt2$value, 4), "\n")
cat("  a =", round(opt2$par[1], 4), "\n")
cat("  b =", round(opt2$par[2], 4), "\n")
cat("  c =", round(opt2$par[3], 4), "\n")
cat("  phi =", round(opt2$par[4], 4), "\n")
cat("  sigma_p =", round(exp(opt2$par[5]), 4), "\n")
cat("  sigma_o =", round(exp(opt2$par[6]), 4), "\n")
cat("  sigma_E =", round(exp(opt2$par[7]), 4), "\n")
cat("  sigma_S =", round(exp(opt2$par[8]), 4), "\n")

cat("\n=== Summary ===\n")
cat("Unconstrained MLE loglik:", round(-opt1$value, 2), "\n")
cat("Constrained (sigma_o>=0.05) loglik:", round(-opt2$value, 2), "\n")
cat("IF2 loglik:", round(ll_if2, 2), "\n")
cat("TMB loglik:", round(ll_tmb, 2), "\n")
