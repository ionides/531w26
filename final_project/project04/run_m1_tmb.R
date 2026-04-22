## run_m1_tmb.R — M1 TMB + KF on simulated data
## Reads m1_simulated_data.csv (from python fish_pomp_m1.py)

library(TMB)

## ── Load simulated data ──
if(!file.exists("m1_simulated_data.csv"))
  stop("Run 'python fish_pomp_m1.py' first to generate m1_simulated_data.csv")

dat_df <- read.csv("m1_simulated_data.csv")
Y_obs <- dat_df$Y
S_obs <- dat_df$S
TT <- nrow(dat_df)

## True parameters
a_true <- 0.3; b_true <- 0.9; c_true <- -0.05; phi_true <- 0.7
sp_true <- 0.25; so_true <- 0.15; sE_true <- 0.25; sS_true <- 0.15

cat(sprintf("=== M1: T=%d simulated observations ===\n\n", TT))

## ── TMB template ──
cpp_m1 <- '
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_VECTOR(y); DATA_VECTOR(s);
  PARAMETER(a); PARAMETER(b); PARAMETER(c_env); PARAMETER(phi_E);
  PARAMETER(log_sigma_p); PARAMETER(log_sigma_o);
  PARAMETER(log_sigma_E); PARAMETER(log_sigma_S);
  PARAMETER_VECTOR(x); PARAMETER_VECTOR(e);
  Type sp=exp(log_sigma_p), so=exp(log_sigma_o);
  Type sE=exp(log_sigma_E), sS=exp(log_sigma_S);
  int T=y.size(); Type nll=Type(0.0);
  nll -= dnorm(e(0), Type(0.0), sE/sqrt(Type(1.0)-phi_E*phi_E), true);
  for(int t=1;t<T;t++) nll -= dnorm(e(t), phi_E*e(t-1), sE, true);
  for(int t=0;t<T;t++) nll -= dnorm(s(t), e(t), sS, true);
  nll -= dnorm(x(0), a/(Type(1.0)-b), sp/sqrt(Type(1.0)-b*b), true);
  for(int t=1;t<T;t++) nll -= dnorm(x(t), a+b*x(t-1)+c_env*e(t-1), sp, true);
  for(int t=0;t<T;t++) nll -= dnorm(y(t), x(t), so, true);
  ADREPORT(sp); ADREPORT(so); ADREPORT(sE); ADREPORT(sS);
  return nll;
}
'
writeLines(cpp_m1, "m1_tmb.cpp")
compile("m1_tmb.cpp"); dyn.load(dynlib("m1_tmb"))

## ── KF exact loglik ──
m1_kf <- function(theta) {
  a<-theta[1]; b<-theta[2]; cc<-theta[3]; phi<-theta[4]
  sp<-exp(theta[5]); so<-exp(theta[6]); sE<-exp(theta[7]); sS<-exp(theta[8])
  if(abs(b)>=1||abs(phi)>=1) return(-1e10)
  d<-c(a,0); Fm<-matrix(c(b,cc,0,phi),2,2,byrow=TRUE)
  Q<-diag(c(sp^2,sE^2)); H<-diag(2); R<-diag(c(so^2,sS^2))
  mu<-solve(diag(2)-Fm)%*%d
  P<-matrix(solve(diag(4)-kronecker(Fm,Fm))%*%as.vector(Q),2,2)
  ll<-0
  for(t in 1:TT){
    mp<-d+Fm%*%mu; Pp<-Fm%*%P%*%t(Fm)+Q
    v<-c(Y_obs[t],S_obs[t])-H%*%mp; S<-H%*%Pp%*%t(H)+R
    ds<-det(S); if(ds<=0) return(-1e10)
    ll<-ll-0.5*(2*log(2*pi)+log(ds)+t(v)%*%solve(S)%*%v)
    K<-Pp%*%t(H)%*%solve(S); mu<-mp+K%*%v; P<-(diag(2)-K%*%H)%*%Pp
  }
  as.numeric(ll)
}

## KF at true params
theta_true <- c(a_true, b_true, c_true, phi_true,
                log(sp_true), log(so_true), log(sE_true), log(sS_true))
ll_kf_true <- m1_kf(theta_true)
cat(sprintf("KF at true params:  %.4f\n", ll_kf_true))

## KF MLE
kf_opt <- optim(theta_true, function(th) -m1_kf(th), method="L-BFGS-B",
                lower=c(-5,-0.999,-1,-0.999,-5,-5,-5,-5),
                upper=c(5,0.999,1,0.999,3,3,3,3), control=list(maxit=1000))
ll_kf_mle <- -kf_opt$value
cat(sprintf("KF MLE:             %.4f\n", ll_kf_mle))

## ── TMB multi-start ──
set.seed(531)
dat <- list(y=Y_obs, s=S_obs)
ns <- 20; fits <- list()
lo <- c(-5,-0.999,-1,-0.999,-5,-5,-5,-5)
hi <- c(5,0.999,1,0.999,3,3,3,3)

for(i in 1:ns){
  p0 <- list(a=runif(1,-1,2), b=runif(1,0.2,0.99), c_env=runif(1,-0.3,0.3),
             phi_E=runif(1,0.3,0.99), log_sigma_p=runif(1,-2,0.5),
             log_sigma_o=runif(1,-2,0.5), log_sigma_E=runif(1,-2,0.5),
             log_sigma_S=runif(1,-2,0.5),
             x=Y_obs+rnorm(TT,0,0.1), e=S_obs+rnorm(TT,0,0.1))
  obj <- tryCatch(MakeADFun(dat,p0,random=c("x","e"),DLL="m1_tmb",silent=TRUE),
                  error=function(e) NULL)
  if(is.null(obj)) next
  opt <- tryCatch(nlminb(obj$par,obj$fn,obj$gr,lower=lo,upper=hi,
    control=list(eval.max=1000,iter.max=500)),error=function(e) list(convergence=99))
  if(opt$convergence==0) fits[[length(fits)+1]] <- list(par=opt$par,nll=opt$objective)
}

best <- fits[[which.min(sapply(fits,"[[","nll"))]]
## Refit
p_best <- list(a=best$par[1],b=best$par[2],c_env=best$par[3],phi_E=best$par[4],
               log_sigma_p=best$par[5],log_sigma_o=best$par[6],
               log_sigma_E=best$par[7],log_sigma_S=best$par[8],
               x=Y_obs,e=S_obs)
o_best <- MakeADFun(dat,p_best,random=c("x","e"),DLL="m1_tmb",silent=TRUE)
opt_best <- nlminb(o_best$par,o_best$fn,o_best$gr,lower=lo,upper=hi)
fe <- summary(sdreport(o_best),"fixed")
ll_tmb <- -opt_best$objective

cat(sprintf("TMB MLE:            %.4f\n", ll_tmb))

## KF at TMB params
ll_kf_at_tmb <- m1_kf(fe[,"Estimate"])
cat(sprintf("KF at TMB params:   %.4f\n", ll_kf_at_tmb))

## ── Load IF2 results if available ──
if(file.exists("m1_if2_results.json")){
  library(jsonlite)
  j <- fromJSON("m1_if2_results.json")
  b <- j$best
  if2_theta <- c(b$a, b$b, b$c_env, b$phi_E,
                 b$log_sigma_p, b$log_sigma_o, b$log_sigma_E, b$log_sigma_S)
  ll_kf_at_if2 <- m1_kf(if2_theta)

  cat(sprintf("\nIF2 best loglik (PF): %.4f\n", b$mean_ll))
  cat(sprintf("KF at IF2 params:     %.4f\n", ll_kf_at_if2))

  ## TMB at IF2 params
  p_if2 <- list(a=if2_theta[1], b=if2_theta[2], c_env=if2_theta[3], phi_E=if2_theta[4],
                log_sigma_p=if2_theta[5], log_sigma_o=if2_theta[6],
                log_sigma_E=if2_theta[7], log_sigma_S=if2_theta[8],
                x=Y_obs, e=S_obs)
  o_if2 <- MakeADFun(dat,p_if2,random=c("x","e"),DLL="m1_tmb",silent=TRUE)
  ll_tmb_at_if2 <- -o_if2$fn(o_if2$par)
  cat(sprintf("TMB at IF2 params:    %.4f\n", ll_tmb_at_if2))
}

## ── Parameter comparison ──
cat("\n=== Parameter Comparison ===\n")
tmb_est <- c(fe[1:4,1], exp(fe[5:8,1]))
kf_est <- c(kf_opt$par[1:4], exp(kf_opt$par[5:8]))
true_vals <- c(a_true, b_true, c_true, phi_true, sp_true, so_true, sE_true, sS_true)
pnames <- c("a","b","c","phi","sigma_p","sigma_o","sigma_E","sigma_S")

tab <- data.frame(Parameter=pnames, Truth=round(true_vals,4),
                  TMB=round(tmb_est,4), KF_MLE=round(kf_est,4))

if(file.exists("m1_if2_results.json")){
  if2_est <- c(b$a, b$b, b$c_env, b$phi_E,
               exp(b$log_sigma_p), exp(b$log_sigma_o),
               exp(b$log_sigma_E), exp(b$log_sigma_S))
  tab$IF2 <- round(if2_est, 4)
}
print(tab)

## ── Cross-validation table ──
cat("\n=== Cross-Validation (loglik) ===\n")
cat(sprintf("                at TMB params  at IF2 params  at true params\n"))
cat(sprintf("KF (exact)      %12.2f  %12.2f  %12.2f\n", ll_kf_at_tmb,
    ifelse(exists("ll_kf_at_if2"), ll_kf_at_if2, NA), ll_kf_true))
cat(sprintf("TMB (Laplace)   %12.2f  %12.2f          ---\n", ll_tmb,
    ifelse(exists("ll_tmb_at_if2"), ll_tmb_at_if2, NA)))
cat(sprintf("PF (IF2 best)          ---  %12.2f          ---\n",
    ifelse(exists("b"), b$mean_ll, NA)))

cat("\nFor linear-Gaussian: TMB = KF (both exact).\n")
cat("PF should match KF within noise.\n")

save(fits, best, fe, ll_tmb, ll_kf_true, ll_kf_mle, file="m1_tmb_results.RData")
cat("\nSaved to m1_tmb_results.RData\n")
