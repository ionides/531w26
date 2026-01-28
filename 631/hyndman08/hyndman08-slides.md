---
title: "Hyndman & Khandakar, 2008"
date: "Jan 28, 2026"
output:
  ioslides_presentation:
    smaller: no
    widescreen: true
    transition: "faster" 
---

## Overview

* Cited over 5000 times on Google Scholar

* Motivation for auto.arima: thousands of time series to analyze

* Practical usage: inexperiences time series users just apply it blindly, and call it the "best" model because it is selected by this algorithm.

## Exponential smoothing (Sec. 2)

* A standard method in economic forecasting and business

* May seem ad-hoc until embedded in a dynamic model, i.e., a state space model (SSM) representation, also known as a partially observed Markov process (POMP) or a hidden Markov model (HMM).

* Eqs (2,3,4) violate the usual POMP assumption that Y_n is conditionally independent of all other state and observation variables, given X_n.


##  Initial values (Sec. 2.4)

* Note that initial values have to be estimated,

* "Most implementations of exponential smoothing use an ad hoc heuristic scheme to estimate $x_0$. However, with modern computers, there is no reason why we cannot estimate $x_0$ along with $\theta$, and the resulting forecasts are often substantially better when we do."

## Why AIC? (Sec. 2.5)

* "Obviously, other model selection criteria (such as the BIC) could also be used in a similar manner."

* Is there anything special about AIC? Not for consistent model selection.

## When does model-based fitting by AIC beat alternatives?

* Competition results: "The methodology is particularly good at short term forecasts (up to about 6 periods ahead), and especially for seasonal short-term series (beating all other methods in the competitions for these series)"

## On diffuse priors (Sec. 3.1)

* Claim that the likelihood is not defined for nonzero differencing parameters, d and D.

* `arima` attempts to solve this using so-called diffuse priors (Durbin & Koopman, 2001).

* HK prefer unit root tests.

* HK note that differencing may produce poor forecasts, but there's no option for a nonlinear trend with ARMA errors .


##  Stepwise selection (Sec. 3.2)

* Compares AIC to neighboring models, with +/- 1 lags for one or two model components.

* Reject models close to a unit root or having numerical errors

## Comparing exponential smoothing to ARIMA (Sec. 3.3)

* The larger class does not always do better, particularly for seasonal models (there are dauntingly many choices for SARIMA).

## Software engineering (Sec. 4)

* S3 classes

* Presumably, high usage reflects good engineering

* Success in competitions can demonstrate strong methodology





