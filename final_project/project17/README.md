## Review comments on Final Project 17, STATS 531 W26

The choice of polio over the 1930–1964 window is well motivated: the 1955 vaccine introduction provides a clear regime change that adds scientifically interesting statistical challenges, and distinguishes this project from previous epidemiological time-series projects.

1. There are various concerns with submitting a large and complex piece of AI-generated code. `blinded.qmd` contains about 2000 lines of AI-generated code that largely duplicate the capabilities of Pypomp, with certain limitations: (i) no quality control tests; (ii) humans cannot readily check validity of large complex codes just by inspection; (iii) no access to JAX for compilation and GPU; (iv) there are ad-hoc decisions made with `N_EFF`, `MAX_NU`, `MAX_LATENT_I`, `MAX_POISSON_LAM` that have unknown and untested effects. **This use of AI, effectively bypassing many of the things covered in the course, does not meet expectations for the course.**

1. TSIR, as defined in the reference provided, involves susceptible reconstruction. It looks like, here, we are closer to a POMP formulation. The term TSIR-POMP is used but what this means is not explained. 
The exact differences between TSIR and POMP, and how and why they were comgined, is not well described.
The purpose of the TSIR model (Finkenstadt & Grenfell, 2000) is to avoid a particle filter by making a susceptible reconstruction. Thus, it is not clear how the model developed in this project relates to the reference.

1. The authors report that the change in transmission due to the vaccine is not detectable in their model. While the report makes some progress (which is all that is expected in a 531 final project) many things could be refined in future analysis. For example, the model includes vaccination as a reduced transmission rate, but it might fit better as a reduced susceptible population.

1. It is curious that the mechanistic model wins on everything but the primary outcome. Another referee found a reason for this.


1. Point-prediction comparison is unfair to SARIMAX. The MAE, RMSE, and peak-error metrics for the POMP are computed against filtered mean log and filtered mean raw, which are $\hat E[I_t | Y_{1:t}=y^*_{1:t}]$. 
These are formed using $y^*_t$ itself, so $y^*_t −  E[ \log (1 + I_t) | Y_{1:t}=y^*_{1:t}]$ is approximately the observation noise rather than a forecast error.
SARIMAX results.fittedvalues, by contrast, returns one-step-ahead predictions $E[Y_t | Y_{1:t-1}=y^*_{1:t−1}]$. 
The peak-timing claim (“a week vs. a month for SARIMAX”) is a direct consequence of this asymmetry: the filtered mean tracks the observed
series almost exactly, so its annual argmax must agree with the observed peak. 
A fair POMP analogue is the predicted mean $E[I_t | Y_{1:t}=y^*_{1:t−1}]$, available from the propagated particles before the
time-t weight update.

1. The $\delta_\mathrm{post}$ profile likelihood was not run. The trade-off between $\delta_\mathrm{post}$ and $\sigma_A$ is a central
scientific question. The script `poliopomp_robust.py` already contains a profile delta post
function and a `--run-profile` flag, so running the profile is one command-line change and a few
extra cluster hours. Its absence leaves the main question — can the data tell apart a vaccine-era
regime shift from a run of negative annual shocks? — unanswered.


1. The effective population size limit of $2\times 10^5$ is much lower than the population modeled (the entire USA). 

1. The log-likelihood comparison between SARIMAX and POMP is not valid. The selected SARIMAX model uses $d = 1$ and $D = 1$ with seasonal period $m = 12$, so statsmodels computes its log-likelihood on the doubly-differenced series using approximately 407 effective observations rather than all 420. The POMP marginal log-likelihood covers all $T = 420$ observations from the start of the series. These quantities cannot be directly subtracted, and the conclusion in Section 6 that “SARIMAX achieved a better log-likelihood” is not meaningful as stated.

1. The simulation-based model check described in Section 3.2.7 never appears in the results.
The methods explicitly promise 200 forward sample paths with a pointwise 95% envelope.
What appears in Section 5.5 is the filtered mean trajectory, which is the posterior mean conditioned on all observations and will always appear smoother and closer to the data than honest forward simulation.
The distinction matters especially for the post-1955 period.

1. The POMP results are hard-coded into the qmd document with no random seed documented. It would be more transparent to use the qmd with cache workflow developed in class. Also, there is a path mismatch between
`aggregate.py` (which saves output to the current directory) and `sarima_search.py` (which reads from a `data/` subdirectory), causing an immediate reproducibility failure without a manual fix.

1. No information criterion is reported for the TSIR-POMP model. The SARIMAX
benchmark was selected by BIC (reported as 401.82), but no AIC or BIC is computed for the
TSIR-POMP. From the reported values — log-likelihood −226.33 and 13 free parameters — an
approximate AIC for the TSIR-POMP works out to −2 × (−226.33) + 2 × 13 $\approx$ 479, compared to
roughly −2 × (−168.25) + 2 × 11 $\approx$ 359 for the SARIMAX. Including this in the comparison table would help readers.

1. Parameter estimates are reported without uncertainty. All 13 parameters are reported
as point estimates only. No confidence intervals, profile likelihoods, or standard errors are given for any parameter
