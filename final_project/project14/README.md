## Review comments on Final Project 14, STATS 531 W26

The goal, to compare school-term seasonality with a more flexible alternative for measles transmission, is well-motivated and explained. 

1. Fig 3 looks compelling. The Fourier basis is trying to fit the termtime, but it is too smooth to do this accurately. If this lack of smoothness really is worth 25 units of log-likelihood (i.e., if the model coding and optimzation are correct) then this is a nice discovery. However, there is a substantial flaw in the reasoning.
The code in `targeted_if2.py` shows that the search for the Fourier basis does not re-estimate other parameters; only the basis coefficients have positive random walk std dev. That could explain the results. A model comparison based on AIC or log-likelihood is only valid if both models are evaluated at their respective MLE. The authors likely failed to explore the region of the parameter space where the Fourier basis might actually excel. 
The authors were well aware of this issue, and Sec 5.4 explains the justification for fixing to avoid numerical problems:
"A full joint MLE would require addressing the particle collapse problem by other
means — particle MCMC or profile likelihood — and could potentially narrow or widen this gap."
The full MLE could only narrow this gap, mathematically.
The problem maximizing the full MLE was probably not to do with the method but the implementation.
At a minimum, starting at the constrained MLE and then estimating more parameters could only improve the log-likelihood, and in this region the authors report no filtering problems.

1. Non-matching reference: Tan, Ionides & Hooker (2024) is not a reference for JAX.

1. The project is thin on diagnostics. The multiple start approach looks reasonable, but how could we tell if some of the algorithmic parameters are set poorly?

1. Reproducibility is weak. The qmd file has hard-coded numbers. It is akwward to track down where these come from. This can cause problems. For example, `precision_eval.py` hardcodes parameter vectors
and log-likelihoods from `last_attempt.py` directly as Python lists rather than loading them from the saved JSON output. If any upstream script is re-run, the downstream script will silently use stale values without warning. Better to put everything in the qmd file, and use explicit caching to control re-evaluation.

1. Likelihood profiles would be more informative than the presented slices.

1. The global search uses only 10 random starts, which might not be enough for a 6-dimensional
parameter space. A global search should use enough starts that multiple independent runs
converge to the same region, to verify that the global optimum has been reached rather than
a local one. Furthermore, the optimization table in the supplementary material reports only
the single best log-likelihood from the global search, without showing how the other 9 starts
performed. Without the full information regarding each start, it is difficult to assess how
sensitive the results are to initialization, whether all starts converged to a similar value or
whether the best one happened to be an outlier.

1. The Fourier basis model lacks an intercept term, meaning it cannot independently control the mean level of seasonal transmission. This is not a major problem because the scale is determined by R0. The school term transmission in model A averages to 1. The exponentiated Fourier basis will average to something close to 1. Since R0 is not separately estimated, there will be a small problem that could explain the observed log-likelihood discrepancy. 
