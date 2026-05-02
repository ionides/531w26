## Review comments on Final Project 6, STATS 531 W26

A strength is the integration of the Parkinson high-low range proxy alongside standard close-to-close returns, providing a measurement model with a more granular intraday volatility signal than daily returns alone. This is innovative in the context of previous 531 projects.

1. Most code is reproducible within the qmd, though some main-text numbers are hard-coded.

1. Fig 4 shows major convergence problems; at least, one run fails.
The leverage convergence plot looks like a coding problem, not weak identification.
The figure shows log-likelihood traces on a scale of $10^8$, with values around $−10^8$. The final
model logs likelihoods around 7,300. That is indicative of 
like a numerical overflow or an incorrectly specified log-likelihood function in the
leverage code.
The authors attribute it to weak identificatioon, but a genuine
identification problem would still produce finite log-likelihoods in the same ballpark as
the simpler model. This discrepancy should be investigated rather than explained away.


1. The STL plot is, indeed, not very informative. The authors recognize this and relegate it to the supplement. Arguably it still shows nothing more than a plot of the returns, since the trend and seasonality are negligible, as expected. 

1. Profile likelihoods would be approrpiate to explore the likelihood surface and obtain confidence intervals.

1. The degrees of freedom for the measurement model could be estimated. It is defined for an arbitrary positive number, so it can be estimated like other such parameters.

1. A code remark: The out-of-sample held-out likelihood appears to be a single evaluation. Given the inherent
variance of particle filters, the authors should have reported the standard error over multiple
replications (reps) of the particle filter on the test set to confirm the predictive superiority of
their chosen model.

1. The project models using static range-bias correction factor across a twenty-year observation period spanning from 2006 to 2026.
Over these two decades, the financial markets experienced massive structural transformations such as the shift in algorithmic trading, major changes in macroeconomic policy, and varying overall market liquidity.
Assuming a constant effect may be questionable, but it is a natural starting point.

1. The constraints imposes on $\phi$ and $\sigma_h$ affect inference, which is problematic unless they have strong scientific justification that is absent here. From Fig. 3, we see that $\sigma_h$ tends to the upper permitted bound, 0.5, and $\phi$ is quite often at its lower permitted bound, 0.75. A likelihood ratio test against an unconstrained alternative could indicate how much this constraint affects the fit to the data.


1. There is an issue with comparing the log-likelihoods of the two models as they are on different scales; bivariate and univariate.
Furthermore, there are two coding differences that are problematic forlog-likelihood comparison:
`dmeas_leverage` uses a Gaussian distribution for returns while `dmeas` uses Student-t.
Also, the leverage training data is mean-centered while the main model uses raw returns.


1. The results and conclusions are unclear. Exactly what is being compared and how? Also, a benchmark likelihood, such as GARCH, could help with evaluating the mechanistic models.

1. Figure 4. During the model execution, transformed_leverage_params computes: `sigma_nu = jnp.clip(jnp.exp(theta_[“log_sigma_nu”]), 1e-6, 5.0)`.
This means that no matter how large log_sigma_nu gets, the model can never actually use a
`sigma_nu` above 5.0. However, when generating the convergence plot, add_transformed_leverage_columns computes: out[“sigma_nu”] = np.exp(out[“log_sigma_nu”]), which does not
have the same constraint. So the plot shows a sigma_nu that is much larger than what the
model is actually using. Therefore, the paper’s conclusion on Figure 4 about weak identifiability is a conclusion that is partially based on a confusing plot.


1. The model has two measurement equations, one for returns, one for the range proxy. Section 7
only performs diagnostics for the range equation using ACF, but neglects the return equation.
Both are important since the joint likelihood is the sum of both log densities, so the model could
be fitting the range signal well, but not so well with the return equation. Therefore, analogous
diagnostics for the return equation should also be performed.

1. From a reader with some domain knowledge: "Parameter estimates are never translated into economically meaningful quantities.
The best-fit $\mu_h$ of −9.951 corresponds to a long-run daily volatility around 0.7%,
noticeably below the 1.38% empirical standard deviation in the data. That gap is worth
discussing rather than leaving implied. A brief conversion to annualized volatility or VIX-comparable units would ground the results for any reader with a market background."

1. The held-out model is initialized with the same `rinit` function as the training model, so the latent
state at the beginning of the test period is reset to `mu_h` rather than being propagated from the
filtered distribution at the end of the training period.

1. I think there should also be some kind of clearer evidence that the iterated filtering cooling
schedule behaves as intended when `mif` is called repeatedly with `M=1`; otherwise the
parameter traces may reflect continued random perturbation rather than any actual
convergence.

