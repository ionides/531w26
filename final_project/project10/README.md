## Review comments on Final Project 10, STATS 531 W26

A comprehensive analysis of the Breto (2014) model. Although the model itself did not require development, the authors went beyond previous STATS 531 projects, trying some new analysis including probes.

1. The AI statement is reasonable and consistent with the report.

1. The model is the same as Chapter 17. This is unambitious, but it can be legitimate to carry out a careful extension of previous work rather than looking for greater novelty.

1. Sec 5. Avoid raw code in the main text; better to write equations and reference the code.

1. Fig 7 claims to be doing MCAP, but the cutoff is 1.92 and the profile is not smoothed, so it appears MCAP is not used. The code has a switch that either does `pp.mcap` or something else. It appears that the text CI is from MCAP and does not correspond to the non-MCAP construction in the figure.

1. Perhaps larger GARCH benchmarks would be more competitive? More importantly, non-Gaussian distributions for GARCH and POMP models (e.g., t-distributed) would likely give higher likelihoods, allowing finer resolution of other model-based conclusions.

1. Table 6. It is intetesting that ACF(1) is the clearest anomaly. This matches the rather surprising ACF(1) term in Fig 1.

1. Sec 13. The boundary correction for the likelihood ratio test is nicely done. However, it would be good to have a reference for the Chernoff/Self-Liang result. That assumes that the boundary has some regularity, and it is not obvious it holds exactly here. Also, there is substantial Monte Carlo error that could be taken into account for the p-value.

1. Sec 6. The simulation study should be done from the fitted parameters; here it is done with parameters drawn from an initial box.

1. The global IF2 search gives $\hat\sigma_\nu = 0.0499$, but the profile likelihood
section reports a profile MLE near 0.0010, which is also at the lower edge of the grid. These are very
different results. If the profile estimate is close to zero, then the data may not strongly support a genuinely
time-varying leverage process.
