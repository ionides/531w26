## Review comments on Final Project 13, STATS 531 W26

An ambitious project that investigates a PanelPOMP model, i.e., a collection of time series modeled with independent dynamics. Some parameters are shared between time series, others are specific to each time series.

1. Reproducibility problems are admitted and discussed, and placed in the context of the project goals.  However, it is hard to evaluate the issues that were not found because of reproducibility issues. Data unavailability is inescapable due to privacy considerations, but reanalysis of simulated data could still be shared.

1. The masked-data prediction benchmark is appropriate for the project goal. Holding out 20% of
observed hourly Fitbit data and evaluating reconstruction quality directly tests whether the model
can estimate missing rewards. The comparison between filtering and smoothing is also useful
because rewards can be backfilled after future data are observed.

1. Figures should have numbers and captions.

1. Sec 7. The results table shows a higher log-likelihood for ARMA, but also a much higher RMSE. The proposed explanation in the text is unclear. Elsewhere, the authors acknowledge that ARMA predicts the observed reward while the POMP models predict a latent count that always exceeds the observed value. The measurement model should correct the scaling, but perhaps something went wrong?

1. The regularization idea is interesting but incompletely developed. Adding a penalty to `dmeas` may have some useful consequences, but also raises difficulties. For example, the resulting quantity is no longer a likelihood, so AIC, profile likelihood, etc are not applicable without some suitable correction.

1. The hour-of-day and weekday baselines are mentioned but never actually
described
The log-linear predictor for latent step count includes hour-of-day effects and day-of-week
effects that are described as estimated empirically for each participant from the observed
step series, with further details deferred to supplementary material. However, the
supplementary section does not actually explain how these estimates are computed —
for example, whether they are raw sample means per hour and weekday, whether any
smoothing is applied, or how weeks with high missingness are handled during estimation.
This is an important part of the model because it sets the baseline expected activity level,
and its absence makes the model harder to evaluate or replicate.

1. The claimed differences between pooling regimes
are small, as the authors explain (”807 vs 809 is likely within range of Monte Carlo noise”).  But, for a final project, transparent discussion and evaluation of results is more expected than obtaining a definitive scientific discovery. 

1. There are conceptual issues in the measurement model. The hour-of-day
and weekday effects ($\alpha_h$, $\delta_w$) are said to be ”estimated empirically for each participant from the observed step series,” and are then treated as known covariates
inside the POMP. Since theses are fit from the same data used to evaluate the likelihood, this has consequences for estimation of degrees of freedom.

1. Additional references would be appreciated since this is an interesting but unusual topic for 531 projects.

