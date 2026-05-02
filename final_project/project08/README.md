## Review comments on Final Project 8, STATS 531 W26

A coherent project, building quite closely on previous STATS 531 projects, but adding its own material. The question is clear and falsifiable. $\phi> 0$ is appropriate to test for game-to-game
hot-handedness, and a profile likelihood is a principled tool for this.
The qmd file is written to a good standard of reproducibility. 

1. “missing usage %” was recognized as a suitable teammate-availability proxy. This is an extension of the cited baseball project rather than just copying it, and filtering to players averaging 12+ minutes was considered suitable domain knowledge.
Other readers thought there was inadequate evaluation of the role of playtime. 
Apparently, the issues were not entirely clear to those who do and don't have a basketball background.
Existing literature on evidence for hot-handedness in basketball or sports in general
could be cited to improve the introduction. Perhaps it would help readers with less sports background.

1. The Fig 5 profile must have some mistake. The maximized log-likelihoods are far below the MLE, but each profile should pass through the MLE at its maximum. 

1. For the ARMA modeling, the report references an AIC table in the supplementary material, but the table contains only NaN values. This makes it difficult to verify the reported log-likelihood
for the baseline model. From the code, it appears that a try/except block is used, where failed ARIMA(p,0,q) fits result in NaN values. Additionally, based on the referenced ACF plot, there
appears to be a significant lag at 20, as the autocorrelation exceeds the significance bounds.

1. The global search window for phi is confusing.
If they only search across phi in [0.01, 0.99], but all
the phi’s they get are negative, then they haven’t chosen the best window.

1. When they compare the Poisson vs Negative Binomial measurement models, they get
log likelihoods of -355 and -352 respectively, which are 20 points lower than the log likelihood
achieved by their prior search with Poisson (reported in Section 4.3).
Because they had already run a global MLE seach with the Poisson model, they could have used the same settings for the negative binomial search, for a fair and consistent comparison.

1. $\beta_2$ and $\beta_3$ are swapped between text and code. Text has $\beta_2$ on home, $\beta_3$ on missing usage, but the cluster code has `beta 2 * Usage Missing + beta 3 * home`. Trace plots and prior ranges become mislabeled, and the NB results use the swapped version.

1. Negative $\phi$ deserves more thought. The authors call it “misspecification,” but $\phi < 0$ has a clean interpretation: mean reversion. This is evidence against the hot hand and should be framed that way.

1. The inline POMP model built in Sections 4.1–4.3 uses a standardized defensive rating covariate (DF_centered), while the cluster code shown in the appendix uses the raw `Rolling_DEF_RATING` without any centering.


1. The ARMA equation includes a differencing operator that is not in the code.
The written formula has a (1−B) term, implying an integrated ARIMA with d=1. The
estimation code uses order = (4,0,4), i.e., d=0, which is correct for a stationary series.
The equation in the text should be corrected to match the code.

1. Power analysis could be helpful. Would the limited data be sufficient to detect a hot hand phenomenon of reasonable size if one existed?

1. Residual diagnostics or simulations from the fitted model would be helpful and are not provided.

