# Review comments on Project 13, STATS 531 W26

This was a well-written project that involves confidential research so cannot be posted. Some minor suggestions and corrections follow.

1. Sec. 3. "suggesting that a stationary model would not be appropriate" is inaccurate. This shows that a white noise model is not appropriate.

1. Fig. 3. Long left tail and short right tail, not "long tails" as described.

1. Page 3. "note the AIC values overall are much lower in Table 2." A Jacobian transformation is needed to make these log-likelihoods comparable.

1. The log-scale maybe is not so good with relatively small counts, as occur in these data. One option is an autoregressive Poisson or negative binomial model, with mean depending linearly on the previous observation.

1. Discrete time series models are outside the taught material in this class, but see, e.g., Raman, B., & Ravishanker, N. (2023, March 18). Dynamic time series models using R-INLA: An Applied Perspective. Chapter 9 Modeling Count Time Series. https://ramanbala.github.io/dynamic-time-series-models-R-INLA/ch-count.html#ch-count

1. Regression with ARMA errors and sine/cosine covariates might be a more sensitive way to look for seasonality from 1 year of data. This could lead to a formal likelihood ratio test for seasonality, which is missing from the report.

1. The project makes appropriate use of AI, and shows strong scholarship. It is reproducible from the provided data and code. It reports on relationships to previous 531 projects.

