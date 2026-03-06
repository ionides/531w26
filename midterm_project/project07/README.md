# Review comments on Project 7, STATS 531 W26

A strong project, with a clear motivation and an accurate execution. Some specific points follow.

1. Abstract announces the discovery of some nontrivial discoveries arising from the time series analysis: Comparing fine and coarse particulate matter is a nice way to gain value, since anything that is different for PM10 and PM2.5 is non-obvious to most readers.

1. Whether there is a detectable weekly effect is also a plausible and non-obvious hypothesis, worth testing. 

1. The inclusion of cross-correlation, coherence, and phase analysis are good additions and give depth
beyond a purely regression-based approach. The spectral analysis is thorough and provides additional
insight into shared periodic behavior between meteorological variables and particulate matter.

1. The fitted ARMA model exhibits roots near the unit circle boundary, suggesting near nonstationarity.
However this can destabilize standard errors, weaken asymptotic approximations, and compromise any subsequent likelihood-based inference.

1. The project does not report any diagnostics to asses heteroskedasticity, normality of residuals, or
any potential volatility clustering. Given the nature of environmental data, some checks of this nature would have strengthened the analysis.

1. The CCF uses first-differenced data to remove shared trends, but the regression immediately after uses the original levels. If differencing is needed for the CCF to be meaningful, the same question applies to the regression, unless the covariate successfully explains the entire trend.

1. One concern is potential shared seasonality where meteorological variables and PM
share strong periodic structure where direct regression of PM on weather can
complicate interpretation. Coefficients may partly reflect shared cycles rather than a direct causal effect. Using distributed
lag terms might help separate immediate associations from shared periodic patterns. 

1. Issues of causal interpretation should be discussed, since readers generally want to know the extent to which associations may have a causal meaning. 

1. Inline results should also be sourced from Python not hard coded. Other than that, the reproducibility standard is high.