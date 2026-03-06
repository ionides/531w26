# Review comments on Project 12, STATS 531 W26

This project tackles a sociological question of clear relevance. Appropriate models are developed, though there is no proposed advance in understanding the sociology of crime and the economy in the context of the epidemic. In fact, the strongest conclusion is negative, "after accounting for the Covid indicator and seasonal dependence terms, unemployment does not look to be associated with either property or violent crime." Bootstrap methods help to reinforce this. Discovering solid evidence of real relationships in these complex systems is hard, so it is not surprising that this analysis failed, but it makes an honest effort. Some specific points follow.

1. The investigation of the relationship is carried out thoroughly, giving some credibility to the eventual negative result (no clear empirical relationship between unemployment and crime)

1. The data are called "crime rate" but we are not told if this is total crime, or crime per, say, $10^5$ inhabitants. Assuming it is total crime, the declining population of Detroit should be taken into account. Indeed, the source code and the
data explorere from the FBI clearly indicate that these are crime counts, not rates.

1. Viewing average crime as a slowly varying structural trending process may be better than treating it as a unit-root stochastic process, since the latter gives less opportunity to study explanations for the trend. The authors don't explicitly use unit-root methods, but their models often have close to unit roots, suggesting that the trend is not sufficiently well parameterized.

1. The bootstrap analysis is ambitious and helps to support the conclusions.

1. The authors choose a model with AIC reported as -57, which is much higher than anything in the tables. An effect this big is unusual, and should raise an alarm that there's probably a problem with the comparison. The report value is hard-coded, so we can't readily trace back to find where it actually comes from. The code seems to fit $(1,0,1)\times(0,0,1)_{12}$ not the reported model.

1. Sec 6.5. The AIC table has many inconsistencies that should be identified and discussed.

1. Most code is reproducibly available in the code chunks, but inline code should also be used for in-text numbers.
Results tables should also not be hard-coded.

1. The authors explicitly state that a log transformation was applied to stabilize variance. However, the formal equation provided for the model defines the response variable directly on the original scale without the logarithmic operator. This reduces precision and clarity of the writing. 

1. For reproducibility, it is better to avoid the hard-coded reference to the location of the Python environment.

1. There is a substantial problem with the post-Covid coefficients (Sec 4, comparison table). Both models return a beta_post of roughly 3.0–3.2, which implies about a 1900% increase in crime relative to the pre-Covid period. This is a red flag that the model's coefficients are uninterpretable. The authors acknowledge it, but the issue is not fully resolved.

1. The authors mention a shift in data reporting starting in 2021, and state that further details are in the
supplmentary material. However, no such such explanation appears there or elsewhere. The authors claim
that they don’t expect it to impact the analysis, but this claim is not justified.

