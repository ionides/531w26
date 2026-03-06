# Review comments on Project 15, STATS 531 W26

The authors investigated a topic of interest to many students and suitable for time series analysis. Referees raised various concerns, as follows.

1. Sec 2.2. It is surprising to see any regular monthly pattern in exchange rates. Is this real? Is there an explanation? Spectral analysis might have helped to clarify this, but this wasn't done.

1. Fig 1. Both decompositions look strange. For example, the level is fast-changing. How is this decomposition calculated? What do the raw data series look like - it might be better to look at a data timeplot rather than a dubious decomposition. 

1. Sec 3.1. ADF is not good for investigating anything other than a unit root in a linear time-invariant process. 

1. Sec 3.2. How do you address comparison of models at different levels of differencing via AIC?

1. Sec 4.1. As explained in class, it is usually a mistake to have ARIMA models this large.

1. Fig 3. There is an error of some type in the WTI analysis, as shown by the residuals. The point of looking at residual plots is to notice when there is a problem and then do something about it. 

1. Sec 4.2 Ljung-Box is not a strong test when a model is selected by AIC from a large number of choices. AIC and L-B look for similar things, so any problem missed by AIC will likely be missed by L-B too. These issues were discussed in class, and also in posted reviews for various previous projects.

1. The qmd report file hard codes values, which is a weakness for reproducibility. Additional code is provided, but it is not so easy to work out what is going on. For example, we can't immediately see what is the code that gave the numbers in Table 3, to find the error. This does not meet the expectations for the midterm project, as developed in homeworks and the midterm template. It is graded as a minor template violation.

1. The extraordinary AIC gap for WTI in Table 3 requires audit. The reported WTI best model has AIC ≈ 60.8 while nearby candidates have AIC ≈ 3983, a difference of roughly 3900. Such a gap is possible only under very different likelihood scales or
modeling setups, and therefore deserves explicit verification. Without reproducible code, it is impossible to rule out inconsistent preprocessing, differing sample sizes, or a coding/reporting mismatch.

1. Saying “Both series require d = 1” is inconsistent with then immediately justify selecting a WTI model with d = 0 by saying SARIMA can “accommodate near unit root behavior via the seasonal AR component.” The new ACF plots make that justification weaker, because the persistence you are diagnosing in WTI is not primarily seasonal persistence at multiples of 5. It is broad low frequency persistence that shows up as a near one ACF across essentially all lags in the level series, which is characteristic of a nonseasonal unit root.

1. Sec 4.3. The explanation of the CCF does not seem to match the presented figure.

1. Sec 4.3 The Granger causality test is relevant and potentially powerful for the question of interest. More details would be good for a test not covered in class. What are its strengths and weaknesses? How reliable is it in this situation? What is the test statistic?

1. The cross-series analysis (CCF and Granger tests) uses first-
differenced series, but the best WTI model uses $d = 0$. So the univariate analysis and the
cross-series analysis are working on different transformations of the same data, and this isn’t
discussed.

1. Multiple readers were concerned that the interpolation of missing values does not preseve causality---it leaks future information into the present. If interpolated points are rare, this might not be a problem. But if we are looking for subtle phenomena, this could become a substantial bias.

1. A small point: The abbreviation WTI is not defined.

1. All the figure text font sizes are small and hard to read.

1. No spectral analysis is included, which would have been a natural way to confirm the period-5 pattern and check if $s = 5$ is actually an appropriate seasonal period.
