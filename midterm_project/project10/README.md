# Review comments on Project 10, STATS 531 W26

A coherent analysis. Many reviewers noted that the authors missed a chance to use regression with ARMA errors to incorporate the pandemic via covariates, perhaps leading to avoidance of the very large prediction intervals estimated in the forecasting section when simply using a time-invariant SARIMA model. Various other comments follow.

1. Fig 2. This seasonal decomposition does quite a good job here. But the seasonal term is perfectly periodic? Some changes in periodicity seem to show up as redisual. There are many ways to do such a composition, and exactly how this one is done is not explained.

1. Sec 3.3. The ADF test is inappropriate here. There is clearly a pandemic effect. Before the pandemic, growth looks approximately exponential, which ADF is not designed to handle. Only use ADF if you think its null hypothesis (a unit root) is relevant. As explained in class, it is bad practice to use it as a generic test for non-stationarity. Nonlinear trends, as we have here, provide an example where ADF may have unexpected behavior.

1. Given the visually obvious trend, an ADF test was not necessary. The authors note that the outcome of the test is “marginally significant at the 5% level,” implying stationarity, but proceed to apply first differencing anyway. 

1. Sec 4.1. "The final 30 months of observations (from June 2023 through November 2025) are reserved for testing. The remaining 287 months from July 1999 through May 2023 are used for model estimation."
This assumes that the differencing makes stationary modeling appropriate despite the pandemic.

1. Sec 4.4. Ljung-Box is not very useful as a check for residuals after fitting a model by AIC. AIC and LB are measuring very similar things, so a problem missed by one will likely be missed by the other. This was also explained in the midterm exam. Moreover, since
the Q–Q plot indicates heavy-tailed residuals driven by the pandemic shock, a stronger
discussion of potential model misspecification would be appropriate. Passing Ljung–Box does
not imply correct distributional assumptions or structural stability, especially in a context
involving a major economic shock.

1. Sec 7. The weak modeling of the pandemic (addressing it by differencing) may help to explain why the resulting predictions have wide uncertainty.

1. The project is appropriately put in the context of a previous project, thouth there should be in-text citations where work or conclusions from this project are
referenced. The project could have benefitted from studying more previous projects.

1. Figure 2. The authors conclude that there is a post-COVID volatility increase. Actually, it appears that the seasonal amplitude increases, but the decomposition used cannot accommodate that.

1. Section 4.4: An in-text reference to Table 1 would be appropriate here and in
section 5.1. Sections 4.4, 5.1 and 5.2 appear to repeat information regarding
model selection and diagnostics.

1. Section 4.5: Three metrics to evaluate forecast accuracy are described and calculated, but only one is referenced in the discussion. Explaining the choice of the metric used and perhaps not including the others if they were not going to be used would be beneficial.

1. Residual diagnostics reveal heavy tails and a clear deviation from normality in the Q–Q plot.
Non-Gaussian residuals imply that Gaussian-based forecast intervals may underestimate uncertainty. While this is an appropriate
first step, some Bootstrap or simulation-based interval construction could provide more reliable coverage, especially in the presence of future shocks.

1. Section 5.3. "The model achieves MAPE = 3.21%, indicating strong short-term forecast accuracy." Why is 3.21% strong rather than weak? Presumably, we need a point of comparison. How should this be interpreted in the context of the very wide prediction intervals in Fig 4?

1. Regression with ARMA errors could have been used (in various possible ways) to provide an intervention analysis, modeling the pandemic effect. Lack of that may explain the wide prediction intervals.

1. BIC results are presented but not discussed. Better to avoid that here, unless you want to engage in the difference between these measures.
