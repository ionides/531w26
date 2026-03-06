# Review comments on Project 4, STATS 531 W26

Various relevant time series techniques are tested on a well motivated task. Some curious results were in fact code or interpretation errors.  

1. Abstract. Finding weekly and seasonal periodicity in airline travel data is not surprising. If you have some reason to think that this is a non-obvious discovery, you should explain this.

2. Abstract. SARIMA(2,1,1) is not a complete specification. The seasonal term needs to be specified.

3. Sec 1. This help to motivate the study of TSA checkpoint data, in the context of previous STATS 531 projects.

4. Sec 2.3 and 5. What is the value of studying the spectral entropy? It was not clear what was learned from this. The reviewers just passed over this section without comment, but the material should be better motivated or removed.

5. Sec 6.0 The SARIMA model, from the code, turns out to have a (0,0,0) seasonal component, so it is actually ARMA. That may explain why it fails to fit the seasonality. The details of this, promised to be in the supplementary material, were not found there. If we run SARIMA $(2, 1, 1)\times(1, 1, 0, 52)$, we see a substantial improvement to the MAPE_pct, from 9.44 to 2.11, just barely edging out the Naive (last year) model. This resolves the surprising results.

6. Sec 6.1. The results table looks very surprising. Why should last year be a much better predictor than last week? it looks like there is some misunderstanding here. If a naive seasonal rule dominates, the project should probably try to strengthen the forecasting approach until it reliably beats the seasonal benchmark.

7. The STL decomposition looks like it has problems. Too much high-frequency information is showing up in the seasonal component. This could not reasonably be used for forecasting, since you don't know the future seasonal component, so it is not clear how that was done in this project.

8. It is nice to use TSA data collected directly, rather than using a historical dataset.

9. The reference format is non-standard. In future, please use a standard citation style such as APA (American Psychological Association).

10. The QQ-plot for SARIMA residuals looks wrong - perhaps the residuals need to be standardized?

11. Apparently, there's a huge outlier near the start of 2022 - this should be identified and discussed.

12. Sec 6.0.1: the inconsistencies in the AIC table should be noted and discussed

