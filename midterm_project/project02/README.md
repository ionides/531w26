# Review comments on Project 2, STATS 531 W26

A thorough investigation, broadly correct. Readers identified the following points: 

1. The conclusion and abstract are underwhelming. The goal of developing a statistical model is to learn something, or do something, but the conclusion is happy with just having developed a model.

2. Sec 4.2. "All six predictors are statistically significant ($`p < 0.001`$), confirming that the pre- and post-COVID trend structures differ meaningfully." This appears to be based on an OLS fit, though the results are not shown. Here, there is strong dependence and OLS standard errors should not be considered even if OLS is used to estimate a trend.

3. Sec 4.4. "Is shown above" is wrong; the AIC table is in Table 1, which has ended up below. Better to refererce as, "Is shown in Table 1."

4. Sec 4.5. The residuals are long tailed. The extreme outlier is noted, but its consequences are not investigated. The project does not even report when it occurred, just "likely corresponding to an anomalous week during the early COVID-19 period". It would be useful to confirm or deny that conjecture.

5. Sec 8. Limitations should be discussed in the main text. The meaning is unclear of "without causal inference" and "restrains the power of the time series model at its assumption of stationarity". 

6. Sec 4.5. Ljung-Box is not very useful as a check for residuals after fitting a model by AIC. AIC and LB are measuring very similar things, so a problem missed by one will likely be missed by the other. This was also explained in the midterm exam.

7. This analysis does not present the estimated coefficients from the SARMA model. Readers cannot assess:
Which AR/MA parameters are statistically significant? Whether the estimated roots lie inside the unit
circle? Whether any parameters are near the boundary of the stationarity region? This omission is problematic for readers who want to understand the fitted model, and interpret the economic/epidemiological meaning of the dynamics.

8. The pandemic indicator is set roughly at May 2020. Nevertheless, Figure 2 depicts the ILI going down from
around week 10-15 of 2020 around March-April much earlier. The May turning point, which is chosen, may
be too late to catch the pandemic effects’ initial onset. The authors could clarify how this breakpoint was
determined: was it a data-driven, theory-driven, or an arbitrary decision?

9. For completeness, the authors could have looked at a periodogram; likely, it would show not much new and end up in a supplement.

10. Written following good standards of reproducibility and scholarship.
