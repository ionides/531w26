## Review comments on Final Project 12, STATS 531 W26

The project takles a real issue and is written to a high standard of reproducibility, clarity and academic precision. The project is not especially ambitious - it is similar to many previous 531 projects, with a new dataset. However, every dataset and scientific situation requires careful attention. On top of that, moving to Pypomp is an advance over previous projects.

1. Figures could use numbers and captions.

1. Sec 3.2. These large plots do not add much information. The seasonal plot in the Supplement (Sec 8.1) is more informative.

1. The model equations are missing overdispersion.

1. The use of a Colab GPU is innovative.

1. The SARIMA log-likelihood can be compared to SEIRS with the help of a Jacobian adjustment. 

1. The global search has a lower log-likelihood than the best local search, which is inconsistent and should be addressed.
Specifically, the report states that the local search reached “a best endpoint log-likelihood of roughly
−275,” while the global search achieved −304.8. The local value of −275 comes from the
MIF iteration trace — a noisy estimate computed during the perturbation phase.
The code block that evaluates the local endpoints with an independent particle filter is
marked `#| eval: false` and was never executed. The global value of −304.8, by contrast, is a
proper particle-filter evaluation at J = 10,000.
of the methodological difference rather than a genuine result.
The local endpoint pfilter should be run and reported; if it really is much higher, the global search should be considered unsuccessful and should be debugged.

1. Scatterplots and/or profiles would help to investigate weak identifability.
It may be good to know that some biologically plausible parameters are statistically plausible, but for data analysis it's also good to know what biologically surprising values are statistically plausible. 


1. The COVID disruption period is not accounted for in the SEIRS model.
The SARIMA model includes explicit COVID-step regressors covering 2020–2022 and
validates that these terms are statistically significant. The SEIRS model has no analogous
mechanism: it simply tries to fit through the pandemic suppression period using the same
transmission parameters as the rest of the series. This is a structural misspecification — the model is asked to simultaneously explain near-zero hospitalization rates during 2020–2021 and normal seasonal peaks before and after, using a stationary sinusoidal $\beta(t)$.
A time-varying $\beta(t)$ (beyond seasonality) or a multiplicative regime-shift parameter (similar to the regression analysis) would substantially improve the mechanistic model and allow the two approaches to be compared on equal terms.


1. The response variable is rounded hospitalization rates, not counts.
The POMP observation model is a negative binomial over “pseudo case counts,”
constructed by taking `df_weekly['Rate'].round()`. Many weeks have rates below 1.0 per
100,000 (especially during 2020–2021), so rounding maps them to zero. Rounding a
continuous rate to an integer before passing it to a count-based likelihood introduces
discretization error and information loss. The report flags this in Section 8.5 but treats it as a minor issue; in fact, it has potentially large impacts.
It would be entirely okay to use a continuous measurement model to acknowledge that the data are rates.
This would help to make the log-likelihood comparable to the benchmark (if Jacobian effects are dealt with correctly).


1. In Sec 3.4, it is better not to report a p-value from OLS with highly autocorrelated residuals. That is what regression with ARMA errors is for.

1. The authors admit that the use of a pseudo-population $N = 10^5$ to turn
their rates into counts may not be appropriate, the authors did not perform model checks with
different values of $N$ , so we do not get to see how sensitive parameters are to different values
of $N$ in this setup. Or, perhaps some alternative model would have been more suitable for this dataset so that one could use the rate directly.
There's no real need to use a discrete measurement model - if your data are real-valued, a continuous measurement model is natural.
