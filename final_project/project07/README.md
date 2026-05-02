## Review comments on Final Project 7, STATS 531 W26

This report demonstrates a full range of POMP tools on a substantive scientific problem. Use is made of JAX and Pypomp. 

1. Reproducibilty is weak - the qmd file does not contain the code, and the mentioned file `analysis_clean.py` is absent from the submission (added as a late addition). 

1. Too much space (3/10 pages) is spent on ARMA type models for an article that aims to focus on mechanistic models. 

1. Attention to effective sample size looks promising, but Fig 7 shows poor global optimization, with estimates spread across 7 orders of magnitude of log-likelihood.

1. Fig 9. The simulations exceed the data by a factor of approximately 17. The text
describes this merely as the model tending to “overshoot the peaks,” which is a severe understatement. An order-of-magnitude scale discrepancy of this kind indicates a major model misspecification issue. On the timescale of a final project, these things can't always get resolved but they should get acknowledged.

1. Fig 10. It is good to look at probes. Here, various probes show incompatibilities between the fitted model and the data (as does Fig 9). An honoest assessment of these highlights that the model needs more work.

1. Fig 11. The profile log-likelihood is also very noisy, and the noise on the scale of order $10^6$ shows there is a problem.

1. Sec. 7. You can do a Jacobian calculation to obtain the log-llkelihood on the natural scale when a model is fitted on a log scale. Doing this would help you notice that your POMP model is fitting poorly.

1. In Section 6, there are various missing results, set as `TK` in the qmd and the report. This is a substantial incompleteness.

1. Contradiction in the simulation identifiability study (Figure 6).
The caption of Figure 6 states “logLik at truth = −2240.0; logLik after refit = −593283.6.”
This is a discrepancy of approximately 591,000 log-likelihood units. Yet the accompanying
text asserts that “the refit log-likelihood is within Monte-Carlo error of the truth.” This is a massive inconsistency; either a typo of some sort or a major indicator of a methodological error such as a coding bug. In the context of Fig 11, a bug seems possible.

1. GARCH is mentioned as a comparison point in the abstact (twice) but does not appear in the report or supplement. GARCH is usually inappropriate outside finance, so the issue is surprising at multiple levels. The use of a financial modeling project as a template may explain this weaknesses. It would have been better to learn more from strong past epidemiology projects. 

1. The conclusions fail to return to the question of race-stratified dynamics, since only white data are modeled. This may be a necessary limitation on the timeframe, but deserves discussion as a limitation to make the report coherent.

1. The model uses a closed population of $N = 10^6$ as an “effective susceptible-pool size,” but the White stratum of the United States contains roughly 200 million people. No justification is offered for this choice beyond computational convenience.

1. The SEIR model has no demographic renewal (births, deaths, aging) over a 22-year horizon, and no waning immunity. A model that cannot explain replenishment of the susceptible pool over multiple epidemic cycles will struggle to fit the data - which appears to be what has happened.

