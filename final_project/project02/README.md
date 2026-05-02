## Review comments on Final Project 2, STATS 531 W26

The project makes clearly reasoned conclusions that are supported by the analysis and relate to a substantial question for quantitative finance.

1. The lack of code makes various things harder to check. Pre-computed results are read into the report, without source code provided. This is  a violation of the final project expectations. It also magnifies the errors identified below, because they cannot be tracked down.  

1. Ref [4] does not support the point it is claimed to reference on page 2

1. Sec 2.2. $\mathrm{eq\_v}_t$ is surprising notation. Also, it is defined to be a statistic, but is then set equal to an expected value, which is confusing and presumably imprecise or incorrect.

1. Fig 5. Presumably the bottom line is an error of some sort?

1. Sec 3.2. The audience may not know what MZ is. If you introduce something new, it is helpful to explain clearly, and perhaps demonstrate how it improves over more widely used approaches.

1. Table 3 is missing column headers.

1. Table 4. BIC is computable for the POMP model from the provided Log-lik and k, so why is it NAN?

1. Use of AI: the approach described in Section 6.4 is, "The findings and figure generation were done on our own but the placements and wordings in the paper were edited and reviewed by AI." Arguably, this is the wrong way around. It is more appropriate to use AI for assisting in original drafts of code, and the editing and final reviewing should be done using your own human judgement.

1. The project is 15 pages. The specification limit is 10.

1. Too much time was taken on benchmarks in the context of the project goal to develop a mechanistic model.

1. Sec 2.3. Ref [9] does not support the point it is claimed to reference. The whole sentence is incoherent. This is poor writing by a human or AI. 

1. There is frequent reference to notebooks that are not provided, sometimes explicitly such as 04_vrp.ipynb.

1. There is no clear scientific reason provided for the decision to analyze weekly data.
The leverage effect is a highly transient, high-frequency market dynamic.
By aggregating price data to a
weekly frequency, the intra-week leverage shocks are largely smoothed over and washed out.
This temporal mismatch could explain the “mixed” and noisy evidence
the model produced regarding time-varying leverage ($\sigma_\nu$). The report fails to justify the choice
of weekly over daily data or address how this aggregation distorts the leverage state transition
equations.

1. Though not directly shown in the qmd, examining `artifacts/garch/aic ranking.csv` shows the log-likelihoods for all 27 GARCH specifications. The GARCH log-likelihoods are quite variable, and not all consistent with nesting, suggesting numerical issues with GARCH.

1. The AICs and log-likelihoods between different GARCH models with different $p$, $q$ may not be directly comparable. I think one way to get around this is to adjust the value of
the `hold_back` parameter when building these arch models; this should make it so that the
same amount of data at the beginning of the series is being held out for each model, since
the issue is that different $p$, $q$ will fit to the series starting at different indices. I did not see
any evidence that the authors did this. It was not mentioned in the document, and I don’t
have any evidence of it in the code since I cannot see it. This also means that the AICs
and log-likelihoods between the GARCH models and the Heston model may not be directly comparable.


1. Exploratory analysis showed a great fit of the residuals on a QQ-plot with the
t-distribution at low $\nu$, indicating heavy tails, but the authors fit their Heston model using
normal measurement errors. The authors mention this as a potential extension at the end,
but really I think it would have been natural to do straight away since this would be a
relatively quick tweak to the POMP model, and since the EDA indicated it would make sense.
(Also, settings like these, as seen in previous projects, generally tend to benefit from using t-distributed errors instead of normal).

1. Explanation of relationship to previous 531 projects on related topics is minimal, and none are referenced.