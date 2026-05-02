## Review comments on Final Project 3, STATS 531 W26

A cleary written project that uses the flexibility of the POMP framework to explore several model variations relevant to understanding stochastic volatility.

1. The AI usage statement in Sec 13 is appropriate and consistent with the report. 

1. The project is written to a good standard of reproducibility. However, a global random seed is set anywhere.
Because the rendered numbers are loaded from pickles, the report itself is reproducible, but anyone who wants to rerun the
search scripts will get slightly different MLEs each time. A one-line
`jax.random.PRNGKey` setup at the top of each script would close this gap.

1. The log-likelihood appears interestingly multimodal, from the collections of convergence points in the diagnostic plots. This could be discussed. What are the consequences?

1. Profile likelihoods would be useful for obtaining confidence intervals and exploring identifiability and multimodality.

1. There are many variations on GARCH that can be more competitive as benchmarks than standard GARCH. Here, the main test is between two stochastic volatility models, so the role of GARCH is perhaps not critical beyond showing that the SV models are not unreasonable.

1. The authors notice that `hold_back` can be used to make log-likelihoods comparable between GARCH models, which is a good innovation over previous 531 projects. The same could be done with POMP models by omitting the conditional log-likelihood components for the `hold_back` data.

1. Bimodality in POMP Likelihood Inspection of the mif2 traces and pfilter results shows
strong bimodality in the final log-likelihood distribution for every model with t or skew-t errors.
The project acknowledges ”several bad paths” but does not connect to the fact that this bimodality
is a documented property of the Bret´o model.( i.e. in the 2022 Ethereum 531 project using the same
model). The following comment on that project apply to this one too: “The paper should have
cited this known behavior, explained its cause, and discussed whether 250 starts with 250 iterations
is sufficient to be confident the global optimum is reliably found. The MLE values are likely correct
given the pfilter re-evaluation step, but the convergence quality should be made explicit to the
reader. ”

1. More diagnostics would be helpful. While ESS is a good diagnostic, it does not
substitute for standard residual diagnostics like the ACF of $y_t$ or $\exp(H\_{t\_{\mathrm{filtered}}}/2)$, QQ plots.
These would directly assess whether fat-tailed errors resolve the misfit seen in the basic model.

1. The report compares the model variants primarily using log-likelihood and effective sample size but
does not compare the resulting parameter estimates across models in the main report. Although
the parameter values are reported in the supplementary material, they are not interpreted or used
to assess how the model modifications change the fitted volatility dynamics. This is particularly
important given the apparent instability in the trace plots, which suggests that some parameters
may be weakly identifiable. From the report, it is unclear if similar likelihood values correspond
to very different parameter configurations, making it difficult to draw strong conclusions about
the effects of the model extensions.

1. In general, the authors link their work to prior Stats 531 projects well but the link to academic
literature is limited. Aside from two citations, there is little discussion of how the chosen modeling
approach relates to standard methods in financial econometrics. It is difficult for the reader
to understand how their work connects to broader literature without this information provided
anywhere in the report. For the AR(2) Bretó model extension, the supporting citation is a stack
exchange post instead of an academic source. While such sources can be useful for intuition, citing
peer-reviewed literature would provide a more rigorous foundation for the modeling choices.

1. ESS collapse on the tail of the series is never resolved. Every Bretó variant —
standard, t, skew-t, AR(2) — shows the same deterioration of effective sample size near
the end of the data. The author flags this in the Discussion but does not investigate.
Because the issue is shared across all of the candidate models, it is not affecting the
ranking, but it does mean that the best log-likelihood is being computed on a partially
degenerate filter on the same span of the data. A simple fix would have been to bump the
particle count for the affected segment or to compute the per-observation conditional
log-likelihood and look for the outliers driving the drops.

1. Hyperparameter choices for IF2 do not appear in the rendered PDF. Particle counts,
IF2 iterations, cooling fractions, and random-walk standard deviations all live in the Python
search scripts but are not summarised in the report itself. For someone reading just the
PDF, it is hard to tell whether the ~3 log-likelihood gap between t and skew-t is
comfortably above the IF2 noise. A small table of search settings would have fixed this.

