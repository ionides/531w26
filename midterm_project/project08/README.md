# Review comments on Project 8, STATS 531 W26

An original project making a claim of general interest. Referees had some questions about how solid the findings are. If they hold up to rigorous criticism, the project appears to be a contribution to political science.

1. Sec 3. The QQ plot has long tails. The red line hides this somewhat, since a straight line fitting theoretical quantiles from -1 to 1 would show the long tails more clearly. It is not clear whether this amount of non-Gaussian behavior has an effect, but it should be noted and perhaps explored (e.g., by t-distributed tails; sometimes this suggests a transformation of the data, but not here)

1. Figs 5 and 6. The strong lag-zero relationship is primarily driven by the collapse of the Soviet Union (1990) and also perhaps by the end of the second world war. It would be interesting to see if the pattern remains outside these two special historical events.

1. Fig 6. The "polarization as a lead indicator" looks consistent with a general pattern of negative (but marginally insignificant) effects as well as the specific (marginally significant) lag of 10. This is a plausible discovery.

1. Overly Absolute Causality Inference: When interpreting the CCF results, the authors use phrasing with strong causal implications (e.g., "induce," "feedback loop"). However, statistical cross-correlation (lagged correlation) does not equate to physical causality in the real world. It is better to be more cautious and stick with associative language.  "leading indicator" is probably okay for this; it only indicates a lag assocation.

1. Incorporating an analysis of major global events (e.g., World War I & II, the Cold War, the Great Depression) in the discussion section would be beneficial. Examining whether these unobserved historical shocks acted as confounders simultaneously driving both polarization and regime changes would be interesting.

1. Multiple testing issues should be discussed in the interpretation of the cross-correlation analysis. Figure 6 tests CCF values at 31 lags and
treats several spikes above the 95% confidence bounds as meaningful findings, but with 31
simultaneous tests at the 5% level, one or two significant results are expected by chance alone. No
multiple testing correction is applied. More importantly, the regime change series in Figure 4 is
dominated by one extreme spike around 1990 from the post-Soviet democratization wave, and this
single event likely drives much of the observed CCF pattern. A basic robustness check would be:
if we remove the five years surrounding 1991, do the significant lags at -10 and +10 to +15 survive?
If they disappear, the entire “10-year leading indicator” narrative collapses into a one-off historical coincidence. This check is never done, which makes the bivariate conclusions much harder to trust.

1. The CCF code is misintepreted. It uses
`ccf_vals = sm.tsa.stattools.ccf(pol_change, reg_change, adjusted=False)`, which means
`pol_change` leads `reg_change`. Conversely,
`ccf_backwards = sm.tsa.stattools.ccf(reg_change,pol_change, adjusted=False)`
means `reg_change` leads `pol_change`. However, when they
are combined, the backwards values are placed at negative lags after reversing, so lag -10 in
the `plot is ccf_backwards[10] = corr(reg[t], pol[t+10])` and this implies that regime change
leads polarization by 10 years. However, the paper interprets it backwards: “A significant
negative spike at lag –10 suggests that an increase in polarization today predicts a decline in
democratic regime growth a full decade later.” On the other hand, the positive lags are where
polarization leads regime change, but the paper mislabels this as “regime change as a leading
indicator” as well.

1. The bivariate narrative is intruiging but perhaps overconfident. By ignoring the 1990 outlier and the compositional shifts in the V-Dem dataset, the authors
have turned what may be a historical coincidence into a causal-sounding theory. Toning down the
language from “proves” to “suggests” and running a basic robustness check on the 1991 period
would make the analysis more reasonable.

1. The abstract claims that SARIMAX modeling is used, but this may be a little misleading
because the project is using a linear regression model with ARMA errors as the main model.
This could be called ARMAX.
ARMA models are commonly fitted via the general SARIMAX function in Python, but more clarity would be preferable.

1. Scholarship is handled well overall as well. The references are all well-cited and the relationship to prior work
is clearly described. The code is appropriately provided and reproducible. The paper correctly
distinguishes its contribution relative to the prior project, clearly explaining what is new rather
than leaving it to the readers to figure out the distinction.

1. Pre-whitening (a method not covered in class) could be used to help add interpretability to the cross-correlation plot.


