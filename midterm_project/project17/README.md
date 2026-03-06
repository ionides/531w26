# Review comments on Project 17, STATS 531 W26

The authors found a suitable time series data analysis for employing the methods covered in class to address a substantial question.  This is a clearly written and reproducible report. Reviewers raised many issues, listed below, some of which are significant concerns. That is the point of peer review!  

1. Fig 4. The first residual is unreliable and should be removed before plotting and computing the residual ACF. This may also cause problems for other diagnostic methods, including Ljung-Box

1. Various typos remain, e.g., the broken code in Sec 3.6.1

1. The report goes over the 8 page requirement. Conciseness is a useful skill for scientific writing. This will be graded as a minor template violation.

1. The differencing may make it very hard to identify a change in trend. Regression with ARMA or SARMA errors usually makes more sense if the error distribution is stationary, not integrated.

1. Sec 3.6 correctly identifies a limitation of the ability to make a causal conclusion. It would be better if this were articulated also at an earlier point, perhaps even in the abstract.

1. Fig 6a has some problem; it is not clear what is going on, but it doesn't make sense.

1. The code is reproducible.

1. Some discussion of roots, causality and invertibility would be helpful, and relevant to material taught in class.

1. The reference list is okay but minimal; additional context could strengthen the report.

1. The sensitivity analysis is useful - a weakness of regression with ARMA errors is that the specification of the model is not estimated, and this supplement investigates that.

1. The conclusion for Table 3 says the opposite of what the ramp coefficient shows. Table 3 shows the step coefficient is 0.0049 (p = 0.516, not significant) and the ramp is -0.0013 (p = 0.009, significant). The conclusion then states "the evidence favors a continued decline without an
additional post-2015 acceleration." A significant negative ramp is an acceleration of the decline — that
is what the parameter means. The wording seems inconsistent with the regression output. A cleaner
framing would be: no evidence of a level shift, but a significant slope change exists, and whether that
reflects ICI effects specifically is hard to establish from this data alone.
This model is not written out in mathematical notation, making it harder for the researchers (and readers) to figure this out.

1. Table 4 shows baseline SARIMA AIC = -1339.2 and ITS SARIMAX AIC = -1334.4. More negative
means better fit, so the baseline is preferred by about 5 AIC units. The report calls this "slightly improved fit" for the ITS model, but the direction is reversed.

1. The baseline and ITS models are not directly comparable.
The baseline SARIMA is fit with trend="n" (no deterministic trend), while the ITS SARIMAX uses
trend="c" plus an explicit linear time regressor in the exogenous matrix. The two models differ in both
trend handling and the presence of intervention terms. Some of the AIC change could come from the
trend specification rather than the intervention regressors. A cleaner comparison would hold the trend
setup constant across both and then add the step and ramp on top.

1. The sensitivity scan prefers June 2014, not January 2015
Figure 7 shows the minimum AIC across candidate breakpoints is at June 2014 (AIC = -1364.7), about
30 units better than the chosen January 2015 date. June 2014 is more than a year before the first FDA
approval of a checkpoint inhibitor for lung cancer (pembrolizumab, October 2015). The report notes that
"AIC is lower in the mid 2010s" but does not flag that the data-preferred breakpoint predates the
proposed mechanism. Either something else was shifting mortality around mid-2014, or the series has a
gradual change that a single sharp breakpoint cannot capture well. Either way it deserves more than a
passing comment.

1. The authors claim that “On the log scale, seasonal oscillations appear closer to constant amplitude,” but this does not seem to be the case - the plots look almost identical. This undermines the conclusion drawn in the paragraph.
