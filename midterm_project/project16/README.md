# Review comments on Project 16, STATS 531 W26

The authors study financial volatility directly, by analyzing derived volatility series rather than direct market returns. This turns out to be suitable for ARMA modeling, focusing on risk rather than directly on returns. This is generally a carefully conducted analysis, though the following points contain a few nontrivial concerns raised in peer review.

1. The monthly cycle seen from the spectral analysis is perhaps not expected but is given a convincing explanation.

1. Sec 4.2. Models with VIX and Volume as covariates are a superset of VIX only models. So, the AIC cannot mathematically be 200 units worse. There is some error.

1. Sec 4.3 This looks reasonable and plausible, but uncertainty arises from the problem with Sec. 4.2.

1. Sec 6.2 Ljung-Box is not a strong test when a model is selected by AIC from a large number of choices. AIC and L-B look for similar things, so any problem missed by AIC will likely be missed by L-B too.

1. This is an informative discussion of forecast accuracy.

1. Using contemporaneous VIX could create mild look-ahead or simultaneity biases. Using lagged VIX would offer a better specification for predictability. If this subtle but important issue has somehow been attended to, it needs to be explained so that readers are not concerned. 

1. Tables, figures and in-text numbers are hard-coded. There is some code is main.ipynb, but this is a weaker standard of reproducibility that requested for the midterm project. It is hard to be sure if the code is complete, and whether the numbers are correctly copied into the article. This will be graded as a minor template violation.

