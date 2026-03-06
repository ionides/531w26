# Review comments on Project 9, STATS 531 W26

A comprehensive time series analysis on a substantive topic. 
The reviewers found much room for improvement, but that is the point of the exercise!

1. Section 3.1, the authors say: “The variance is all of equal scale. Visually, we conclude that the series is covariance stationary, but not mean stationary.” This statement is contradicted by the time plot on Page 3. There is evidence of heteroskedasticity. 

1. The amplitude of cycles is also increasing. For this sort of exponentially growing series, a log transform may be appropriate.

1. Sec 3.2. The ADF and KPSS tests sadly will not tell you if a log transform is appropriate. The differenced time series shows heterskedasticity (higher variability later in the time series) which is not detected by ADF.

1. Sec 3.2. the time plot shows a clear increasing trend, indicating nonstationarity, yet the team still conducted an ADF test and KPSS test. Half a page is spent describing the tests and their results, although ultimately they end up confirming the nonstationarity and provide no additional information. Similarly, they show a plot of the residuals and perform an ADF test on the residuals, when only including the plot would have been sufficient.

1. Sec 4.2. The authors state, “Some coefficient estimates should be interpreted cautiously due to potential
near-singularity of the covariance matrix.” A warning of near-singularity here points to the ill-conditioned
information matrix. This implies that the parameter estimates can be extremely changed by very slight
variations in the data, thus large standard errors and unreliable inference can result. The warning is brought
up but ignored without any follow-up.

1. The chosen SARIMA $(0, 1, 3)\times (2, 1, 2)_{12}$ model fits seven parameters. Checking the coefficient table, only two
parameters are significant at conventional levels, and one is marginal, which casts doubt on the overfitting
concern. The authors should have contemplated a simpler model. The authors very briefly acknowledge this,
but do not address it in their primary analysis.

1. Sec 4.3. Residuals have an artifact of high values at early lags (here, apparently, up to one seasonal cycle) in the statsmodels implementation of SARIMA.

1. Sec 4.3. ADF on SARIMA residuals is inappropriate. ADF tests for a unit root which, by construction, would have been removed by fitting SARIMA. Any remaining nonstationarity must be of some other type, which ADF would not find.

1. This project lacks a discussion connecting to prior 531 projects and lack of citations throughout the report. For example, three previous projects analyze electricity consumption: projects 6 and 12 from Winter 2024 and project 48 from Winter 2020. While it does not appear any of these projects used the same data or had a focus on Texas, it should not be on the reader to
determine how this project is unique from previous projects. In addition, there are only 2 citations:
one for the data source and one providing background information in the introduction. There are
zero citations in the methods or analysis section, which could have been useful in justifying why
they chose a model based on AIC, why they plotted the ACF plot of the residuals, and why they
differenced the data.

1. A frequency domain analysis could be provided, at least as a supplementary result. Quite likely, it would not show any surprising periodicities, but it is worth looking.

1. Limited references: the bibliography contains only two references. No previous 531 projects are cited,

1. Some discussion of the roots of the fitted SARIMA models could be useful, and would demonstrate mastery of that part of the course.

1. The main body of the project goes over 8 pages, thus, it is not in line with the assignment formatting
guidelines. Conciseness is a professional skill. This will be graded as a minor template violation.

