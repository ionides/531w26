## Review comments on Final Project 16, STATS 531 W26

A clear abstract, combining technical detail with a qualitative explanation of the discoveries from the data analysis.

1. Fig 4. There is quite a large spread in log-likelihood values, with searches getting stuck in that appear to be local maxima. 

1. Table 2. It would be nice to have scientific units for these quantities.

1. Sec 3.4. The authors correctly conclude that this loss of likelihood is a big price to pay for the mechanistic fit. The probe analysis is a nice way to identify things that could be fixed in the model. Ditto the log-likelihood anomaly analysis.

1. The authors propose that extreme observations may be a major explanation of the likelihood shortfall, but it is possible that model misspecification plays a role as well, since the performance
gap is quite large to be explained by only a few extreme observations. Also, Fig 6 suggests that many small negative CLL values are as important as a few large ones - the latter are somewhat balanced between positive and negative. 

1. Readers reported successful reproducibility from the provided code.

1. Acronyms should be identified at first occurrence, e.g., MLE, CLL, PBL, ARMA, PM are all undefined in the abstract.

1. The authors fix the initial state to an arbitrary constant,
which in return is artificially constraining the variance of the early trajectory and biases the
likelihood of the early data.

1. Profile likelihoods and confidence intervals would strengthen the report.

1. A reproducibility problem: I can inspect the POMP and regenerate the CSV summaries, but I could not find
the code that fits the standalone ARMA model, the regression plus ARMA model,
or the conditional log-likelihood anomaly file. Since a central conclusion is that
the POMP is comparable to the standalone ARMA model but inferior to the
regression plus ARMA model, the benchmark-fitting code should be included.

1. The conditional loglikelihood diagnostic is a good addition, but the interpretation in the report is
too narrow. The report says the largest negative anomalies correspond to sudden PM spikes
that the mechanistic model cannot anticipate. The saved anomaly file does not support this
generally. The largest negative anomalies include 2024-09-21, 2023-12-12, and 2022-06-07,
with observed PM2.5 around 1 to 2 micrograms per cubic meter, not spikes. These appear to be
unusually low days for which the POMP assigns very small probability. There are also large
positive anomalies following some of these events. This suggests the issue may be more about
abrupt regime changes, extreme low values, particle degeneracy, or model inability to handle
sharp drops and rebounds, rather than only unexplained high-pollution spikes. The proposed
extension of a regional background or event state is sensible, but it should be motivated by a
careful inspection of both high and low outliers.

1. The report says `X_0` is fixed to the first-week observed mean, approximately 10 micrograms
per cubic meter, but the code fixes `X_I_0_FIXED = 10.0`, while the first seven observations in
the supplied data average about 27.4 and the full-series mean is about 27.2. The effect may dissipate after the first few days, but will bias model parameters toward regions of parameter space with higher dissipation. 

