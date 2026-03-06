# Review comments on Project 11, STATS 531 W26

Various methods are tested on a stock index. The conclusions are somewhat weak: they appear to support the development of an ARMA model that the results show is inferior. It is natural to demonstrate ARMA methods, since they are an important class of time series models and have featured heavily in the first half of the semester. However, when confronted with a situation where you find they are weak, you should not conclude that they are strong. Some specific points follow.

1. Abstract and conclusions. The stated goal is to develop a model, but a model should have a goal, not be a goal. Clarity about the purpose of having a model would help motivate the project and might help direct the analysis toward a purpose.

1. The decision to investigate data up to 2018 could also use motivation. What is the purpose of this kind of historical analysis?

1. Sec 1.0.1. Is this figure log(price)?

1. Sec 1.0.2 The ACF plot for undifferenced DAX is a waste of reader attention. We have already seen from Sec 1.0.1 that a covariance-stationary model for this is inappropriate.

1. Sec 1.0.3 The histogram of DAX (undifferenced) does not have value as an estimate of the probability density of any reasonable model, so it is best omitted.

1. Sec 1.0.4. ADF is not useful here. It tests for a unit root, which has already been investigated at this point. Other types of nonstationarity, including heteroskedasticity or nonlinear trends, it is not designed to see. In the supplement, 
the ADF test hypotheses are incorrectly stated. The supplement
writes H0 as “the process is nonstationary” with a distributional inequality condition, but the ADF
test specifically tests for a unit root, not for general nonstationarity. Stationarity and absence of a
unit root are not equivalent concepts.

1. Sec 1.0.5. "Returns are stationary, and repeat roughly every 3.3 stock measurements". It is not clear what this sentence means. The estimated spectral density is quite flat, so there is not clear evidence for cyclical behavior. This might be interpreted as possible indication of a small weekly periodicity.

1. Sec 2.0.1.1.2. Interpreting the QQ plot as showing a good fit for normality is extremely wrong. On this scale, it is hard to see, but there is already an indication of the long tails that would become clearer on a better plot.

1. Sec 2.0.2. SARMA is generally not considered plausible for stock indices. Substantial monthly or annual effects would violate the efficient market hypothesis.

1. Sec 4. It does not quite make sense that the project simultaneously considers the ARMA analysis to be successful and the GARCH analysis to be greatly superior. Better to spend less time on ARMA and explain GARCH in more detail.

1. Fitting ARIMA to raw stock prices is conceptually problematic. The project fits
ARIMA(3,1,2) to DAX opening prices and claims it is a good model because the predicted values
“overlap almost perfectly” with the actual data. However, this is misleading—an ARIMA(0,1,0)
random walk would also appear to track prices closely because each prediction is essentially the
previous day’s price. The near-perfect visual overlap is an artifact of the integrated (d = 1)
component, not evidence of model quality. The financial time series literature generally argues
that stock prices follow approximately a random walk, so the ARIMA(3,1,2) specification needs
stronger justification.

1. Sec 5.1. The null for ADF is a unit root for a linear process, not a general nonstationary model. This was explained in class.

1. Plots are not numbered and captioned. Log differences should be labelled as "return" not "price".

1. "Mean zero residuals" as a concluding accomplishment is not a big deal. Residual processes naturally have mean zero, in many situations including this, because the population mean has been estimated and subtracted. 