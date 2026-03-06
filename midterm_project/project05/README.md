# Review comments on Project 5, STATS 531 W26

This is an ambitious project. The extensive analysis comes at the expense of attention to detail. The reviewers worked hard to track down various problems, but these are the responsibility of the author. This is an interesting and thought-provoking project which has value despite its substantial flaws.

1. Sec 2.1. The coefficient of 1.26 is hard-coded (not reproducible) and the p-value appears to be from OLS assuming uncorrelated errors, which should not be reported in this situation.

1. Sec 2.1.1. ADF test is not well explained. What test? What test statistic and p-value? In various places, rejectign ADF is described as confirmation of stationarity, which is not correct; it simply rejects a unit root (in a linear Gaussian  time-invariant time series model, with additional constraints determined by undescribed arguments to the specific version of ADF used).

1. An ADF test on residuals from a regression requires special attention to
detail, so a formal cointegration test may provide more insights than an ADF test on the residuals. Simply doing an ADF test on these residuals may require you to test on different critical
values (eg. choosing a different $\alpha$). This issue is not discussed.

1. Sec 2.1.1 Finding the AR coefficient close to unity is hard to reconcile with ADF rejecting a unit root.

1. Sec 2.1.1 finding a constant close to zero is a necessary mathematical consequence of havihng removed a constant in the previous regression. It says nothing about agreement with theory.

1. Sec 2.2. The AIC table has many mathematical inconsistencies. This should be identified and discussed.

1. Sec 2.3. The ADF statistic is hard-coded. There are various arguments to the ADF test, so what is the justification for the choices made? Do these choices affect the conclusions?

1. Figure 3 and Sec 2.2. The Student-t ARMA model is an interesting development but there is little explanation of how the likelihood is calculated. From the code, it is a conditional log-likelihood, dropping a numnber of initial points that depends on the order of the model. Thus, these are not directly comparable within the table, or between the table and the statsmodels Gaussian ARMA log-likelihood. Introducing new methods is interesting, but must be done carefully to give clear and correct results.

1. Sec 2.4 A graphical demonstration of volatility clustering might be simpler and more persuasive for the reader than an undescribed test.

1. Sec 2.4.1. It is not clear what $`X_t`$ is here. Is it the ARIMA residuals? In that case, there is no need for $`\mu`$. Is it the regression residuals? In that case, there is clear dependence so GARCH is not suitable. We have to go into the source code to find it is the latter.

1. Sec 2.4.3. Earlier, in sec 2.4.0, you interpreted a p-value close to zero for the ARCH-LM test as evidence for heteroskedasticity, here you interpret it as evidence against it. Since you don't say what the test is, it is hard to be sure which is right, but how can they both be right? The code for the second application seems to be missing from the file, making it hard to check. By reading more about the test, and seeing the error in the analysis concerning removing a conditional mean before fitting GARCH, it becomes clear that it is the second that is wrong.

1. Sec 2.4. The ARCH-LM test is run on the raw spread instead of on the AR(1) residuals. When the mean has not been removed, the test picks up both mean effects and volatility effects at the same time, so it does not show cleanly that a GARCH term is needed. On top of this, the GARCH model is fit on the raw spread with just a constant mean, which conflicts with the AR(1) mean already estimated. The paper ends up with two different mean models that are never reconciled. To address this, it is necessary to fit the mean and volatility together in one ARMA-GARCH model.

1. Sec 3. The trading strategy investigation is a nice advance on previous STATS 531 projects, examining the consequences of the proposed model for a plausible strategy. Nevertheless, the backtest may have weaknesses. The fitted model uses a fixed hedge ratio, but the backtest uses a rolling
one, so they are working with different spread series. The GARCH volatility estimate is never used
in the trading signal or position sizing, which makes the whole GARCH section feel disconnected
from the rest of the paper. The return figure of 36.4% is also hard to interpret: the P&L is computed
from spread differences, which are differences of log prices, not dollar amounts. Turning that into a
percentage return would require knowing the dollar size of each leg, which is not specified.

1. Further, the trading strategy appears not to involve the fitted AR(1) model, except that it uses a hardcoded `phi = 0.99`, which came from the AR(1) coefficient.

1. The report uses hard-coded inline results, which is a reproducibility weakness. Othewise, the code appears to be available.

1. There is no attempt to put the results into the context of previous 531 projects.

1. The range of methods used was a double-edged sword, as they went well beyond the material in the course, and without many cited sources (or maybe just poorly cited) it was hard to understand everything that they were able to achieve by the end.

1. There are supplementary analyses mentioned in the text but not shown. These could be included in an appendix.

1. One reader had a technical suggestion for the trading strategy: "The plotted “returns” are essentially Position*(Change in Spread). This change in the fitted residual
is being treated as actual returns. That is, the underlying prices could stay the exact same, but
moving to a more or less volatile day than the beginning of the window would adjust the computed
returns."

1. Another major technical concern is that the presented trading strategy should not be considered because it has look-ahead bias. Look-ahead bias refers to the idea of using future
unavailable information in predictions. This occurs within the computed Z-score statistic
because the equation:
    ```math
    Z_t = \frac{\mathrm{Spread}_t - \mu_{t,w}}{\sigma_{t,w}}
    ```
    captures the day’s closing information (captured within $`\mu_{t,w}$ and $\sigma_{t,w}`$) to execute trades within the same day. In other words, the authors accidentally capture the future closing prices in
    their calculations of mean and standard deviations, using them execute trades on the same day. This could have caused inflated return estimates.

1. The project could have implemented a few more ideas from the course contents, such as the periodogram; even if this is unlikely to make it in the main body of the paper, maybe it could have ended up
in a supplementary material section, if only to
say something like “We tried XYZ and found no evidence of periodicity, but the results are shown in the supplementary material”.

1. The authors’ two-step method of running
OLS regression and then using an ADF test on the residuals is exactly the Engle-Granger
procedure, yet they fail to mention this. Including this in the report may have both boosted
the credibility of this project while assisting with the ordering, limitations, and conclusions
around this procedure.
To complete that argument, it is also necessary to check that the series are each well modeled using a unit root process before the regression.

