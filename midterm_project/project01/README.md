# Review comments on Project 1, STATS 531 W26

A technically sound project with broad interest to the reader group.

1. Abstract and conclusions: with some care, this project argues that it is possible to identify an effect consistent with a political bias in Fed decisions. The project is suitably circumspect about this potentially interesting finding.

2. Is there an assumption that political conditions are exogenous to the economy?

3. Abstract. What does it mean that macro-economic conditions are "exogenous"? We have not defined that term in class. In econometrics, a quantity is exogenous if it can affect the system under study but not be affected by it. Sadly, putting a variable into the `exog` argument in `statsmodels` does not guarantee that it is well characterized as exogenous.

4. Sec 1.3. "This project reflects a more cohesive approach to time series modeling." What exactly does "cohesive" mean here? Be more specific.

5. Sec 3.1. The coloring on the AIC table is nice.  When an AIC comparison indicates a mathematical inconsistency, the error could be on either party, but the figure blames it on the bigger one. This is reasonable, assuming that the issue is primarily optimization not evaluation.

6. Sec 3.3. The long tails do indeed raise concerns about the use of a Gaussian ARMA. One could try ARMA with t-distributed errors, not covered in class.

7. The robustness check to the outliers (e.g., Volker) helps to reinforce the conclusions.

8. The DiffLog transformation, defined as $`\Delta\log(1 +Y_t)`$, poses conceptual challenges for modeling
interest rates. Since the federal funds rate is already expressed as a percentage, calculating the
difference of the log of (1 + rate) is unconventional and substantially obscures the economic
interpretation of the resulting coefficients. In macroeconomic modeling, policy rate changes are
typically evaluated in simple differences, such as basis points, rather than logarithmic ratios.

9. A residual plot, and/or ACF plot of the residuals, are missing diagnostics. They could go in a supplement if they don't show anything noteworthy.

