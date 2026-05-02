## Review comments on Final Project 15, STATS 531 W26

The project is written to a reasonable standard of academic writing and reproducibility. However, some numerical results are hard-coded in the qmd file. Figures could use numbers and captions, like in the template.

1. Sec 3.2. The STL is not insightful in this case - by the end of STATS 531, we know how to do better.

1. Too much time is spend on ARMA and SARMA.

1. One of the main things that can be found from ARMA/SARMA is a good benchmark to assess model sppecification, but that is not done here.

1. It is not clearly explained that the ARMA models are fitted on a log scale, so the log-likelihood needs a Jacobian correction to be useful as a benchmark.

1. Sec. 3.7. The "profile likelihoods" look surprisingly neat. There may be something to dig out from further study of the source code, but no reviewer found a major issue in the code. Mathematically, we know there is a problem: all profiles should have the same maximized log-likelihood, since matheamtically all profiles pass through the MLE. There is 300 log units discrepancy here. The log-likelihood of 1600 is much higher than the reported MLE.

1. The model dynamics lack extrademographic stochasticity. 

1. The project was put in the context of previous STATS 531 flu projects, and addressed some previous concerns. However, too much time on linear analysis and failure to compare benchmarks continue to be a problem.

1. The global MLE gives $\mu_{RS} = 0.822$ (page 10).
That means the average time someone stays immune before becoming susceptible again is 1/0.822 ≈ 1.2 weeks. For influenza, immunity against the same
strain usually lasts months to years. A week and a bit makes no biological sense.
It is not just a weird number. At $\mu_{RS}= 0.822$/week, the probability of leaving the $R$ compartment in
a single sub-step is $1 − \exp\{−0.822/7\} \approx 0.111$, which means over half of recovered people are back in the
susceptible pool within a week. Thus, the SEIRS model is basically behaving like
an SIS model where R is just a brief pit stop.
This issue is not discussed. A profile would help, but is not provided.

1. The code fixes $N= 10^6$, $S_0 = 0.70$, $E_0 = 0.02$, $I_0 = 0.01$, $R_0 = 0.27$, and all five have perturbation sigmas of zero in the IF2 runs, so they never get updated. The methods section does not explain any of
these choices.
The $N = 10^6$  choice has real consequences, not least for the interpretation of reporting rate. 

1. Look into the gap between the IF2 traces and the pfilter evaluation. The local IF2 trace plot on page 9
shows log-likelihoods around −1200 to −1600 during the MIF runs, but the final pfilter evaluation gives −2565.89. A gap of 700–1000 units is large. There is always some difference between what MIF tracks
internally and the unbiased pfilter estimate, but this much is unusual. The authors should verify that
evaluate_theta_list is actually computing the log-likelihood at the right parameter values.

1. Providing units would be helpful, especially for time variables. Rates can be per day, per week, per year, etc. You need to know in order to interpret the result.

