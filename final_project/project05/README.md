## Review comments on Final Project 5, STATS 531 W26

1. There is no abstract. Figures do not have numbers and captions. References are cited at the end of the project, but not when used in the text. The expected standard of scholarship involves citation at the relevant point in the text. The generic acknowledgement of prior class projects should be made specific.

1. The AI statement is not credible. It says "we used AI for limited assistance with organization, revision and debugging. All statistical modeling, computation, verification of results, and final interpretation were done by us own." But, there is ample evidence that AI was used for much of the coding. For example, the references cited to STATS 531 teach a different methodology than is adopted in this project. No explanation is given justifying the actual methodology implemented. Large amounts of untested code were written implementing this unexplained and mis-referenced approach. **Whether or not AI was used, this is not a good approach.**

1. Sec 6.4.1. Using Nelder-Mead not a good choice.
`particle_filter_loglik` is a custom-made particle filter, rather than using Pypomp. 
The authors made various ad-hoc hard-coded "safety guards," perhaps required by their custom implementation.
The seed is fixed at 42 for all Monte Carlo calculations, leading to an unknown bias.
With the help of AI, it is not impossible to write out all the code without using packages, but then it is harder to make sure you are doing a correct and reeasonable analysis.
The authors suggest that maybe they should use iterated filtering, based on their experiences, but why use a deterministic optimizer for a Monte Carlo function if you think it is inferior to what was taught in class? One reason would be to compare with the class method to see if it is in fact superior, but that was not done.

1. Sec 6.4.1. "2. Core parameters are identifiable: ... show tight convergence across starts." That is not what the plots show. They show lack of tight convergence.

1. In Sec 5, there is not a probability of being in a state, but rather a weighted blend of states.

1. The model comparison results (Table 3) are hard-coded

1. "all converged trajectories reach log-likelihood values comfortably below the ARIMA baseline." should be "above"?

1. Fig 1. One of the trajectories seems to have likelihood higher than the presented MLE. 
The searches become very quickly flat and parallel, probably due to problems with the optimization. 


1. Particle initialisation does not match the stationary distribution. Particles are initialised
as $\nu_1, \nu_2 \sim N (0, 0.1^2)$. The stationary variance under the fitted $( \alpha_1, \sigma_1) = (0.993, 0.52)$ is
$\sigma_1^2 /(1 − \alpha_1^2)\approx 0.27/0.014 \approx 19$, so the initial cloud is two orders of magnitude tighter than its
long-run spread. With half-life ≈ 100 trading days for $\alpha_1$, the filter takes several hundred days
to “warm up”, and likelihood contributions in this period are not on the same footing as the rest.
A burn-in discard or a stationary-distribution initialisation would fix this.

1. The code reinvents a particle filter and a searching method. This method has problems, not least the particle filter is always carried out at the same seed. It may be correct, but it has not been tested; if you use high quality software, that software will have unit tests that provide some guarantee. Also, the course taught methods that avoid the substantial bias from fixing a seed, that allow parallelization, 

1. Table 4 shows substantial differences between observed values and simulated values, yet these results are interpreted as "Simulations reproduce the key stylized facts of the observed series."

1. Hardcoded ARIMA and GARCH results undermine reproducibility.  The ARIMA log-likelihood (-7712.48) and
AIC (15436.96) are hardcoded as literals in the model comparison table rather than being computed from a fitted model object within the document. Similarly, the GARCH
values (garch ll = -6774.15, garch k = 6, garch aic = 13560.30) are reassigned to
hardcoded constants in the comparison section, overwriting the values computed from
the fitted arch model object earlier in the notebook. If the data window changes (which
it will, since the data is downloaded live from FRED with no pinned end date), these
hardcoded values will be silently inconsistent with the re-estimated models.
The AIC comparison, which is central to the paper’s conclusions, therefore cannot be verified without independent re-estimation.

1. Stationarity constraint not reproducible. The paper describes a quadratic penalty on
`alpha_1` and `beta_2` exceeding 0.998, but no such penalty appears in the
particle_filter_loglik function shown in the document. The optimization wrapper
implementing this constraint is absent. Since a specific 2.3 unit cost is claimed, the
implementation must be included for the result to be reproducible.

v