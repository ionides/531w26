## Review comments on Final Project 9, STATS 531 W26

Modeling the dominant serotype as a latent Markov chain is a creative idea and biologically grounded:
the 2014 outbreak was driven by DENV-1 and the much larger 2015 outbreak by DENV-2, so
successive years genuinely had different dominant strains, and some mechanism for that switch
belongs in the model. Whether a Markov chain on $\{1, 2, 3, 4\}$ is the right formalization is debatable,
but the motivation is real.

1. The probe diagnostics are a useful contribution. Comparing growth rate, residual SD, ACF1,
and peak size across 500 simulations against the observed values gives specific information about
what the model gets wrong, and the paper reports these results honestly even when they are
unflattering.

1. The qmd report provides good reproducibility. This helped the referees to find many problems that could be addressed in future work. The authors also make a reasonable effort to identify and explain limitations that they were aware of. 

1. The model is short of extrademographic process noise. 

1. GARCH for disease data is count-intuitive and probably a poor choice. GARCH is designed specifically for financial data, for which time-varying volatility is a well-established phenomenon.

1. The project has 15 pages, well over the 10-page limit.

1. The estimates of $\rho$ violate constraints. The text says $\rho\in (0, 1)$ is a reporting
probability, but the MLE giveŝ $\rho= 2.22$ and the global trace shows chains drifting up to $\rho≈ 20$.
A reporting probability above 1 is not interpretable. Either the parameter is actually a scaling factor and should be renamed, or it needs a logistic transform to enforce the bound.
High reporting rates correspond to low incidence (conditional on the data).
The data may prefer this because the dynamic model has no overdispersion, and very low incidence amplifies the demographic noise. Additionally, it compensates for the error in coding the accumulator variable.

1. The H accumulator isn’t accumulating. In `rproc` the line `H new = dN EI` overwrites `H` instead of adding to it. With `nstep=7` and `accumvars=("H",)`, weekly `H` is just the last day’s new infections, so it’s roughly 7x too small. Should be $H_{t+1} = H_t + \Delta N_{EI}$ .

1. The model has no way for adding susceptibles. For a longer time series, this could be a big problem. Even for two transmission seasons, gain and loss of immunity may be important.

1. $\kappa_1$ is fixed to zero to define a comparison group, but why is $\kappa_2$ also zero?

1. In the switching model, there is perfect cross-immunity: people in $S$ are susceptible to all serotypes, and after infection they move to $R$ and are protected from all serotypes. Also, after switching serotypes, the only thing that changes is the transmission rate, so the model describes previous type 1 dengue being transmitted into type 2, for example. Building the model is nevertheless a technical accomplishment demonstrating POMP modeling in Pypomp.

1. Dengue `imports` is applied 7 times per week,  once per `rproc call`, but rproc runs 7 times a week. It should be scaled by `dt`.

1. The transmission model has a phase, $\phi$, as well as both $\sin$ and $\cos$ terms. Here, $\phi$ is redundant. 

1. There is a coding error in `rproc`: the key split produces five subkeys, but `key_IR` is used twice. This may not have a big effect in practice.

1. The serotype offsets for DENV-1 and DENV-2 are both fixed at zero. This is a consequential modeling decision that directly blocks the scientific question from
being answerable. $\kappa_1 = \kappa_2 = 0$ and both have `rw_sd = 0.0` throughout the estimation. Since the
2014 outbreak was dominated by DENV-1 and 2015 by DENV-2, the model is supposed to capture
a transmission difference between those two years via the serotype switch from $K = 1$ to $K = 2$.
But if both serotypes carry the same $\kappa$, switching between states 1 and 2 has no effect on $\beta(t)$,
and the Markov chain contributes nothing to the likelihood when the dominant serotype is either
DENV-1 or DENV-2. The paper cannot answer whether serotype switches explain the back-to-back
outbreaks because the model is not set up to express such a difference.

1. Overdispersion parameter $k$ is fixed and never estimated. The value $k = 40$ is set in the
initial parameter dictionary with `rw_sd = 0.0` and left there. The probe diagnostics show the
model underestimates residual noise in the data, which suggests the measurement model may be
wrong, but since $k$ was never estimated there is no way to know if adjusting it would help. It could be estimated.

1. No prior 531 projects are cited. The scholarship section is one sentence. The w24 archive
has directly relevant entries: a Taiwan COVID-19 analysis (project 13) and a Kent County SEIRS
with identical inference structure (project 12).

1. Inconsistency between model description and implementation for $I_0$ and $E_0$ without justification. The report text (Section 3.1) explicitly states:
“$I_0$ and $E_0$ are also treated as free parameters.” However, both have random-walk perturbation
scales of zero in the code, meaning they are never estimated and remain fixed at 400 throughout.

1. The report acknowledges AI use in a single sentence:
“ChatGPT and Claude were used in the coding process.” This gives no indication of how AI was used.

