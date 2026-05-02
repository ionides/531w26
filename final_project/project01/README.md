## Review comments on Final Project 1, STATS 531 W26

A well-written project, conducted to the standards of a first draft of a scientific paper.
The central finding here, that $q$ is not identifiable from the available RSV sequence data, is stated openly and backed up by the profile likelihoods. Overall, the authors are successful at stating worthwhile conclusions but not over-stating.

1. A substantial technical concern is that the mechanistic model may be too complex for the
available data. The dataset has only about five years of weekly subtype counts, but the model
includes two strains, seasonal transmission, susceptibility, reporting rate, dispersion, cross-
immunity, and effective population size. This creates many parameter trade-offs. The report
identifies weak identifiability for q, but this problem likely also affects parameters such as initial
susceptibility, reporting rate, and effective population size.
If it is necessary to fit such a small dataset with a complex model, many parameters will have to be fixed based on scientific assumptions. 


1. The authors pull a concrete lesson from the Washtenaw County
project (w21), that a simpler ARMA benchmark beat the SEIR model there partly because weekly
reporting artifacts weren’t modeled, and use it to justify both the count-based benchmark and the
time-varying reporting rate $\rho(t)$. The connection to the Kent County project (w24) is also made
clearly, rather than as a citation-drop.

1. The claim that AI was used productively, but "AI was never used as a scientific authority" is appropriate and supported by the research product.

1. Fig 3. Shows highly noisy search - is there model misspecification or some other problem?

1. Fig 5. Monte Carlo adjusted profile confidence intervals could be applicable despite the noise.

1. Reproducibility is generally good; uses qmd with caching for practical reproducibility. Some text results are hard-coded when they should be read from computational results.

1. On the cross-immunity formulation and its cited support. The paper cites Bhattacharyya (2015) and White (2007) as the basis for the two-strain SEIRS model, but neither reference supports
the specific formulation used. Bhattacharyya (2015) studies cross-immunity between distinct
paramyxovirus species (RSV, HPIV-1/2/3, and hMPV) using recovered compartments with reduced
susceptibility in a standard two-strain SIR framework. The White et al. (2007) model is single-strain
entirely: it compares eight nested SEIRS variants for RSV seasonality, none of which involve a second
subtype. So neither paper addresses RSV-A versus RSV-B cross-immunity, and more importantly,
neither uses the force-of-infection scaling approach the project adopts.

1. In the model here, $q$ appears in $\lambda_A = \beta A(I_A + (1 − q)I_B)/N$  and scales how much $I_B$ drives transmission pressure on the shared susceptible pool $S$. This is a population-level coupling effect, not immune memory in recovered individuals. In the two-strain literature (e.g., Gog & Grenfell 2002,
Restif & Grenfell 2006), cross-immunity is more typically encoded through partial-susceptibility
compartments $R_i$, where the recovered class from strain $i$ carries reduced susceptibility to strain $j$.
The practical difference matters: the model as written cannot distinguish between genuine cross-immunity and correlated seasonal forcing that produces apparent suppression through susceptible
depletion. Within the scope of a 531 project, this is nevertheless an interesting and original modeling decision.

1. Log-likelihood comparison in the conclusions. The conclusions compare the partial immunity
model (−1159.3) against the no-immunity model (−1162.4) and note the difference is about 3 units.
But −1162.4 is the no-immunity local search best. The no-immunity global search, reported a few
pages earlier in the same paper, reaches −1155.9, which is better than the partial model’s global
best. Comparing global-search values for one model against local-search values for the other is
inconsistent, and it happens to reverse the apparent direction of the comparison. The authors are
correct that Monte Carlo noise makes the difference uninterpretable either way, but they should say
so using the same class of reference value for both models.

1. Seasonality in the model is not explained until the supplement. The report would be easier to read if the model equations could be presented in the main text. The seasonality is implemented by a random effect, in a somewhat original way. More discussion of the choice and its motivation would be appropriate.

1. The $\rho_{\mathrm{slope}}$ parameter is hard-coded rather than estimated, and its formulation is problematic. The text states that $\rho_{slope}$ "allows the reporting rate to increase linearly by 3% each year" but the code sets `rho_slope = 0.03 * rho0 / 52` (line 519). This is fixed at the initial value of `rho0 = 0.01`, giving `rho_slope ≈ 5.8e-6` per week. Over 250 weeks this adds only ~0.0014 to the reporting rate, a negligible change. If the intent is a 3% annual increase, this should be multiplicative or at least relative to the current reporting rate, not a fixed additive increment based on the initial guess. Additionally, this parameter is not estimated via MIF (it has `rw_sd = 0.0`), so the assumed reporting trend is never validated against the data.
