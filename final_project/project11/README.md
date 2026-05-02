## Review comments on Final Project 11, STATS 531 W26

This is an ambitious project, with novel aspects, but there are quite a few implementation issues.
There is no abstract. The first sentence of the introduction is poorly written, since it appears to suggest that influenza is especially relevent to Florida. Figures should have numbers and captions. In Section 2.0, the descriptive statistics of the raw data have little value on top of the time plot.

1. Sec 2.1 The period of 181.67 has units of weeks, not days, since the data are weekly. Also, the expected dominant period for flu should be near 52
weeks and this is not discussed.

1. It may not be necessary to spend so long on the linear models, if the mechanistic models are the main contribution.

1. SIR without births is inappropriate for an endemic disease with yearly outbreaks; where do the new susceptibles come from?

1. Table 3. All the mechanistic models here have major problems, indicated by their log-likelihood extremely far below the benchmark. The main problem is the lack of births and/or loss of immunity. Overdispersion may also be needed to explain data.

1. Sec 5.2.1 shows something wrong with the log-likelihood, which is apparently constant at an unfeasible value (order $-10^{38}$). This shows a bug that needs to be fixed, or possibly an entirely implausible modeling assumption.

1. The seasonal period in SARMA looks wrong. The model uses a seasonal lag of 12 weeks but weekly flu data has an annual cycle of around 52 weeks. A 12 week lag picks up a quarterly pattern with no clear epidemiological meaning for flu. This seems like an error in the specification.

1. The population size in the model appears to be 2.3 million, not 23 million. The introduction
mentions Florida has around 23 million people, but the SEIR traces show N fixed at $2.3\times 10^6,$ and Section 5.3.1 explicitly reports N as 2300000. If the model ran with N at 2.3 million the transmission dynamics are calibrated to the wrong population.

1. Sec. 7. The profile likelihood intervals have zero width. Section 7 reports 95 percent confidence intervals like [18, 18] and [0.1, 0.1], which are single points. The profile was likely not computed correctly or the parameter range was too narrow. The plots are also too small to read in the submitted PDF.

1. The SEIHDR model is applied to flu without enough justification. Section 5.4 borrows a COVID-19
model from Wen et al. 2024 on the basis that flu and COVID-19 share transmission similarities. It
gives the worst fit of any model tested (-8,256,150) and there is no discussion of what that tells us.

1. The lack of cases in recent years seems surprising, e.g. <https://medicalxpress.com/news/2025-02-flu-florida-doctors.html>. After low flu during the pandemic, one expects it to have rebounded. It's hard to check whether the presented data are what they claim to be. Inspecting the time series, we see
   ```
   data = pd.read_csv("FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv")
   time_series = data['TOTAL SPECIMENS']
   ```
   The data are titled "US Influenza by week", which is a confusing mistake since the data file is clearly Florida. However, `TOTAL SPECIMENS` counts the number of tests, not the number of positives. Also, the number of tests differs wildly, and apparently there have been fewer tests recently. Other datasets, such as "hospitalizations for influenza-like illness" might be more reliable measurements of flu incidence. Although this is a scientifically serious problem, the authors get credit for sufficient reproducibility that it could be tracked down.

1. Sec 11.2. The null for ADF is a linear, Gaussian unit root process. Rejecting this is not good evidence for claiming to have demonstrated "stationarity" in this situation.

1. Sec 11.3.4. This supplementary analysis is unclear. How is it relevant? The "cases" in this figure look unlike the original data.

1. For the POMP models there are no diagnostic plots at all.

1. The project is 1 page over-length

1. While the text in section 3 claims to fit an ARIMA(3,0,2) model, the code seen in the .qmd (line 313) actually fits a different model, ARIMA(2,0,2).

1. The trigonometric SEIR model uses the R (recovered) compartment in the measurement model instead of I (infected). 
In `final_seir_poisson.py`, the `dmeas` and `rmeas` functions use `R = X_["R"]` and compute `mu = rho * R`. This means the model observes reported cases as a fraction of the *recovered* population, not the infected population. Epidemiologically, cases should be observed as a function of the infected (I) or newly infected (incidence) compartment. Using the recovered compartment, which grows monotonically, would produce an observation model that increases without bound rather than showing the seasonal peaks and troughs present in the data. This error likely contributes to the extremely poor likelihood of -3.403e+38.

1. In the unmodified SEIR model, the H compartment accumulates `dN_SE` (new exposures) rather than `dN_EI` (new infections), so it tracks cumulative exposures, not cumulative cases. Since H is never reset to zero (it is not listed in `accumvars=()`), it grows without bound, making it unsuitable as a basis for the observation model without proper accumulation variable handling.
