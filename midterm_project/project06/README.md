# Review comments on Project 6, STATS 531 W26

It's hard to know how to evaluate this project because of a major flaw that turns most of the analysis and discussion into nonsense. The effort and skill that went into this project is appreciated. Nevertheless, the error is a result of careless coding that should have been avoided.

1. The code actually switches to a parallel analysis triggered by a `using_real_data = False` flag. 
Thus, the project does not in fact investigate the  true TESS light curve as mentioned in the report. The only indication in the report
that the data is simulated is the first figure title, which says simulated in parentheses. This is a massive oversight and the use of simulated data should have been reported more clearly.

1. Data: the text explains the data are downloaded for the hot Jupiter host star HD208458, but the code has `TARGET = "HD 209458"`.

1. It would be nice to be told what `flatten` does to detrend. Except, as discussed above, the actual presented analysis does not do this because of the  `using_real_data = False` flag.

1. The simulation code (line 101) injects a transit at period = 3.5 days. The proxy period-scan recovers 1.7354 days — roughly half, so it found a harmonic rather than the fundamental. The text, however,  describes "the dominant peak near 7.05 days," which does not appear anywhere in the plot, the printed output, or the simulation. That description was written for a different run and never updated. HD 209458b has a known orbital period of about 3.52 days, so the injected value is physically grounded — the issue is that the proxy BLS method, which is not a standard BLS and is never explained, fails to
recover it. Reconciling this, and explaining why a harmonic was picked up, would turn a weakness into an interesting methodological observation.

1. Viewing the first plot of the series in the data section, they discuss detrending the data. However,
there does not appear to be any large or obvious upwards or downwards trends in the data which
would require detrending. It does not seem to be a necessary step despite their claim that it clarifies
the underlying temporal structure. This conclusion is supported by the plots shown before and after detrending which appear identical.

1. Sec 3. "Two separated windows" : this is not clear from the plot.

1. Sec 4. There is a big peak at 3.4 days and a comparable (narrower but taller) at 1.7 days. It is unusual to plot period rather than frequency for a spectrum, since frequencies are equally spaced. The report prints a “best period” around 1.735
days, but later claims the dominant peak is near 7.05 days, and the phase-folded plot is labeled using the ∼1.735-day period. This needs reconciliation (units, alias/harmonic, search grid mismatch or a reporting/coding bug).

1. The figure (un-numbered) for phase-folded light curve: reported "pronounced dip at 0.5" that is "concentrated within a narrow phase interval" is not evident from the plot.

1. Sec 4.1. "transformed" how? Does this mean "detrended"?

1. Section 5 says the ARMA is fit after removing "conceptually" the transit signal, but no removal step is implemented in the code. The model is fit directly on the standardized residuals, which still contain periodic transit dips. This means the ARMA is partly capturing deterministic periodic structure rather than stochastic noise, which undermines the core goal of separating the two. 

1. Sec 5.1. This is too much unformatted output.

1. Sec 5.2. There appears to be pattern, including regularly spaced long-tailed observations, to the residual time plot that is not mentioned in the discussion. What is the explanation? 

1. The project is based on the required template, but is rendered to html not pdf, and perhaps as a consequence it is missing figure numbers.

1. The BLS method is not described; it is an important part of the analysis, and it should be explained and referenced.

1. Written to a reasonable standard of reproducibility, with minor weaknesses: absolute file references, and errors in in-line Python output.

1. Sec. 5. The Box-Ljung suggests very strong evidence of model violation, which is not discussed. Why show the test and not interpret it? These p-values don't seem qualitatively consistent with the residual ACF - is this a matter of practical vs statistical significance, or something else?

