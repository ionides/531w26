## Review comments on Final Project 4, STATS 531 W26

 The report addresses a substantive methodologically question, and could be extended into a publishable project.
The AI acknowledgement is reasonable and consistent with the document.
However, it appears that "AI suggestions accepted only after human review" does not prevent hallucinated references.

1. When you introduce the data in section 4.2, it would be nice to talk more about what the data
represents? As it stands, the reader isn't told what a "fish recruitment index" or the "Southern
Oscillation Index" used as a covariate in the original paper actually means. These seem to
domain-specific terms, and the reader has to go to the original Shumway and Stoffer paper to
learn what they are.

1. The text on figure 3 is so small that it's unreadable
without zooming in. These also appear to be premade images, since there is noticable aliasing
in the PDF.

1. The project has a reason to include R, for comparison to TMB. It still demonstrates competence with the Pypomp methods taught in class. The project is reproducible, and the pypomp code is provided as separate files. Various figures and a good portion of the numerical results are loaded externally from files (m3_pf_sweep.csv and
m3_final_pf_summary.json ) or directly as images. It is not clear that all these files are present. Combining everything into a single qmd (as recommended in class) would have solved that. Next best is to have a complete reproducibility script that runs all the needed files from a single command.

1. Sec 1. This confuses TMB (a software library) with POMP (a class of models). For example, not all methods for POMP models are plug-and-play. Maybe the authors mean Pypomp, which is a library, but that also can do a range of inference methods.

1. There is an AI-fabricated reference:

   Valpine, Perry de, and Alan Hastings. 2002. “Review of Methods for Fitting Time-Series Models
with Process and Observation Error.” Bulletin of Mathematical Biology 64: 223–49.

   This may be hallucinated from two related papers:

   de Valpine, P. (2002). Review of methods for fitting time-series models with process and observation error and likelihood calculations for nonlinear, non-Gaussian state-space models. Bulletin of Marine Science, 70(2), 455-471.

   De Valpine, P., & Hastings, A. (2002). Fitting population models incorporating process noise and observation error. Ecological Monographs, 72(1), 57-76.


1. Define acronyms at first appearance, for example. TMB, POMP.

1. One weakness here is that the M2 particle filter log-likelihood has not converged. Table 2
shows that as J goes from 1000 to 20000, the mean estimated log-likelihood rises ~51 units.
This is a considerable shift, but the authors do note that this could be due to the downward
log-bias of the BPF which shrinks as J tends to infinity. But, the convergence from 10000 to
20000 is a shift of ~4 units which is still somewhat substantial for the 5-10 unit Laplace bias
that the authors are measuring. Given that the standard deviation at J=20000 is 3.74 and is
a good deal larger than the standard error of the mean, and that the mean is also visually still
trending upward in figure 3, it might be that 20000 particles isn’t sufficient to get convergence.
Maybe trying even more Monte Carlo with 50000 or even 100000 particles could make it even
more convincing, although it would be computationally expensive.

1. In Section 2.2, the paper says that Laplace should have a positive bias, but it does not fully
justify why the bias should be positive. The cited sources support the idea that Laplace can
be biased in nonlinear models, but they do not clearly show that the bias must go in this
direction for bilinear terms like $b_tX_{t−1}$. Later, the paper describes Laplace as “optimistic,”
meaning that it overestimates the likelihood, but this needs either a short explanation or a
stronger citation. Since Laplace bias can be positive or negative depending on the shape of
the posterior, the paper should better justify why overestimation is expected in this specific
model.


1. In section 3.3 and Table S1, the M1 parameter recovery is overstated. Both Section 3.3
and the Discussion claim that the M1 control experiment shows “both methods recover the
true parameters,” but Table S1 conveys different information. TMB gives $\hat\sigma_S= 0.007$ versus
a truth of $\sigma_S = 0.150$—--off by 95%, with the observation noise on S essentially collapsing to a
boundary solution.
The honest M1 takeaway should not be that parameter recovery is excellent, but
that TMB and the Kalman filter agree with high precision, and IF2 finds a similar likelihood region within PF noise.

1. In section 4.4, the off-diagonal cells are hard-coded constants. The off-diagonal
cells of Table 3 are hardcoded string literals in the QMD, appearing without any traceable
computation. Both −137.7 (Laplace at $\hat\theta_{IF2}$) and −141.2 ± 5.2 (PF at $\hat\theta_{TMB}$), the two cells
most central to the Laplace-bias argument—appear as raw constants on lines 360–367 of the
source. While the diagonal cells are computed at render time, the off-diagonals are not. The
Laplace-bias claim is therefore not reproducible from the submitted document.


1. Table 4 has some noticeable discrepancies, in the $c$ and $\sigma_S$ lines. These should be discussed or investigated. 

