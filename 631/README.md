---
title: "STATS 631"
author: "Instructor: Edward L. Ionides"
output:
  html_document:
    toc: no
---


## Schedule

STATS 631 is new this year, so suggestions on how to make it work well are particularly appreciated.
In addition to all the activities of 531, we will arrange a one-hour meeting at a time we call all make, at which we will discuss a paper.
Everyone is expected to have something to say, for example, a fairly brief reading of the paper should be sufficient to have an opinion on one or more of the "generic questions" listed below.
You are expected to spend at least an hour reading the paper.
The primary intention here is that we learn about research in time series analysis by discussing some strong papers (old and new) which have helped to define the current state of the art. 

Meetings are Thursday 10:30-11:30 in 438 West Hall.

1. Week starting Jan 13: [__akaike74__](https://doi.org/10.1109/TAC.1974.1100705)
<br>
Hirotugu Akaike's information criterion (AIC) is used in many recent time series papers, including those later on our reading list.
Akaike developed his ideas in a sequence of papers, with time series analysis being one of his main motivations.
This paper is the first one which focuses on the current standard definition of AIC.
Akaike's foundational papers are 50 years old, and there has been much work on model selection since then.
Why have  Akaike's ideas been so persistent?  

1. Week starting Jan 20: [__box76__](https://doi.org/10.1080/01621459.1976.10480949)
<br>
George Box's work popularized the autoregressive integrated moving average (ARIMA) framework for time series.
Among many other notable contributions, he is responsible what is perhaps the most widely quoted advice for applied statistics, "All models are wrong, but some models are useful."
This influential discussion of the relationship between science and statistics, and the role of models in this relationship, is informed by Box's extensive work in time series analysis.
Look for places where dependence, including temporal dependence, play a role.

1. Week starting Jan 27: [__hyndman08__](https://doi.org/10.18637/jss.v027.i03)
<br>
Rob Hyndman's many contributions to time series analysis include the development of `auto.arima`, a widely used approach for choosing ARIMA models.
This paper explains the construction of this procedure and its motivation, as well as mentioning alternative approaches. 

1. Week starting Feb 3: [__taylor18__](https://doi.org/10.1080/00031305.2017.1380080)
<br>
This paper introduces a widely used modern forecasting tool, Facebook Prophet, implemented by the R package [__prophet__](https://cran.r-project.org/web/packages/prophet/index.html).
Facebook Prophet is not based on ARIMA modeling, and the difference between these approaches is worth consideration.

1. Week starting Feb 10: [__lim21__](https://doi.org/10.1098/rsta.2020.0209)
<br>
Deep learning has been influential throughout statistics, and time series analysis is no exception.
This review discusses the deep learning for time series, situated before the widespread popularization of transformers.

1. Week starting Feb 17: [__gruver23__](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eb7ca52e8207697361b2c0fb3926511-Paper-Conference.pdf)
<br>
When is GenAI useful for time series analysis? 

1. Week starting Feb 24: [__bjornstad01__](https://doi.org/10.1126/science.1062226)
<br>
There are surprisingly many important ideas about time series analysis for nonlinear stochastic dynamic systems discussed in this compact paper.
The issues it raises have prompted much work over the past two decades, and some issues remain unresolved.

Spring Break

8. Week starting Mar 10: [__doucet09__](http://www.warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/publications/dj11.pdf)
<br>
Particle filtering facilitates time series analysis for many nonlinear systems. We study a review of this technique by two leading experts.  

1. Week starting Mar 17: [__kristensen16__](https://doi.org/10.18637/jss.v070.i05)
<br>
Perhaps the main alternative to likelihood-based inference for partially observed Markov process (POMP) models is the locally Gaussian approximation used in the integrated nested Laplace approximation method.
A popular implemention is Template Model Builder, described in this paper.

1. Week starting Mar 24: [__stock21__](https://doi.org/10.1016/j.fishres.2021.105967)
<br>
Fishery management is a major economic and environmental task that is based on POMP models for time series data.
Template Model Builder is widely used in this context.
Would there be an advantage to using particle filter methods to avoid the approximations inherent in Template Model Builder, or does this class of problems expose the limitations of the particle filter?

1. Week starting Mar 31: [__wheeler24__](https://doi.org/10.1371/journal.pcbi.1012032)
<br>
A paper dealing with various practical issues in data analysis via mechanistic models, including residual analysis and benchmarking to help identify model misspecification.

1. Week starting Apr 7: [__subramanian21__](https://doi.org/10.1073/pnas.2019716118)
<br>
Many papers were written fitting mechanistic models to learn about COVID-19 transmission and to forecast its trajectory. This paper shows how it is critical to understand both the reporting process and the disease dynamics.

1. Week starting Apr 14: [__madden24__](https://doi.org/10.1371/journal.pcbi.1012616)
<br>
Integrating deep learning with mechanistic modeling for time series data, in the context of measles epidemiology.

## References for the STATS 631 reading group

[__akaike74__](https://doi.org/10.1109/TAC.1974.1100705).
Akaike, H. (1974). A new look at the statistical model identification. _IEEE Transactions on Automatic Control_, 19(6), 716-723. 

[__bjornstad01__](https://doi.org/10.1126/science.1062226).
Bjørnstad, O. N., & Grenfell, B. T. (2001). Noisy clockwork: time series analysis of population fluctuations in animals. Science, 293(5530), 638-643. 

[__box76__](https://doi.org/10.1080/01621459.1976.10480949).
Box, George E. P. (1976). Science and statistics. _Journal of the American Statistical Association_, 71 (356): 791–799.

[__doucet09__](http://www.warwick.ac.uk/fac/sci/statistics/staff/academic-research/johansen/publications/dj11.pdf).
Doucet, A., & Johansen, A. M. (2009). A tutorial on particle filtering and smoothing: Fifteen years later. _Handbook of Nonlinear Filtering_, 12(656-704), 3. 

[__gruver23__](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eb7ca52e8207697361b2c0fb3926511-Paper-Conference.pdf).
Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are zero-shot time series forecasters. _Advances in Neural Information Processing Systems_, 36.

[__hyndman08__](https://doi.org/10.18637/jss.v027.i03).
Hyndman, R. J. & Khandakar, Y. (2008) Automatic time series forecasting: The forecast package for R. _Journal of Statistical Software_, 26(3).  

[__kristensen16__](https://doi.org/10.18637/jss.v070.i05).
 Kristensen, K., Nielsen, A., Berg, C. W., Skaug, H., & Bell, B. M. (2016). TMB: Automatic Differentiation and Laplace Approximation. _Journal of Statistical Software_, 70(5), 1–21. 

[__lim21__](https://doi.org/10.1098/rsta.2020.0209).
Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. _Philosophical Transactions of the Royal Society A_, 379(2194), 20200209. 

[__madden24__](https://doi.org/10.1371/journal.pcbi.1012616).
Madden, W.G., Jin, W., Lopman, B., Zufle, A., Dalziel, B., E. Metcalf, C.J., Grenfell, B.T. and Lau, M.S. (2024). Deep neural networks for endemic measles dynamics: Comparative analysis and integration with mechanistic models. _PLOS Computational Biology_, 20(11), e1012616.

[__stock21__](https://doi.org/10.1016/j.fishres.2021.105967).
Stock, B. C., & Miller, T. J. (2021). The Woods Hole Assessment Model (WHAM): a general state-space assessment framework that incorporates time-and age-varying processes via random effects and links to environmental covariates. _Fisheries Research_, 240, 105967.

[__subramanian21__](https://doi.org/10.1073/pnas.2019716118).
Subramanian, R., He, Q., & Pascual, M. (2021). Quantifying asymptomatic infection and transmission of COVID-19 in New York City using observed cases, serology, and testing capacity. _Proceedings of the National Academy of Sciences_, 118(9), e2019716118.

[__taylor18__](https://doi.org/10.1080/00031305.2017.1380080).
Taylor, S. J., & Letham, B. (2018). Forecasting at scale. _The American Statistician_, 72(1), 37-45. 

[__wheeler24__](https://doi.org/10.1371/journal.pcbi.1012032).
Wheeler, J., Rosengart, A., Jiang, Z., Tan, K., Treutle, N., & Ionides, E. L. (2024). Informing policy via dynamic models: Cholera in Haiti. _PLOS Computational Biology_, 20(4), e1012032. 


## Generic questions

1. Which parts of the paper might be worthwhile for me to read in more detail, and why? If you see an immediate benefit to obtaining a better understanding of part of the paper, then you may spend extra time on it to the extent that fits into your schedule. 

1. What is the strongest part of the paper? i.e., something that the paper demonstrates which deserves to be widely known.

1. What is the weakest part of the paper? Is there a limitation that may make the paper less useful in practice, or even misleading. (This may be rare, or hard to find, in high-impact papers.)

1. Has the paper had an impact on statistical theory and/or methodology and/or applications? Why or why not?

1. Technical questions include: (a) why was the notation set up this way? (b) what steps need additional explanation to be clear to this reader?

1. Study the numerical results, figures and tables. To what extent do they support the conclusions of the paper?
    
## Grading policy

* Grading is on attendance and participation in a weekly 1-hour discussion.

* Minimal preparation for participation means spending one hour reading the paper and thinking about its contribution. This is a useful academic skill---we read many more papers superficially than we can read in detail.

* 631 students will also complete the same assignments as 531 students. The 631 meeting will count for 20% of the grade, and other components of 531 will be scaled accordingly.






