# Review comments on Project 3, STATS 531 W26

This project tells a coherent story, and is broadly correct, but has many small weaknesses in execution.

1. The units of recruitment and SOI are unclear.

2. The project specification required use of a project template that uses the Python engine. This project uses the R engine. That is not critical for scientific validity, since the core concepts are language independent. Howver, it does decrease the relevance of this project to the STATS 531 W26 class, which is taught in Python, and therefore less well written for the target audience. Specifically, it avoids discussion or demonstration of technical issues specific to time series analysis in Python that were studied in class. This will be graded as a violation of the template. 

3. Fig 3. The PACF shows slower decay after differencing, not quicker decay as reported.

4. Sec 4.1. "An augmented Dickey–Fuller test confirms stationarity (p < 0.01)." ADF is a test for a unit root in a linear Gaussian time series model against the alnative of a linear Gaussian time series model with no root. It is not a general test for stationarity.

5. Sec 4.2. AIC is not directly comparable between levels of differencing. Discussing it in the context of selecting $d$ and $D$ is an error.

6. Section 4.3: The Ljung-Box test does not add much to the analysis. As discussed in class, if you select a model with good AIC, it almost always has satisfactory L-B because they measure very similar things. Similarly, they will miss similar things.

7. Sec 5. The AIC here is comparable to the undifferenced SARIMA AIC (-146). Here, you don't have seasonality. So, seasonality is more important than SOI? You could include both.

8. Sec 5. In the univariate modeling section, the authors find that the log recruitment series requires both first order and seasonal differencing
to achieve stationarity. Yet, in the subsequent regression section, they model the exact same log recruitment series using a stationary ARMA error structure without any differencing. This is a technical contradiction. If the univariate series truly possesses a stochastic trend requiring differencing, the regression model must also account for that nonstationary behavior.

9. The discussion of the limitations of regression with ARMA errors is well supported.

10. Fig 7 shows heavy left tails on the log scale, and we are not shown the distribution on the natural scale. The log scale is quite natural here, but it is not carefully checked and there are some signs it may behave poorly. 

11. Supplementary Fig 10 is shown but not referenced in the main text.

12. The results in Table 3 (Supplement) were described in the report, and an in-text
reference would have been helpful to connect it with the description.

13. If you compute a raw CCF on any strongly autocorrelated series and then use peaks to motivate a lagged regression, that is potentially problematic unless you prewhiten it beforehand. Without prewhitening, CCF structure can reflect internal persistence and seasonality rather than any meaningful relationships. This point goes beyond what was discussed in class.

14. Section 1.1: The authors reference covariate regression from W24 project on
Detroit precipitation and a peer review comment on lagged effects from the
project. After a quick read, I do not observe covariate analysis or any comments
on lagged effects in the peer review for this W24 project. It is possible that the
reference was supposed to be for the W25 Lake Malawi water level project.

15. Section 1: A reference for SOI would have been helpful for the reader.

16. Define acronyms and abbreviations before using (e.g. ENSO, Section 3; similarly, it was unclear that TMB was a reference to an R package).

17. Section 4.1: Stating that the ADF test confirms stationarity is a bit strong. More precisely, the results “reject the null hypothesis of a time-homogeneous linear process with a unit root.”

18. This project re-analyzes a time series that is studied extensively in one of the course textbooks. That makes it quite hard to convince the reader that all the analysis and interpretations are independent of the course textbook. Perhaps this was not a good choice of dataset?

19. The overall conclusion is a negative finding - the role of SOI is unclear from this analysis. Since SOI is generally considered a major ecological phenomenon in the south pacific, that failure might likely be a shortcoming of the model rather than a scientific discovery. The authors show some understanding of this, but could the relationship be found in the context of a different specification of a regression with ARMA errors? 
