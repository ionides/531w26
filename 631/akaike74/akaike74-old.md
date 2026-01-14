---
title: "akaike74"
date: "Jan 24, 2026"
output:
  ioslides_presentation:
    smaller: no
    widescreen: true
    transition: "faster" 
---

----

* over 70,000 citations on Google Scholar

* Akaike has a sequence of papers leading to AIC and then investigating it. This paper focuses on time series and gives perhaps the clearest derivation of AIC.


----

```
png(file="akaike74-plot.png")
  x=seq(0,10,length=200)
  plot(x,y=10-(x-5)^2,ylim=c(0,20),ty='l',col="red",axes=FALSE,ylab="loglik")
  text(4.8,0.1,expression(theta))
  text(7.2,0.1,expression(hat(theta)))
  text(0,14.25,"A")
  text(0,10.25,"B")
  text(0,6.25,"C")
  box()
  lines(x,y=14-(x-7)^2)
  abline(h=14,lty="dotted")
  abline(h=10,lty="dotted",col="red")
  abline(h=6,lty="dotted")
  abline(v=5,lty="dotted",col="red")
  abline(v=7,lty="dotted")
dev.off()
```

----

<img src="akaike74-plot.png" alt="Picture explanation of AIC" width="500" height="500">

----

* A : loglik at MLE

* B : loglik at truth, estimates  $E_{\theta_0}[ \log f(X;\theta_0)]$

* A-B : expected over-fitted likelihood for this dataset is $\mathrm{dim}(\theta)/2$

* B-C : expected log-likelihood penalty for using $\hat{\theta}$ on a new dataset is the same as the over-fitting.

* C : estimates  $E_{\theta_0}[ \log f(X;\hat{\theta)}]$

----

* What are the alternatives to AIC?
    + BIC
    + WAIC
    + GCV

Some derivations are at

https://dept.stat.lsa.umich.edu/~kshedden/Courses/Regression_Notes/model_selection.pdf


----

"Although the present author has no proof of optimality of MAICE it is at present. the only procedure applicableto every situation where the likelihood can be properly defined and itis actually producing very reasonable results without very much amount of help of subjective judgement. The successful results of numerical experiments suggest almost unlimited applicability of MAICE in the fields of modeling, prediction, signal detection, pattern recognition. and adaptation."

* AIC penalizes model complexity so far as it affects expected predictive skill measured by log-likelihood. Additional parsimony may have scientific value.

