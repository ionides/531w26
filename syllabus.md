---
title: "Syllabus for STATS 531/631 (Winter 2026) <br><it>Modeling and Analysis of Time Series Data</it>"
author: "Instructor: Edward L. Ionides"
output:
  html_document:
    toc: no
---

## Course information

Course components will include:

* classes Mon/Wed, 2:30-4:00 p.m. in WEIS 260
* homework
* group midterm project, with individual peer review
* group final project, with individual peer review
* two in-class midterm exams
* For [631](631) only, weekly discussion of a research paper

Instructor contact information:

* email: ionides@umich.edu 
* web: dept.stat.lsa.umich.edu/~ionides
* office hours: Wed 4-5pm, Thu 2-3pm in 453 West Hall. 

GSI: Aaron Abkemeier

* email: aaronabk@umich.edu
* office hours: Tue, 4:30-6:00 PM, and Friday, 3:00-4:30 PM. Angell Hall, Room G219.


Computing support. If you have a coding problem you cannot debug, first try aksing AI and internet search and your colleagues. If the problem persists, develop a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) that others can run to help you. You can share this, and the error message you obtain.

Course notes and lectures are posted at https://ionides.github.io/531w26/ with source files available at https://github.com/ionides/531w26

Supplementary textbook: R. Shumway and D. Stoffer _Time Series Analysis and its Applications_, 4th edition (2017). 
A [pdf](https://link.springer.com/book/10.1007%2F978-3-319-52452-8) is available using the UM Library's Springer subscription.

Recommended pre-requisites:

* Theoretical and applied statistics. STATS 500 and prior or concurrent enrollment in STATS 511, or equivalent. STATS 413 and STATS 426 is sufficient in conjunction with a strong math and computing background. For review, see "Mathematical Statistics and Data Analysis" by J. A. Rice.

* Linear algebra. A certain amount of basic linear algebra will be required. For review, see 
[www.sosmath.com/matrix/matrix.html](http://www.sosmath.com/matrix/matrix.html).

Statistical computing background:

* We carry out data analysis using R. There is no formal R prerequisite, but we will be working with R extensively and so you should allow extra time for this course if you are new to R programming. If you are not familiar with R then it is helpful to have some background in Python or a similar language to learn R fairly easily. The R Core Team's  [Introduction to R](https://cran.r-project.org/doc/manuals/r-release/R-intro.pdf) may be useful for review or study. Another resource is the [Tutorial Introduction to R by King, Field, Bolker and Ellner](https://kingaa.github.io/R_Tutorial/).

-----------

## Course outline


1. Introduction to time series analysis.

2. Time series models: Estimating trend and autocovariance.

3. Stationarity, white noise, and some basic time series models.

4. Linear time series models and the algebra of autoregressive moving average (ARMA) models.

5. Parameter estimation and model identification for ARMA models.

6. Extending the ARMA model: Seasonality and trend.

7. Introduction to the frequency domain.

8. Smoothing in the time and frequency domains.

9. Case study: An association between unemployment and mortality?

10. Introduction to partially observed Markov process (POMP) models.

11. Introduction to simulation-based inference for epidemiological dynamics via the pomp R package.

12. Simulation of stochastic dynamic models.

13. Likelihood for POMP models: Theory and practice.

14. Likelihood maximization for POMP models.

15. A case study of polio including covariates, seasonality & over-dispersion.

16. A case study of financial volatility and a POMP model with observations driving latent dynamics.

17. A case study of measles: Dynamics revealed in long time series.

--------------

## Groups

* Groups for the midterm project will be randomly assigned, around the third week of classes.

* Groups for the final project will be re-randomized after the midterm project. 

-------------

##  Grading

* Weekly homeworks (15\%, graded following a [rubric](rubric_homework.html)). 
* A group midterm project (20\%, due 11:59pm on Friday 2/20). In special situations, you can request to write an individual project for the midterm and/or the final project. This may be appropriate if you have a particular dataset or scientific question that has motivated your interest in learning time series analysis. You can also ask your group if it is willing to join collaboratively on your project to make it a group project.
* Two individual anonymous peer review evaluations of other group midterm projects (5%, due 11:59pm on Friday 2/27), each about 500 words long.
Identification of strengths and weaknesses in the technical details is of primary importance, but comments on matters of style, organization and presentation are also welcome.
The reviews should include discussion relating to reproducibility of the project's numerical results.
* A group final project (30%, due 11:59pm on Tuesday 4/21).
* Two individual anonymous peer review evaluations of other group final projects (8\%, due 11:59pm on Wednesday 4/30), each about 500 words long.
Identification of strengths and weaknesses in the technical details is of primary importance, but comments on matters of style, organization and presentation are also welcome.
The reviews should include discussion relating to reproducibility of the project's numerical results.
* Participation (2%). In-class feedback questions; typically single-question Canvas quizzes that do not require any preparation.
* Two midterms (10% each), in class on Monday 2/16  and Wednesday 4/15.
* Course letter grades are anticipated to be mostly in the A, A-, B+ range customary for courses at this level. In the past, this has corresponded to overall scores of approximately 95% for A+, 90% for A, 85% for A-, 80% for B+. However, the exact cutoff used will be determined in the context of the course for this specific semester.

### Learning goals and the roles of AI

1. Learn to carry out a time series analysis using linear time series models in a situation where these are appropriate. The target is to write a midterm report that carries out a comprehensive, well-motivated and clearly explained data investigation.

2. Learn to carry out inference for a scientifically-motivated stochastic dynamic model using time series data. The target is to write a final report that carries out a comprehensive, well-motivated and clearly explained data investigation, usually involving inference for a nonlinear partially observed Markov process model.

3. Learn how to critically evaluate the strengths and weaknesses of a time series data analysis. This is carried out via peer review of midterm and final projects done by other groups.

4. Learn how to use AI to support practical applied statistics research in the context of writing and evaluating data analysis.

5. Learn how to follow scientific standards for citation of references, credit to sources, and construction of reproducible data analysis embedded in a scientific report. 

6. Build sufficient understanding of time series theory and methods to take full responsibility for decisions and choices made during data analysis, whether or not AI is involved.

To assess your progress at 6, there will be a mini-quiz in each class and two in-class midterm exams.

Practical skills for (1-5) will be trained via weekly homework. Homework will also help train for (6). Homework questions will clearly state which parts are considered foundational knowledge that may be tested in a paper exam.

Homework will not be graded for correctness. Answers are available online, even before the solutions are posted. Hints are available from GenAI. Homework will be graded based on completion and scholarship. 


### Grading credit for scholarship

**Demonstrated effort**. It is the student's responsibility to convince the grader that thought has gone into the homework.
If the solutions look too close to a source, or to GenAI output, the student should anticipate that and


** Attribution of sources**.

+ GenAI is a source. You are welcome to use it, but its role must be credited. Also, note that current GenAI can write poorly in some situations. You should be careful to edit and error-check any material written using GenAI. You take full responsibility for work submitted under your name.

+ Explaining how you used GenAI can also be part of your "demonstrated effort."

+ In group work, you are responsible for checking that the sources of your collaborators are properly documented. The whole group must take responsibility for material that the group submits. 

----

## Student Mental Health and Wellbeing

University of Michigan is committed to advancing the mental health and wellbeing of its students. If you or someone you know is feeling overwhelmed, depressed, and/or in need of support, services are available. For help, contact Counseling and Psychological Services (CAPS) at 734.764.8312 and  https://caps.umich.edu during and after hours, on weekends and holidays. You may also consult University Health Service (UHS) at 734.764.8320 and https://www.uhs.umich.edu/mentalhealthsvcs.

----------

## Acknowledgements

Many people have contributed to the development of this course, including all former students and instructors. See the [acknowlegements page](acknowledge.html) for further details.

-----------


