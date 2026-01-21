---
title: "Syllabus for STATS 531/631 (Winter 2026) <br><it>Modeling and Analysis of Time Series Data</it>"
author: "Instructor: Edward L. Ionides"
output:
  html_document:
    toc: no
---

## Course information

__Course components__.

* Classes, Mon/Wed, 2:30-4:00 p.m. in WEIS 260. Notes are posted at https://ionides.github.io/531w26. A provisional outline of topics for each class is [posted](cal/cal.pdf).
* Homework
* Group midterm project, with individual peer review
* Group final project, with individual peer review
* Frequent mini-quizzes for self-evaluation and feedback
* Two in-class midterm exams
* For [631](631) only, weekly discussion of a research paper

__Instructor contact information__. email: ionides@umich.edu. web: https://ionides.github.io/. office hours: Tue 1-2pm, Wed 10-11am in 453 West Hall. 

__GSI__, Aaron Abkemeier. email: aaronabk@umich.edu. office hours: MF 5:30-6:00. Location: G219 Angell Hall

__AI policy__. Appropriate use of AI is encouraged for projects and data analysis homework. Uses of AI can be effective or ineffective, as well as honest or dishonest, and we seek honest and effective use of AI. The midterm exams are pencil and paper, and will test foundational knowledge and reasoning skills without AI support. Some homework problems will be labeled as exam preparation and should be mastered without AI support. Mini-quiz problems will be drawn from the same problem pool as the midterm exams.

__Course GitHub site__. Documents, code and data are posted at available at https://github.com/ionides/531w26.

__Use of GitHub issues and pull requests__. Discussion on course material or course assignments can be generated or followed via [GitHub issues](https://github.com/ionides/531w26/issues). Set `Watch` on the top right of the GitHub repository to `All Activity` to make sure you are notifed of all posted issues. Pull requests that fix bugs and errors on the course website are especially appreciated. Up to 2% extra credit is available for contributions to the course git repository. If you have a contribution to make but are not yet comfortable with git, the GSI and instructor will be happy to teach that.

__Supplementary textbooks__\
1. R. Shumway and D. Stoffer, _Time Series Analysis and its Applications_, 4th edition (2017).  A [pdf](https://link.springer.com/book/10.1007%2F978-3-319-52452-8) is available using the UM Library's Springer subscription.\
2. C. Huang and A. Petukhina, _Applied Time Series Analysis and Forecasting with Python_ (2022). A [pdf](https://link.springer.com/book/10.1007/978-3-031-13584-2) is available using the UM Library's Springer subscription.  

__Prerequisites__. STATS 500 and prior or concurrent enrollment in STATS 511. For undergraduates, STATS 525 or 510, in conjunction with STATS 413 and STATS 426. For review, see "Mathematical Statistics and Data Analysis" by J. A. Rice.

__Statistical computing background__.  We carry out data analysis using Python. There is no formal programming prerequisite, but we will be working with some advanced statistical computing by the end of the semester. Statistical computing training to the level of DATASCI 507 is helpful. 

__Computing support__. If you have a coding problem you cannot debug, first try aksing AI, internet sources, and your colleagues. If the problem persists, develop a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) that others can run to help you. You can share this, and the error message you obtain, via a GitHub issue or otherwise.

-----------

## Groups

* Groups for the midterm project will be established around the third week of classes. Groups will be self-assigned, of size 3 (preferable) or 2 (acceptable).

* Groups for the final project can be the same or different.

-------------

## Grading

* Weekly homeworks (15\%, graded following a [rubric](rubric_homework.html)). 
* A group midterm project (20\%, due 11:59pm on Friday 2/20). In special situations, you can request to write an individual project for the midterm and/or the final project. This may be appropriate if you have a particular dataset or scientific question that has motivated your interest in learning time series analysis. You can also ask your group if it is willing to join collaboratively on your project to make it a group project.
* Two individual anonymous peer review evaluations of other group midterm projects (5%, due 11:59pm on Friday 2/27), each about 500 words long.
The primary task of peer review is to identify strengths and weaknesses in the technical details; comments on matters of style, organization and presentation are also welcome.
Reviews should include discussion relating to reproducibility of the project's numerical results.
* A group final project (30%, due 11:59pm on Tuesday 4/21).
* Two individual anonymous peer review evaluations of other group final projects (8\%, due 11:59pm on Wednesday 4/30), each about 500 words long.
The primary task of peer review is to identify strengths and weaknesses in the technical details; comments on matters of style, organization and presentation are also welcome.
Reviews should include discussion relating to reproducibility of the project's numerical results.
* Participation (2%). In-class feedback questions; typically single-question Canvas quizzes that do not require any preparation.
* Two midterms (10% each), in class on Monday 2/16  and Wednesday 4/15.
* Course letter grades are anticipated to be mostly in the A, A-, B+ range customary for courses at this level. In the past, this has corresponded to overall scores of approximately 95% for A+, 90% for A, 85% for A-, 80% for B+. However, the exact cutoff used will be determined in the context of the course for this specific semester.

### Learning goals and the roles of AI

1. Learn to carry out a time series analysis using linear time series models in a situation where these are appropriate. Write a midterm report that carries out a comprehensive, well-motivated and clearly explained data investigation.

2. Learn to carry out inference for a scientifically-motivated stochastic dynamic model using time series data. Write a final report that carries out a comprehensive, well-motivated and clearly explained data investigation, usually involving inference for a nonlinear partially observed Markov process model.

3. Learn how to critically evaluate the strengths and weaknesses of a time series data analysis. This is carried out via peer review of midterm and final projects done by other groups.

4. Learn how to use AI to support practical applied statistics research in the context of writing and evaluating data analysis.

5. Learn how to follow scientific standards for citation of references, credit to sources, and construction of reproducible data analysis embedded in a scientific report. 

6. Build sufficient understanding of time series theory and methods to take full responsibility for decisions made during data analysis, whether or not AI is involved.


### Grading for scholarship

Homework will be primarily based on scholarship, not correctness. Answers are available online, even before the solutions are posted. Hints are available from AI. In practice, that means that the grading will emphasizes scholarship, i.e., the process by which the solution was obtained and the explanation of this. Scholarship is also an important component of the grading for projects.

**Attribution of sources**. Explaining what sources were used, and where, is central to scientific work. It ensures you get credit for your own contribution but not somebody elses. It also facilitates fact-checking, and tracking down the source of any error that might arise. Failure to attribute sources may lead to deduction of points and can become academic misconduct. 

**Demonstrated effort**. You are expected to put a reasonable amount of time and thought into the task. Don't let your sources do all the work.

**AI and scholarship**.

+ AI is a source, and so its role should be credited. Note that current AI can write poorly in some situations. You should be careful to edit and error-check any material written using AI. You take full responsibility for work submitted under your name.

+ Explaining how you used AI can also be part of your "demonstrated effort."

+ In group work, you are responsible for checking that the sources of your collaborators are properly documented. The whole group must take responsibility for material that the group submits. 

----

## Student Mental Health and Wellbeing

University of Michigan is committed to advancing the mental health and wellbeing of its students. If you or someone you know is feeling overwhelmed, depressed, and/or in need of support, services are available. For help, contact Counseling and Psychological Services (CAPS) at 734.764.8312 and  https://caps.umich.edu during and after hours, on weekends and holidays. You may also consult University Health Service (UHS) at 734.764.8320 and https://www.uhs.umich.edu/mentalhealthsvcs.

----------

## Acknowledgements

Many people have contributed to the development of this course, including all former students and instructors. See the [acknowlegements page](acknowledge.html) for further details.

-----------


         