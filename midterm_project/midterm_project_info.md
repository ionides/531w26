---
title: "Midterm project instructions"
author: "STATS 531, Winter 2026"
output:
  html_document:
    toc: yes
---

\newcommand\prob{\mathbb{P}}
\newcommand\E{\mathbb{E}}
\newcommand\var{\mathrm{Var}}
\newcommand\cov{\mathrm{Cov}}

**Midterm project outline**. Find a time series dataset of your choice. Carry out a time series analysis, taking advantage of what we have learned so far in this course. You may investigate the relationship between two or more time series (see Chapters 6 and 9). Write a report, submitted on Canvas as a zip file by the deadline, 11:59pm on Friday 2/20. The zip file should contain the following:

1. A file called `blinded.qmd` and its compiled version `blinded.pdf` in which all identifying text is removed. This version will be used for anonymous peer review and posted on the course website.

2. Data files and any other files needed to compile the qmd via `quarto render blinded.qmd`. You can assume that the grader and peer reviewers will install required libraries as needed.

**Groups**. Sign up for a group of 2-3 people on Canvas. In special situations you can request to write an individual midterm project. This may be appropriate if you have a particular dataset or scientific question that has motivated your interest in learning time series analysis.

**Choice of data**. The time series should usually have at least 100 time points. You can have less, if your interests demand it. Shorter data needs additional care, since model diagnostics and asymptotic approximations become more delicate on small datasets. If your data are longer than, say, 1000 time points, you can subsample if you start having problems working with too much data. Come ask the instructor or GSI if you have questions or concerns about your choice of data.

**Data privacy and project anonymity**. The projects, together with their data and source code, will be posted anonymously on the class website unless you have particular reasons why this should not be done. For example, you may have access to data with privacy concerns. The projects will be posted anonymously. After the semester is finished, you can request for your name to be added to your project if you want to.

**Page limit and format requirements**. Use the project template at [https://github.com/ionides/531w26/tree/main/template](https://github.com/ionides/531w26/tree/main/template). The main body of the report is limited to 8 pages. The reference list and Supplementary Material sections are not included in this limit, and there is no constraint on their length. You should not change the font size or other format specifications of the template.

**Expectations for the report**. The report will be graded on the following categories.

* Communicating your data analysis. [10 points]

    + Raising a question. You should explain some background to the data you chose, and give motivation for the reader to appreciate the purpose of your data analysis. 

    + Reaching a conclusion. You should say what you have concluded about your question(s).

    + Considering the reader. Material should be presented in a way that helps the reader to appreciate the contribution of the project. Code and computer output are included only if they are pertinent to the discussion. Usually, code remains in the source file, and numerical results are presented in tables or graphs or text, rather than raw computer output. Value the reader's time: you may lose points for including material that is of borderline relevance, or that is not adequately explained and motivated.

    + You will submit your source code, but you should not expect the reader to study it. If the reader has to study the source code, your report probably has not explained well enough what you were doing.

* Statistical methodology. [10 points]

    + Justify your choices for the statistical methodology.

    + The models and methods you use should be fully explained, either by references or within your report.

    + Focus on a few, carefully explained and justified, figures, tables, statistics and hypothesis tests. You may want to try many different analyses, but the main body of the report should focus on the strongest data-supported evidence linking your question to your conclusions. 

    + Correctness. Obviously, we aim to avoid errors in the math we present, our code, or the reasoning used to draw conclusions from results. 

* Scholarship. [10 points]

    + You are expected to look at previous projects, including those at https://ionides.github.io/531w25/midterm_project and https://ionides.github.io/531w24/midterm_project.
    You can also reference the peer review for these projects, posted online next to the projects.

    + 3 of the scholarship points are for putting your project into the context of previous 531 projecs. How is your work similar or different? What relevant things have you learned from previous peer review? It may be helpful to employ teamwork to survey a collection of past projects.

    + If you address a question related to a previous project, you should put your contribution in the context of the previous work and explain how your approach varies or extends the previous work. 
It is especially important that this is clearly explained: substantial points will be lost if the reader has to carry out detective work to figure out clearly the relationship to a previous project.

    + Your report should make references where appropriate. For a well-written report the citations should be clearly linked to the material. The reader should not have to do detective work to figure out what assertion is linked to what reference.

    + You should properly acknowledge any sources (humans, AIs, documents or internet sites) that contributed to your project.
    
    + When using a reference to point the reader to descriptions elsewhere, you should provide a brief summary in your own report to make it self-contained. 

    + Credit between group members will be addredded via a separate individual Canvas assignment, set up to discuss individual contributions. Unless evidence arises of extreme team dysfunction, all team members will receive the same score. Usually, a few sentences is enough to explain how the work was divided.

    + You may seek assistance from AI. It should be credited in the same way as any other source. You are advised to use AI only for support roles, such as debugging or editing. It is not appropriate to place your own name as author of material created by AI. In addition to ethical considerations, AI may produce low quality technical writing.

**Plagiarism**. If material is taken directly from another source, that source must be cited and the copied material clearly attributed to the source, for example by the use of quotation marks. Failing to do this is [plagiarism](https://en.wikipedia.org/wiki/Plagiarism) may result in zero credit for the scholarship category and the section of the report in which the plagiarism occurs. Here is how the [Rackham Academic and Professional Integrity Policy](https://rackham.umich.edu/policy/section8/) describes plagiarism:

> <b> 8.1.2 Plagiarism </b>
>
> Includes:
>
>    Representing the words, ideas, or work of others as one’s own in writing or presentations, and failing to give full and proper credit to the original source.
>
>    Failing to properly acknowledge and cite language from another source, including paraphrased text.
>
>    Failing to properly cite any ideas, images, technical work, creative content, or other material taken from published or unpublished sources in any medium, including online material or oral presentations, and including the author’s own previous work.

