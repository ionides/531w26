---
title: "Final project instructions, STATS 531 W26"
output:
  html_document:
---

**Final project outline**. Find a time series dataset of your choice.
Carry out a time series analysis, taking advantage of what we have learned in this course.
It is expected that part of your project will involve a POMP analysis, using the modeling and inference approaches we have studied in the second half of this semester.
A common goal of POMP analysis is to connect theory to data.
To do this, you must think about both the theory and the data.
If possible, choose a dataset on a topic for which you know, or are willing to discover, some background theory.

A good way to get ideas for topics to study and places to find data is to look at the past final projects from [2025](https://ionides.github.io/531w25/final_project/index.html), [2024](https://ionides.github.io/531w24/final_project/index.html), [2022](https://ionides.github.io/531w22/final_project/index.html), [2021](https://ionides.github.io/531w21/final_project/index.html), [2020](https://ionides.github.io/531w20/final_project/index.html), [2018](https://ionides.github.io/531w18/final_project/index.html), [2016](http://ionides.github.io/531w16/final_project/index.html).
Each of these projects should contain the data and information about where the data came from. You may want to search for your own data set, but it is also legitimate to re-analyze data from a previous final project.
If you re-analyze data, you should explain clearly how your analysis goes beyond the previous work, and you should be especially careful to give proper credit to any code you reuse.
Depending on your choice of project, you may be in any of the following situations:

1. A `pypomp` representation already exists for the POMP model you want to use.

1. An `R-pomp` representation already exists for the POMP model you want to use. You will need to translate this to `pypomp`.

1. Your task involves POMP models that are variations on an existing `R-pomp` or `pypomp` representation.

1. Your analysis involves a POMP model which leads you to develop an entirely new `pypomp` investigation that does not inherit from previous `pypomp` or `R-pomp` analysis.

If you develop a `pypomp` representation of a POMP model for a new dataset, test it, and discuss the results in the context of the goals of your data analysis, that is already a full project. 
The more your model derives from previous work, the further you are expected to go in carrying out a thorough data analysis.

The remaining instructions are the same as for the midterm project, except for **(i) a longer page limit, 10 pages; (ii) a revised description of AI scholarship expectations**.
Write a report, submitted on Canvas as a zip file by the deadline, 11:59pm on Tuesday 4/21. The zip file should contain the following:

1. A file called `blinded.qmd` and its compiled version `blinded.pdf` in which all identifying text is removed. This version will be used for anonymous peer review and posted on the course website.

2. Data files and any other files needed to compile the qmd via `quarto render blinded.qmd`. You can assume that the grader and peer reviewers will install required packages as needed.

**Groups**. Sign up for a group of 2-3 people on Canvas. In special situations you can request to write an individual final project. This may be appropriate if you have a particular dataset or scientific question that has motivated your interest in learning time series analysis.

**Choice of data**. The time series should usually have at least 100 time points.
You can have less, if your interests demand it.
Shorter data needs additional care, since model diagnostics and asymptotic approximations become more delicate on small datasets.
If your data are longer than, say, 1000 time points, you can subsample if you start having problems working with too much data.
Come ask the instructor or GSI if you have questions or concerns about your choice of data.

**Data privacy and project anonymity**. The projects, together with their data and source code, will be posted anonymously on the class website unless you have particular reasons why this should not be done. For example, you may have access to data with privacy concerns. The projects will be posted anonymously. After the semester is finished, you can request for your name to be added to your project if you want to.

**Page limit and format requirements**. Use the project template at [https://github.com/ionides/531w26/tree/main/template](https://github.com/ionides/531w26/tree/main/template). The main body of the report is limited to 10 pages. The reference list and Supplementary Material sections are not included in this limit, and there is no constraint on their length. You should not change the font size or other format specifications of the template.

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

    + You are expected to look at previous projects, including those at [https://ionides.github.io/531w25/final_project](https://ionides.github.io/531w25/final_project) and [https://ionides.github.io/531w24/final_project](https://ionides.github.io/531w24/final_project.).
    You can also reference the peer review for these projects, posted online next to the projects.

    + 3 of the scholarship points are for putting your project into the context of previous 531 projects. How is your work similar or different? What relevant things have you learned from previous peer review? It may be helpful to employ teamwork to survey a collection of past projects.

    + If you address a question related to a previous project, you should put your contribution in the context of the previous work and explain how your approach varies or extends the previous work.
    It is especially important that this is clearly explained: substantial points will be lost if the reader has to carry out detective work to figure out clearly the relationship to a previous project.

    + Your report should make references where appropriate.
    For a well-written report the citations should be clearly linked to the material.
    The reader should not have to do detective work to figure out what assertion is linked to what reference.

    + You should properly acknowledge any sources (humans, AIs, documents or internet sites) that contributed to your project.
    
    + When using a reference to point the reader to descriptions elsewhere, you should provide a brief summary in your own report to make it self-contained. 

    + Credit between group members will be addressed via a separate individual Canvas assignment, set up to discuss individual contributions. Unless evidence arises of extreme team dysfunction, all team members will receive the same score. Usually, a few sentences is enough to explain how the work was divided.

    + AI is helpful for developing code in a complex data analysis.
    If used carefully, it can contribute positively to support scientific writing.
    Readers will assume you are using AI, and they will be looking for clear, complete, honest and insightful assessment of its role.
    AI should be credited, like any other source, but its unique capabilities require special attention.
    Your project can be strengthened by a thorough explanation of your research methodology for use of AI.
    You may comment on how you guarded against abuse of AI, to ensure quality control and full human responsibility for the submitted report.
    
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



