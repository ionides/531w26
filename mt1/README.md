
# EXAM 1

`mt1.qmd` has 2 Python flags 
ALL : include all the questions in the generator
SOL : include solutions

mt1.qmd also has a Latex flag
%% \newcommand\exam[1]{#1} %% exam paper formatting
\newcommand\exam[1]{} %% omit exam paper formatting


`mt1.qmd` collates temporary files that have been previously built using `python generate_questions.py`. This file has similar labels, whic should be edited to match `mt1.qmd`.

QLABELS : include question directory labels. usually set equal to ALL
ALL : include all the questions in the generator
SOL : include solutions

For each question x,
python generate_questions.py
does the following:

1. it adds a line to each of the files Qx-n/q.qmd that creates a variable my_dir telling the file what directory it is in. The result is saved in Qx-n/tmp.qmd
2. it concatenates all these files in case ALL = True
3. if ALL = False, it instead pulls out a random question
4. it writes tmp*.qmd for each section

Then, quarto render can be run to make the pdf

When writing questions,

1. the file Qx-n/q.qmd can assume there is an R variable my_dir giving the name of the current directory. 
2. datasets could be in the directory Qx-n and can be accessed, e.g., by
vg <- read.table(paste(my_dir,"/vg_sales.txt",sep="")). However, this has been deprecated in favor of putting all data in mt1/data
3. solutions should be called Qx-n/sol.qmd or Qx-n/sol-a.qmd etc.

In this setup, a question doesn't have to know its question number or directory name, which makes it easier to shuffle questions around without having to rename lots of internal links.

