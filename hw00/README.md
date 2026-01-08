# Homework 0 (STATS 531). Some course preparation.

This homework has nothing to turn in, but it prepares you for future assignments.

## Read the syllabus

* As the [syllabus](../syllabus.html) explains, the [grading rubric](../rubric_homework.html) puts considerable emphasis on scholarship, including the need to cite all sources and to be explicit about the lack of sources when you do not consult any. You are expected to make it easy for the reader to assess your own contribution and what help you did (and did not) get from sources. This is necessary in order to allow full use of internet resources, including the many past projects and homework solutions available online, as well as AI, while maintaining the course integrity. If you follow standard scientific citation practices, the scholarship points are easy credit and we can happily focus on learning time series. If we have to spend time learning how to give proper credit to sources, that is also not wasted effort.

## Internet repositories for collaboration and open-source research: git and GitHub

* All the course materials are in a git repository, at [https://github.com/ionides/531w26](https://github.com/ionides/531w26). Keeping a local copy of the course repo is a good way to maintain up-to-date copies of all the files. Additional features of GitHub, such as issues and pull requests, are also recommended.

* Read [https://missing.csail.mit.edu/2020/version-control](https://missing.csail.mit.edu/2020/version-control) if you are not yet familiar with git and GitHub. Another resource is Karl Broman's [practical and minimal git/github tutorial](https://kbroman.org/github_tutorial). A deeper, more technical tutorial is https://www.atlassian.com/git/tutorials/.

* The GitHub repository is mirrored to the website at [https://ionides.github.io/531w26/](https://ionides.github.io/531w26/). This can be convenient for viewing the material, however, use of GitHub and a local clone of the repo is recommended.

## File formats and IDEs

* The notes use [Quarto (qmd)](https://quarto.org/) format. This is an update of Rmarkdown (Rmd) which allows the use of Python as well as R. 

* Integrated development environment (IDE) options for working with qmd include vscode (using the Quarto extension), Positron and RStudio. [Positron](https://positron.posit.co/) is a recent product by Posit, the makers of RStudio, and it is currently being developed more actively than RStudio.

* The midterm and final projects will be required to be in qmd format, so newcomers are advised to start familiarizing themselves with it. Check that you can render [https://github.com/ionides/531w26/blob/main/01/source.qmd](https://github.com/ionides/531w26/blob/main/01/source.qmd) to give slides.pdf. To do this, you will need a Python installation, and it is good practice to use a virtual environment. Here is the python setup used for the slides:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter matplotlib pandas statsmodels
```
Then, slides.pdf can be built by
```
quarto render source.qmd --to beamer
```
and notes.pdf, a more compact version in an article format, can be built by
```
quarto render source.qmd --to pdf
```
Alternatively, using the Makefile in the repository, you can get the same effect by running
```
make slides.pdf
make notes.pdf
```
If you have difficulties reproducing slides.pdf, please solve the problem via the usual order for debugging: (i) internet search; (ii) ask colleagues; (iii) contact the instructional team by posting an issue on the class GitHub site and/or asking in class and/or email and/or office hours. 

* The notes, the midterm and final projects, and some of the homeworks are reproducible reports that combine text and source code, generates the results by running the code, and inserts the resulting tables, figures and numbers into the finished document. Advantages of this approach are: (i) you can easily modify your report if you want to try doing something differently; (ii) the reader can, if necessary, inspect or run the code that gave the results; (iii) classmates can easily learn effective data analysis techniques from each other.

------
