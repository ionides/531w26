# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the course repository for STATS 531/631 (Winter 2026): Modeling and Analysis of Time Series Data, taught by Edward L. Ionides at the University of Michigan. The repository contains lecture notes, homework assignments, exam materials, and student projects for a graduate-level time series analysis course.

## Build System

The repository uses a Makefile-based build system with shared rules defined in `rules.mk`.

### Common Commands

- **Build lecture slides and notes**: From any chapter directory (01/, 02/, etc.):
  ```bash
  cd 02  # or any chapter number
  make
  ```
  This generates both `slides.pdf` and `notes.pdf` from `source.qmd`.

- **Build homework PDFs**: From homework directories (hw01/, hw02/, etc.):
  ```bash
  cd hw01
  make
  ```

- **Build individual files**:
  - Quarto to PDF (beamer slides): `quarto render source.qmd --to beamer`
  - Quarto to PDF (notes): `quarto render source.qmd --to pdf`
  - Quarto to HTML: `quarto render file.qmd --to html`
  - Rmarkdown to PDF: `Rscript --vanilla -e "rmarkdown::render('file.Rmd')"`
  - Rmarkdown to HTML: `Rscript --vanilla -e "rmarkdown::render('file.Rmd',output_format='html_document')"`

- **Clean build artifacts**:
  ```bash
  make clean  # Remove temporary files (.log, .aux, etc.)
  make fresh  # Full clean including PDFs
  ```

### R Configuration

R scripts use `Rscript --no-save --no-restore --no-init-file` (not `--vanilla` due to Mac compatibility issues). See `rules.mk:3` for details.

## Repository Structure

### Content Organization

- **Numbered directories (01/, 02/, etc.)**: Lecture chapters, each containing:
  - `source.qmd`: Main content (generates both slides and notes)
  - `slides.pdf` and `notes.pdf`: Generated outputs
  - `slides-annotated.pdf`: Hand-annotated slides from lectures (when available)
  - `README.md`: Chapter-specific information with links to materials

- **Homework directories (hw00/, hw01/, hw02/, etc.)**:
  - `q.qmd` or `q.pdf`: Homework questions
  - `sol0N.qmd` and `sol0N.html`: Solutions
  - Homework solutions are HTML files, not PDFs

- **Midterm exam materials**:
  - `mt1/` and `mt2/`: Contain question banks with subdirectories like Q1-01/, Q2-01/
  - Each question directory has `q.Rmd` (question) and `sol.Rmd` (solution)
  - `mt1.Rmd` or `mt2.Rmd`: Test generator that randomly samples from question bank
  - Build with `make` to generate `mt1-all-questions.pdf`, `mt1-all-sol.pdf`, and `mt1-sample.pdf`

- **Quiz materials**: `quiz/` directory with daily class quizzes in Rmd format

- **Project directories**:
  - `midterm_project/`: Midterm project information and past projects
  - `final_project/`: Final project information (when available)

- **Template**: `template/template.qmd` provides a Quarto-based project report template

### Key Files

- `rules.mk`: Shared Makefile rules for building documents across all directories
- `bib531.bib`: Centralized bibliography file
- `syllabus.md`: Course syllabus
- `rubric_homework.md`: Homework grading rubric (emphasizes scholarship and source citation)
- `rubric_midterm.md`: Midterm project rubric
- `python-setup.txt`: Python environment setup instructions

## Technical Stack

### Languages and Tools

- **Python 3.12**: Primary language for data analysis and statistical computing
  - Virtual environment in `.venv/`
  - Key packages: jupyter, matplotlib, pandas, statsmodels, pystan, prophet, scikit-learn, tensorflow
  - Activate: `source .venv/bin/activate`

- **R**: Used for some legacy materials and test generation
  - R Markdown (.Rmd) for exam questions and older content
  - Common R packages: knitr, rmarkdown, tseries

- **Quarto**: Primary document format (.qmd files)
  - Supports both Python and R code blocks
  - Generates PDF (via LaTeX) and HTML output
  - Jupyter kernel for Python execution

### Document Workflows

1. **Lecture notes (.qmd)**: Single source generates both beamer slides and PDF notes using Quarto format directives
2. **Homework (.qmd)**: Questions in .qmd format, solutions rendered to HTML
3. **Exams (.Rmd)**: R Markdown with conditional rendering controlled by flags (ALL, SOL, QLABELS, EXAM)
4. **Projects**: Students use the template.qmd as a starting point

## Exam System Architecture

The midterm exam system uses a sophisticated test generator:

1. **Question Bank**: Questions organized in directories like `Q1-01/`, `Q1-02/` (category 1, variants 1-2)
2. **Dynamic Assembly**: `mt1.Rmd` uses `q_setup()` function to:
   - Randomly sample one question from each category
   - Create temporary files that combine questions with solutions
   - Support multiple modes via boolean flags:
     - `ALL=TRUE`: Show all questions (for question bank PDF)
     - `SOL=TRUE`: Include solutions
     - `QLABELS=TRUE`: Show question labels
     - `EXAM=TRUE`: Format for actual exam
3. **Builds**: Running `make` generates all question variants plus a sample exam

## Python Environment Setup

To set up the Python environment (following `python-setup.txt`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter matplotlib pandas statsmodels tabulate
pip install pystan prophet plotly scikit-learn tensorflow
```

## Course Philosophy and Standards

### Scholarship Expectations

The course strongly emphasizes **transparent source attribution** (see `rubric_homework.md`):
- Students must explicitly cite all sources used, including classmates, AI tools, and previous course materials
- Stating "no sources used" when applicable is considered good scholarship
- Previous course solutions are available but must be acknowledged and built upon
- AI use is encouraged for projects but must be documented

### AI Policy

- AI use is **encouraged** for projects and data analysis homework
- Midterm exams are pencil-and-paper without AI support
- Some homework problems are labeled as "exam preparation" and should be mastered without AI
- Students are expected to demonstrate honest and effective use of AI with proper attribution

## Content Progression

The course follows this general structure (see README.md):
1. **Chapters 1-5**: Classical time series (trend, ARMA models, estimation, identification)
2. **Chapters 6-9**: Frequency domain methods, seasonality, case studies
3. **Chapters 10-18**: State space models, POMP models, advanced topics (forecasting, LSTM, etc.)

As of Winter 2026, chapters beyond 05/ may be in progress or use materials from previous years.

## Development Notes

- The repository uses Git with numbered chapter directories that accumulate throughout the semester
- Chapter content is typically finalized shortly before the corresponding lecture
- When editing chapter content, maintain the dual-output structure (slides + notes) in `source.qmd`
- Bibliography references use `../bib531.bib` relative path from chapter directories
- All course materials are released under a Creative Commons license (see LICENSE)
