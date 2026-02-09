# MT1 Python Conversion Plan

## Overview

Convert the mt1 exam system from R to Python, maintaining all functionality while using Python/Jupyter as the primary engine.

## Current Architecture

- **Main file**: `mt1.qmd` uses R engine
- **Questions**: 25 question files in `Q*-*/q.qmd` using R code chunks
- **Solutions**: 25 solution files in `Q*-*/sol.qmd` using R code chunks
- **Dynamic system**: `q_setup()` R function creates temporary `.qmd` files and concatenates questions

## Target Architecture

- **Main file**: `mt1.qmd` uses Python/Jupyter engine
- **Questions**: Python code chunks (call R via rpy2 if absolutely needed)
- **Solutions**: Python code chunks
- **Dynamic system**: `q_setup()` rewritten in Python

## Phase 1: Infrastructure Setup

### 1.1 Update Main File Engine
- Convert `mt1.qmd` to use `jupyter: python3` instead of R
- Rewrite `q_setup()` function in Python
- Test temporary file generation and child document inclusion

### 1.2 Establish Python-R Bridge (if needed)
- Install and test `rpy2` for R interop
- Document which functions might require R fallback
- Create helper functions for seamless R calling

### 1.3 Library Mapping
Create Python equivalents for R packages:

| R Package/Function | Python Equivalent | Status |
|-------------------|-------------------|--------|
| `stats::arima()` | `statsmodels.tsa.arima.model.ARIMA` | ✓ Available |
| `arima2::arima()` | Custom wrapper with multiple starts | Need to implement |
| `tseries::adf.test()` | `statsmodels.tsa.stattools.adfuller` | ✓ Available |
| `stats::spectrum()` | `scipy.signal.periodogram` or `statsmodels` | ✓ Available |
| `stats::acf()` | `statsmodels.tsa.stattools.acf` | ✓ Available |
| `stats::ccf()` | `statsmodels.tsa.stattools.ccf` | ✓ Available |
| `stats::Box.test()` | `statsmodels.stats.diagnostic.acorr_ljungbox` | ✓ Available |
| `stats::shapiro.test()` | `scipy.stats.shapiro` | ✓ Available |
| `stats::lm()` | `statsmodels.api.OLS` or `sklearn` | ✓ Available |
| `stats::loess()` | `statsmodels.nonparametric.smoothers_lowess.lowess` | ✓ Available |
| `ggplot2` | `matplotlib` + `seaborn` | ✓ Available |
| `tidyverse` data manipulation | `pandas` | ✓ Available |

## Phase 2: Question Category Conversion (Incremental)

Convert questions by category, testing after each category.

### Priority Order:

1. **Q6 (Scholarship)** - No code, pure text → Easiest
   - Q6-01, Q6-02, Q6-03, Q6-04

2. **Q2 (Calculations for ARMA)** - Mostly mathematical, minimal R code
   - Q2-01, Q2-02, Q2-03, Q2-04

3. **Q1 (Stationarity)** - Mixed code and concepts
   - Q1-01 (no code), Q1-02 (simulation), Q1-03 (tidyverse + ADF)

4. **Q5 (Frequency domain)** - spectrum() function
   - Q5-01, Q5-02

5. **Q7 (Data analysis)** - Various functions
   - Q7-01 (ccf), Q7-02 (no code), Q7-03 (acf)

6. **Q3 (Likelihood inference)** - ARIMA fitting
   - Q3-01, Q3-02, Q3-03 (uses arima2), Q3-04

7. **Q4 (Interpreting diagnostics)** - Most complex, multiple packages
   - Q4-01 (acf), Q4-02 (ARIMA + qqplot), Q4-03 (loess + lm + acf)
   - Q4-04 (ARIMA + shapiro), Q4-05 (ARIMA + Ljung-Box)

## Phase 3: Implementation Strategy

### For Each Question Category:

1. **Analyze**: Identify R functions and packages used
2. **Map**: Find Python equivalents (create mapping table)
3. **Convert**: Rewrite code chunks to Python
4. **Validate**: Ensure output matches R version (plots, statistics)
5. **Test**: Render mt1.qmd with converted questions
6. **Document**: Note any discrepancies or limitations

### Key Conversion Tasks:

#### A. Data Loading and Manipulation
```python
# R: read.table, data manipulation
dat <- read.table(file="data/file.csv", sep=",", header=TRUE)

# Python:
import pandas as pd
dat = pd.read_csv("data/file.csv")
```

#### B. ARIMA Modeling
```python
# R: arima(data, order=c(p,d,q))
# Python:
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data, order=(p, d, q))
fitted = model.fit()
```

#### C. Plotting
```python
# R: plot(), acf(), spectrum()
# Python:
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import periodogram

plt.plot(x, y)
plot_acf(data)
f, Pxx = periodogram(data)
```

#### D. Statistical Tests
```python
# R: adf.test(), shapiro.test(), Box.test()
# Python:
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox

adf_result = adfuller(data)
shapiro_stat, shapiro_p = shapiro(data)
lb_result = acorr_ljungbox(residuals, lags=[5])
```

## Phase 4: Main File Conversion

### 4.1 Rewrite q_setup() in Python
```python
import os
import glob
import random

def q_setup(n, ALL=True, SOL=False, QLABELS=True):
    """
    Setup question files for category n.

    Parameters:
    -----------
    n : int
        Question category number (1-7)
    ALL : bool
        Show all questions (True) or sample one (False)
    SOL : bool
        Include solutions
    QLABELS : bool
        Show question labels

    Returns:
    --------
    str : Path to concatenated temporary file
    """
    q_dirs = sorted(glob.glob(f"Q{n}-*"))
    cat_file = f"tmp{n}.qmd"

    # Create temporary files for each question
    for q_dir in q_dirs:
        q_file = os.path.join(q_dir, "q.qmd")
        q_tmp = os.path.join(q_dir, "tmp.qmd")

        with open(q_tmp, 'w') as f:
            # Add question label
            f.write("```{python}\n")
            f.write("#| echo: false\n")
            f.write(f"#| results: {'asis' if QLABELS else 'hide'}\n")
            f.write(f"my_dir = '{q_dir}'\n")
            f.write("print(f'**{my_dir}.**')\n")
            f.write("```\n\n")

            # Append question content
            with open(q_file, 'r') as qf:
                f.write(qf.read())

            # Add solution if requested
            f.write("\n```{python}\n")
            f.write(f"#| child: !expr '{q_dir}/sol.qmd' if {SOL} else ''\n")
            f.write("```\n")

    # Concatenate questions
    q_files = [os.path.join(d, "tmp.qmd") for d in q_dirs]
    if not ALL:
        q_files = [random.choice(q_files)]

    with open(cat_file, 'w') as cf:
        for qf in q_files:
            with open(qf, 'r') as f:
                cf.write(f.read())
                cf.write("\n\n")

    return cat_file
```

### 4.2 Update YAML Header
```yaml
---
title: "Midterm 1, STATS 531/631 W26"
author: "In class on 2/16"
format:
  pdf:
    pdf-engine: xelatex
    toc: false
jupyter: python3  # Changed from R
csl: ecology.csl
---
```

### 4.3 Update Setup Code Block
```python
#| label: setup
#| echo: false

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seeds for reproducibility
random.seed(48)
np.random.seed(48)

# Define q_setup function (as above)
# ...
```

## Phase 5: Testing Strategy

### 5.1 Unit Testing
For each converted question:
- Compare numerical outputs (AIC values, p-values, etc.)
- Visually compare plots
- Check that formatting matches

### 5.2 Integration Testing
- Render full mt1.qmd after each category conversion
- Test with ALL=TRUE (all questions) and ALL=FALSE (sampled)
- Test with SOL=TRUE and SOL=FALSE
- Verify temporary file generation

### 5.3 Validation Criteria
- PDF renders without errors
- Statistical values match R output (within numerical precision)
- Plots are visually equivalent
- All boolean flags (ALL, SOL, QLABELS, EXAM) work correctly

## Phase 6: Special Cases and Challenges

### 6.1 arima2 Package
Q3-03 uses `arima2::arima` with multiple starting points.
- **Solution**: Implement custom function in Python with multiple random initializations
- Consider using `statsmodels` with different starting parameters

### 6.2 Normal Quantile Plots
R's `qqnorm()` and `qqline()` need Python equivalent.
- **Solution**: Use `scipy.stats.probplot` with matplotlib

### 6.3 Tidyverse/ggplot2 Workflows
Several questions use tidyverse for data manipulation and ggplot2 for plotting.
- **Solution**: Replace with pandas for data manipulation, matplotlib/seaborn for plots

### 6.4 R-specific Output Formatting
Some questions display R model output directly.
- **Solution**: Format Python model summaries to match R output style

## Phase 7: Documentation

### 7.1 Create Conversion Guide
Document Python equivalents for common R functions used in questions.

### 7.2 Update CLAUDE.md
Add Python conversion information and testing procedures.

### 7.3 Code Comments
Add comments explaining Python approach, especially where it differs from R.

## Timeline Estimate

- **Phase 1** (Infrastructure): 2-3 hours
- **Phase 2** (Questions): 8-12 hours
  - Q6: 30 min
  - Q2: 1 hour
  - Q1: 1.5 hours
  - Q5: 1.5 hours
  - Q7: 1.5 hours
  - Q3: 2 hours
  - Q4: 3 hours
- **Phase 3** (Main file): 1 hour
- **Phase 4** (Testing): 2-3 hours
- **Phase 5** (Documentation): 1 hour

**Total**: 14-20 hours

## Risk Mitigation

1. **Keep R versions**: Don't delete `.Rmd` files immediately
2. **Git branching**: Work on a Python conversion branch
3. **Incremental commits**: Commit after each successful category conversion
4. **Comparison testing**: Create scripts to compare R vs Python outputs
5. **Fallback option**: Keep rpy2 integration for difficult cases

## Success Criteria

✓ All 25 questions render in Python
✓ Statistical outputs match R (within reasonable precision)
✓ Plots are visually equivalent
✓ PDF generation works
✓ Random sampling works
✓ Solution toggling works
✓ Question labels work
✓ Exam mode works
