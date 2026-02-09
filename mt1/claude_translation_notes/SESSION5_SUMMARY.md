# Session 5 Summary: Q3 (Likelihood Inference for ARIMA)

## Date: 2026-02-03

## Completed Tasks

### ✅ Q3 Category (4 questions - All Converted/Hybrid)

**Files:** `Q3-01/q.qmd`, `Q3-02/q.qmd`, `Q3-03/q.qmd`, `Q3-04/q.qmd` and corresponding `sol.qmd` files

**Status:** All 4 questions working - 3 pure Python, 1 hybrid (Python question + R solution)

**Questions:**

#### Q3-01: AIC table consistency check ✓
**Pure text question** - No code blocks
- Question about mathematical properties of AIC
- Table of AIC values for ARMA(p,q) models
- Asks about inconsistent adjacent pairs
- Already Python-compatible after infrastructure conversion

**Solution:** Pure text, mathematical explanation

#### Q3-02: Fisher information plot ✓
**Converted from R to Python:**
- Simple plot of log-likelihood function
- `seq()` → `np.linspace()`
- Quadratic function: `-2000 - (theta - 5)^2`
- `plot(theta, loglik, ty="l")` → `ax.plot(theta, loglik)`
- `par(mai=)` → `plt.subplots(figsize=)`

**Python packages used:**
- numpy for array operations
- matplotlib.pyplot for plotting

**Key conversion notes:**
- Very straightforward plot conversion
- No statistical computation needed
- Question tests conceptual understanding of Fisher information

**Code in Q3-02/q.qmd (lines 4-21):**
```python
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 10, 200)
loglik = -2000 - (theta - 5)**2

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(theta, loglik)
ax.set_xlabel('theta')
ax.set_ylabel('loglik')
plt.tight_layout()
plt.show()
```

**Solution:** Pure text, mathematical derivation

#### Q3-03: ARIMA model comparison (HYBRID) ✓
**Question converted to Python, Solution kept as R**

This is the special case where we use **hybrid approach**:
- **Question code (Python):** Uses statsmodels ARIMA for standard fitting
- **Solution code (R):** Uses arima2::arima for multiple starting points

**Why hybrid?**
- Per user instruction: arima2::arima is the only function without convenient Python equivalent
- arima2::arima provides multiple random starts for optimization
- No direct Python equivalent with same functionality
- Keeping R code for this special case

**Question code conversion (Q3-03/q.qmd):**
- R: `read.table()` → Python: `pd.read_csv(comment='#')`
- R: `arima(huron_level, order=c(2,0,1))` → Python: `ARIMA(huron_level, order=(2,0,1)).fit()`
- R: Print arima object → Python: `print(model.summary())`

**Python packages used:**
- pandas for reading CSV
- statsmodels.tsa.arima.model.ARIMA for model fitting

**Key conversion notes:**
- Lake Huron data has 21 comment lines starting with #
- Used `comment='#'` parameter in pd.read_csv()
- Header has space before column names: ` Jan` not `Jan`
- ARIMA order tuple: `(p, d, q)` where d=0 for ARMA
- `.fit()` method returns fitted model object
- `.summary()` provides comprehensive output
- `.aic` attribute gives AIC value

**Code in Q3-03/q.qmd (lines 2-19):**
```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

dat = pd.read_csv("data/huron_level.csv", comment='#')
huron_level = dat[' Jan'].values  # Note: space before Jan
year = dat['Year'].values

# Fit ARMA(2,1) and ARMA(2,2) models
arma2_1 = ARIMA(huron_level, order=(2, 0, 1)).fit()
arma2_2 = ARIMA(huron_level, order=(2, 0, 2)).fit()

print(arma2_1.summary())
print("\n")
print(arma2_2.summary())
```

**Solution kept as R (Q3-03/sol.qmd):**
```r
dat <- read.table(file="data/huron_level.csv",sep=",",header=TRUE)
huron_level <- dat$Jan
set.seed(28)
arma2.2.1 <- arima2::arima(huron_level,order=c(2,0,1),
  max_iters=200,max_repeats=20)
arma2.2.2 <- arima2::arima(huron_level,order=c(2,0,2),
  max_iters=200,max_repeats=20)
arma2.2.1
arma2.2.2
```

**Why arima2 is special:**
- `arima2::arima` performs optimization from multiple random starting points
- `max_iters` and `max_repeats` control search thoroughness
- Helps avoid local optima in likelihood surface
- More robust than single optimization run
- No direct Python equivalent with same interface and functionality

**Alternative Python approaches (not implemented):**
- Could manually run ARIMA.fit() with different starting_params
- Could use scipy.optimize with multiple starts
- But arima2 provides convenient, tested interface
- Easier to keep working R code per user instruction

#### Q3-04: Likelihood ratio test and AIC ✓
**Pure mathematics question** - No code blocks
- Conceptual question relating AIC difference to LRT
- Given: AIC for M1 is 0.5 units lower than M0
- Find: p-value expression for LRT
- Tests understanding of relationship between AIC and log-likelihood

**Solution:** Pure text, mathematical derivation showing:
- $AIC_0 - AIC_1 = 2(\ell_1 - \ell_0) - 2 = 0.5$
- Therefore $2(\ell_1 - \ell_0) = 2.5$
- P-value is $P(\chi^2_1 > 2.5) = 0.11$

## Testing Results

### ✓ PDF Rendering
- Created `test_q3.qmd` to test Q3 category
- Successfully rendered `test_q3.pdf` with Python engine
- All 4 questions executed without errors
- Q3-02: Fisher information plot rendered correctly
- Q3-03: ARIMA summaries printed correctly
- Q3-03 solution: R code block executed (arima2::arima)

### ✓ Data File Verified
- `data/huron_level.csv`: 16,800 bytes (Lake Huron levels)
- 21 comment lines starting with #
- Header on line 22: `Year, Jan, Feb, Mar, ...`
- Successfully loaded with pandas using `comment='#'`

### ✓ Temporary File Generation
- Verified `tmp3.qmd` created correctly
- All 4 questions concatenated properly
- Mix of Python and R code blocks working
- Solution includes working as expected

### ✓ Hybrid Python/R Execution
- Q3-03 question runs in Python
- Q3-03 solution runs in R
- Quarto handles mixed language blocks correctly
- No conflicts between Python and R code

## Current Progress

### Converted to Python:
- ✅ Infrastructure (mt1.qmd)
- ✅ Q6: 4/4 questions (scholarship - pure text)
- ✅ Q2: 4/4 questions (ARMA calculations - pure math)
- ✅ Q1: 3/3 questions (stationarity - all converted)
- ✅ Q5: 2/2 questions (frequency domain - all converted)
- ✅ Q7: 3/3 questions (miscellaneous - all converted)
- ✅ Q3: 4/4 questions (ARIMA - 3 pure Python, 1 hybrid)

**Total: 20/25 questions (80%)**

### Still Using R (5 questions):
- Q4: 5 questions (need code conversion - diagnostics, most complex)

## Library Mappings Demonstrated

### Session 5 Used:

| R Package/Function | Python Equivalent | Used In |
|-------------------|-------------------|---------|
| `seq()` | `np.linspace()` | Q3-02 |
| `plot(ty="l")` | `ax.plot()` | Q3-02 |
| `read.table()` (CSV) | `pd.read_csv()` | Q3-03 |
| `arima()` | `ARIMA().fit()` | Q3-03 |
| `AIC()` | `.aic` attribute | Q3-03 |
| `arima2::arima()` | **KEPT AS R** | Q3-03 sol |

### ARIMA in statsmodels vs R

**R's arima():**
```r
model <- arima(y, order=c(p, d, q))
model$aic
model$coef
model$var.coef
```

**Python's statsmodels ARIMA:**
```python
model = ARIMA(y, order=(p, d, q)).fit()
model.aic
model.params
model.cov_params()
```

**Key differences:**
- Python requires explicit `.fit()` call
- R uses `$` accessor, Python uses `.` attributes
- R: `order=c(p,d,q)`, Python: `order=(p,d,q)` tuple
- R: `coef`, Python: `params`
- R: `var.coef`, Python: `cov_params()` method
- Both provide similar functionality for standard cases

**ARIMA summary output:**
- Python: `model.summary()` returns formatted table
- Includes parameter estimates, standard errors, z-scores, p-values
- Shows AIC, BIC, log-likelihood
- More verbose than R's print output
- Similar information, different formatting

## Token Usage

**Session 5 Total:** ~7k tokens
- Read Q3 files: ~2k
- Code conversion Q3-02: ~1k
- Code conversion Q3-03: ~2k
- Testing: ~1k
- Summary: ~1k

**Much less than the 30k budget!**

## Key Technical Notes

### Reading CSV Files with Comments

pandas provides multiple ways to handle comment lines:

1. **comment parameter:** `pd.read_csv(file, comment='#')`
   - Skips all lines starting with comment character
   - Works when comments are at the beginning

2. **skiprows parameter:** `pd.read_csv(file, skiprows=21)`
   - Skips specific number of rows
   - Use when you know exact number of header lines

3. **skip_blank_lines:** Automatically skips blank lines

For Lake Huron data: `comment='#'` is cleanest approach

### Column Names with Spaces

Watch out for spaces in CSV headers:
- Header: `Year, Jan, Feb, ...` has spaces after commas
- Access: `dat[' Jan']` not `dat['Jan']`
- Alternative: `dat.columns = dat.columns.str.strip()` to clean

### ARIMA Model Order

**ARIMA(p, d, q):**
- `p`: AR order (autoregressive)
- `d`: Differencing order (integrated)
- `q`: MA order (moving average)

**ARMA(p, q) is ARIMA(p, 0, q):**
- No differencing (d=0)
- Stationary model

**Common orders:**
- ARMA(2,1): `order=(2, 0, 1)`
- AR(2): `order=(2, 0, 0)`
- MA(1): `order=(0, 0, 1)`

### ARIMA Model Attributes

**Fitted model object provides:**
```python
model.aic          # Akaike Information Criterion
model.bic          # Bayesian Information Criterion
model.llf          # Log-likelihood function value
model.params       # Parameter estimates (array)
model.bse          # Standard errors (array)
model.pvalues      # P-values (array)
model.resid        # Residuals
model.fittedvalues # Fitted values
model.summary()    # Comprehensive summary table
```

### Why Keep arima2 as R

**arima2::arima unique features:**
1. Multiple random starting points (`max_repeats`)
2. Multiple iterations per start (`max_iters`)
3. Returns best result across all attempts
4. Helps avoid local optima
5. Particularly useful for difficult likelihood surfaces

**Python alternatives (would require manual implementation):**
```python
# Would need to manually loop over starting values
best_aic = np.inf
for _ in range(20):
    start_params = generate_random_start()
    model = ARIMA(y, order=(2,0,1)).fit(start_params=start_params)
    if model.aic < best_aic:
        best_model = model
        best_aic = model.aic
```

**Decision:** Keep arima2::arima in R rather than reimplement

### Mixed Language Execution in Quarto

Quarto supports multiple language engines in same document:
- Python blocks: `{python}`
- R blocks: `{r}`
- Each maintains separate namespace
- Can pass data between languages via files
- Q3-03 demonstrates this hybrid approach

## Next Steps

### Session 6: Q4 (Diagnostics) - 5 questions (FINAL SESSION!)

This is the final and most complex category.

**Expected content:**
- ARIMA/SARIMA model fitting
- Residual diagnostics
- Q-Q plots for normality
- ACF/PACF of residuals
- Loess smoothing
- Linear regression
- Shapiro-Wilk test
- Multiple diagnostic plots

**Expected conversions:**
- `arima()` → `ARIMA().fit()`
- `qqnorm()`, `qqline()` → `scipy.stats.probplot()`
- `loess()` → `statsmodels.nonparametric.lowess()`
- `lm()` → `statsmodels.api.OLS()`
- `acf()`, `pacf()` → `plot_acf()`, `plot_pacf()`
- `shapiro.test()` → `scipy.stats.shapiro()`
- Residual plots with matplotlib

**Estimated Session 6:** ~45k tokens (most complex category)

## Success Metrics

✓ Q3-01 compatible (pure text)
✓ Q3-02 converted (numpy, matplotlib)
✓ Q3-03 hybrid approach (Python ARIMA question, R arima2 solution)
✓ Q3-04 compatible (pure text)
✓ All 4 questions render correctly
✓ statsmodels ARIMA works correctly
✓ Lake Huron data loaded successfully
✓ Hybrid Python/R execution works
✓ arima2::arima preserved in R
✓ No regressions in existing functionality

## Files Modified This Session

1. `Q3-02/q.qmd` - Converted R plot to Python
2. `Q3-03/q.qmd` - Converted R arima to Python ARIMA
3. `Q3-03/sol.qmd` - Kept R code with arima2::arima
4. `test_q3.qmd` - Created for testing (deleted after)

## Files Ready for Next Session

Q4 questions (5 total) are the final remaining category to convert.

## Notes

- **Hybrid approach successful** for Q3-03 (Python question + R solution)
- **arima2::arima** is the only R function we're preserving
- **80% of questions converted** (20/25)
- Only 5 questions remain (Q4 category)
- **statsmodels ARIMA** works well for standard cases
- **pandas comment parameter** handles commented CSV files cleanly
- **Mixed language execution** in Quarto is seamless
- Q3 was less work than expected due to 2 pure text questions
- Session 6 (Q4) will be the most complex due to multiple diagnostic techniques

## Special Case: arima2::arima

This is the **only R function** we're keeping throughout the entire conversion:

**Why it's special:**
- Performs global optimization with multiple random starts
- No direct Python equivalent with same convenience
- Well-tested and reliable for difficult optimization problems
- Used specifically when standard optimization may find local optima

**Where it appears:**
- Q3-03 solution only
- Not in main question code (uses standard ARIMA there)
- Documents the value of checking optimization robustness

**How it works in hybrid mode:**
- Question shows standard approach (Python ARIMA)
- Solution shows robust approach (R arima2)
- Demonstrates methodological point about optimization
- Quarto handles both languages seamlessly

This hybrid approach balances:
- Converting to Python where convenient
- Preserving specialized R tools where necessary
- Maintaining pedagogical value of the question
