# Session 3 Summary: Q1 (Stationarity and Unit Roots)

## Date: 2026-02-03

## Completed Tasks

### ✅ Q1 Category (3 questions - All Converted to Python)

**Files:** `Q1-01/q.qmd`, `Q1-02/q.qmd`, `Q1-03/q.qmd` and corresponding `sol.qmd` files

**Status:** All 3 questions now working with Python

**Questions:**

#### Q1-01: Pure text (no code) ✓
- Conceptual question about trend vs differencing
- No code blocks
- Already Python-compatible after infrastructure conversion

#### Q1-02: R simulation code → Python ✓
**Converted from R to Python:**
- `set.seed()` → `np.random.seed()`
- `rep()` → `np.ones()`
- `runif()` → `np.random.uniform()`
- `filter()` with circular convolution → `np.convolve(mode='same')`
- `dnorm()` → `scipy.stats.norm.pdf()`
- `rnorm()` → `np.random.normal()`
- `plot()` → `matplotlib.pyplot` with subplots

**Python packages used:**
- numpy for random generation and array operations
- scipy.stats.norm for normal density
- matplotlib.pyplot for plotting

**Code in Q1-02/q.qmd (lines 2-36):**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(33)
N = 1000
sd1 = np.ones(N)
events = (np.random.uniform(size=N) < 5/N).astype(float)
sigma = 20
amplitude = 10
filter_length = int(5 * sigma)
filter_seq = np.linspace(-2.5*sigma, 2.5*sigma, filter_length)
filter_weights = norm.pdf(filter_seq, scale=sigma) * sigma * amplitude
sd2 = sd1 + np.convolve(events, filter_weights, mode='same')
Y = np.random.normal(0, sd2)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(Y, linewidth=0.5)
ax.set_xlabel('')
ax.set_ylabel('y')
plt.tight_layout()
plt.show()
```

**Solution file Q1-02/sol.qmd:**
- Also converted to Python (code in eval: false block)

#### Q1-03: Tidyverse + ADF test → Pandas + statsmodels ✓
**Converted from R to Python:**
- `library(tidyverse)` → `import pandas as pd`
- `read.csv()` → `pd.read_csv(skiprows=4)`
- `as.Date()` → `pd.to_datetime()`
- `filter()` → pandas boolean indexing
- `group_by() %>% summarise()` → `.groupby().sum().reset_index()`
- `mutate()` → `.assign()`
- `year()`, `month()` → `.dt.year`, `.dt.month`
- `ggplot() + geom_line() + ...` → matplotlib plotting with date formatting
- `tseries::adf.test()` → `statsmodels.tsa.stattools.adfuller()`

**Python packages used:**
- pandas for data manipulation
- matplotlib.pyplot for plotting
- statsmodels.tsa.stattools for ADF test

**Key conversions in Q1-03/q.qmd:**

**Data processing (lines 8-29):**
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

Collision_data = pd.read_csv("data/nyc_motor_collision_injuries.csv", skiprows=4)
Collision_data['date'] = pd.to_datetime(Collision_data['date'], format="%m/%d/%Y")
Collision_data = Collision_data[(Collision_data['date'] >= '2013-01-01') &
                                  (Collision_data['date'] < '2024-01-01')]

monthly_data = (Collision_data
    .assign(year=Collision_data['date'].dt.year,
            month=Collision_data['date'].dt.month)
    .groupby(['year', 'month'])['person_injured']
    .sum()
    .reset_index()
)
monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
monthly_data = monthly_data[['date', 'person_injured']]
```

**Plotting with date formatting (lines 31-39):**
```python
fig, ax = plt.subplots(figsize=(5, 2.5))
ax.plot(monthly_data['date'], monthly_data['person_injured']/1000, linewidth=1)
ax.set_xlabel('Year')
ax.set_ylabel('injuries (thousands)')
ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()
```

**ADF test (lines 45-52):**
```python
result = adfuller(monthly_data['person_injured'])
print(f"data:  monthly_data['person_injured']")
print(f"Dickey-Fuller = {result[0]:.4f}, Lag order = {result[2]}, p-value = {result[1]:.4f}")
print(f"Alternative hypothesis: stationary")
```

**Solution file Q1-03/sol.qmd:**
- Pure text, no code blocks
- Already compatible, no changes needed

## Testing Results

### ✓ PDF Rendering
- Created `test_q1.qmd` to test Q1 category
- Successfully rendered `test_q1.pdf` with Python engine
- All 3 questions executed without errors
- Both plotting blocks (Q1-02 simulation, Q1-03 time series) rendered correctly
- ADF test printed correctly

### ✓ Temporary File Generation
- Verified `tmp1.qmd` created correctly
- All 3 questions concatenated properly
- Python code blocks generated with correct labels
- Solution includes working as expected

### ✓ Data File
- Verified NYC collision data exists: `data/nyc_motor_collision_injuries.csv`
- File size: 67,933 bytes
- Successfully loaded and processed with pandas

## Current Progress

### Converted to Python:
- ✅ Infrastructure (mt1.qmd)
- ✅ Q6: 4/4 questions (scholarship - pure text)
- ✅ Q2: 4/4 questions (ARMA calculations - pure math)
- ✅ Q1: 3/3 questions (stationarity - ALL CONVERTED!)

**Total: 11/25 questions (44%)**

### Still Using R (14 questions):
- Q3: 4 questions (need code conversion - ARIMA, keep arima2 as R)
- Q4: 5 questions (need code conversion - diagnostics)
- Q5: 2 questions (need code conversion - spectrum/periodogram)
- Q7: 3 questions (need code conversion - ACF/CCF)

## Library Mappings Demonstrated

### Session 3 Used:

| R Package/Function | Python Equivalent | Used In |
|-------------------|-------------------|---------|
| `set.seed()` | `np.random.seed()` | Q1-02 |
| `rep()` | `np.ones()` | Q1-02 |
| `runif()` | `np.random.uniform()` | Q1-02 |
| `filter(circular=T)` | `np.convolve(mode='same')` | Q1-02 |
| `dnorm()` | `scipy.stats.norm.pdf()` | Q1-02 |
| `rnorm()` | `np.random.normal()` | Q1-02 |
| `plot()` | `matplotlib.pyplot` | Q1-02, Q1-03 |
| `library(tidyverse)` | `import pandas as pd` | Q1-03 |
| `read.csv()` | `pd.read_csv()` | Q1-03 |
| `as.Date()` | `pd.to_datetime()` | Q1-03 |
| `filter()` | Boolean indexing | Q1-03 |
| `group_by()` | `.groupby()` | Q1-03 |
| `summarise()` | `.sum().reset_index()` | Q1-03 |
| `mutate()` | `.assign()` | Q1-03 |
| `year()`, `month()` | `.dt.year`, `.dt.month` | Q1-03 |
| `ggplot()` | matplotlib | Q1-03 |
| `tseries::adf.test()` | `statsmodels.tsa.stattools.adfuller()` | Q1-03 |

## Token Usage

**Session 3 Total:** ~4k tokens
- Read Q1 files: ~1k
- Code conversion Q1-02: ~1k
- Code conversion Q1-03: ~1k
- Testing: ~1k

**Much less than the 25k budget!**

## Key Technical Notes

### pandas Groupby Pattern
The R tidyverse pattern:
```r
data %>%
  group_by(year = year(date), month = month(date)) %>%
  summarise(total = sum(value)) %>%
  ungroup()
```

Converts to pandas:
```python
(data
    .assign(year=data['date'].dt.year,
            month=data['date'].dt.month)
    .groupby(['year', 'month'])['value']
    .sum()
    .reset_index()
)
```

### Matplotlib Date Formatting
For year labels on x-axis:
```python
ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
```

### ADF Test Output
statsmodels `adfuller()` returns tuple:
- `result[0]`: Test statistic
- `result[1]`: p-value
- `result[2]`: Number of lags used
- `result[3]`: Number of observations
- `result[4]`: Critical values dictionary

## Next Steps

### Session 4: Q5 + Q7 (Frequency + Analysis) - 5 questions

**Q5-01**: Periodogram/spectrum
- R: `spectrum()` → Python: `scipy.signal.periodogram()`

**Q5-02**: More frequency domain
- Similar conversion

**Q7-01**: CCF (cross-correlation function)
- R: `ccf()` → Python: `statsmodels.tsa.stattools.ccf()`

**Q7-02**: Pure text/math (may already be compatible)

**Q7-03**: ACF (autocorrelation function)
- R: `acf()` → Python: `statsmodels.tsa.stattools.acf()` and `plot_acf()`

**Estimated Session 4:** ~40k tokens (more plotting and diagnostics)

### Session 5: Q3 (Likelihood Inference) - 4 questions
- ARIMA with statsmodels
- **Q3-03: Keep `arima2::arima` as R call via rpy2** (per user instruction)
- Estimated: ~30k tokens

### Session 6: Q4 (Diagnostics) - 5 questions (Most Complex)
- Multiple packages: ARIMA, qqplot, loess, lm, ACF, Shapiro-Wilk
- Estimated: ~45k tokens

## Success Metrics

✓ Q1-01 compatible (pure text)
✓ Q1-02 converted (numpy, scipy, matplotlib)
✓ Q1-03 converted (pandas, matplotlib, statsmodels ADF)
✓ All 3 Q1 questions render correctly
✓ Temporary file generation works
✓ Date formatting works in matplotlib
✓ ADF test works with statsmodels
✓ No regressions in existing functionality
✓ Data file loaded successfully

## Notes

- **pandas datetime operations** (.dt accessor) are very convenient for date manipulation
- **matplotlib date formatting** requires explicit locator and formatter setup
- **statsmodels ADF test** returns tuple, not object with named attributes like R
- **All Q1 questions now pure Python** - no R dependencies in Q1
- The tidyverse → pandas conversion was straightforward using method chaining
- Filter with circular convolution in R maps cleanly to numpy's convolve with mode='same'

## Files Modified This Session

1. `Q1-02/q.qmd` - Converted R simulation to Python
2. `Q1-02/sol.qmd` - Converted R code in solution
3. `Q1-03/q.qmd` - Converted tidyverse + ADF to pandas + statsmodels
4. `test_q1.qmd` - Created for testing (can be deleted)

## Files Ready for Next Session

Q5 and Q7 questions still use R and are next in the conversion plan.
