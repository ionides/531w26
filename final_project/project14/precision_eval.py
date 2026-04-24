"""
Precision evaluation and diagnostics for Fourier basis model.

Three things:
1. High-precision pfilter (J=5000, 10 reps) on the 3 best starts
   to get definitive log L estimates
2. 1D likelihood slices along each bs dimension
3. Better convergence figures using all 10 start results

This gives Ionides the "compelling evidence" he asked for.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pypomp
from pypomp import RWSigma
from pypomp.pomp_class import Pomp
from pypomp.ParTrans_class import ParTrans
from scipy.interpolate import make_smoothing_spline
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(531)

OUTDIR = os.path.expanduser('~/final_out5')
FIGDIR = os.path.join(OUTDIR, 'figs')
os.makedirs(FIGDIR, exist_ok=True)

UNIT = 'London'
FIRST_YEAR, LAST_YEAR = 1950, 1963
t0_wall = time.time()

print("="*65)
print("Precision evaluation + likelihood surface diagnostics")
print("="*65)

# ── setup ─────────────────────────────────────────────────────────
raw = pypomp.UKMeasles.subset(units=[UNIT], clean=False)
measles = raw['measles']
mles = pypomp.UKMeasles.AK_mles()
theta0 = mles[UNIT].to_dict()

sys.path.insert(0, os.path.expanduser('~'))
import model_spline as ms

demog = raw['demog'].drop(columns=['unit'])
times = np.arange(demog['year'].min(), demog['year'].max()+1/12, 1/12)
pop_bspl = make_smoothing_spline(demog['year'], demog['pop'])
births_bspl = make_smoothing_spline(demog['year']+0.5, demog['births'])
covar_df = pd.DataFrame({
    'time': times, 'pop': pop_bspl(times),
    'birthrate': births_bspl(times-4)
}).set_index('time')

dat = measles.copy()
dat['year'] = dat['date'].dt.year
dat_f = dat[(dat['year']>=FIRST_YEAR)&(dat['year']<=LAST_YEAR)].copy()
dat_f['time'] = (
    (dat_f['date']-pd.Timestamp(f'{FIRST_YEAR}-01-01')).dt.days/365.25
    + FIRST_YEAR)
dat_f = dat_f[(dat_f['time']>FIRST_YEAR)&(dat_f['time']<LAST_YEAR+1)]
dat_f = dat_f[['time','cases']].set_index('time')
t0_pomp = float(2*dat_f.index[0]-dat_f.index[1])

theta_spline = {k: v for k, v in theta0.items() if k != 'amplitude'}
theta_spline.update({
    'bs1': 0.1449, 'bs2': 0.1159, 'bs3': -0.1615,
    'bs4': -0.0201, 'bs5': 0.2177, 'bs6': -0.2198,
})

def make_model_B(theta):
    return Pomp(
        ys=dat_f, theta=theta, covars=covar_df,
        t0=t0_pomp, nstep=None, dt=1/365.25, ydim=1,
        accumvars=ms.accumvars, statenames=ms.statenames,
        rinit=ms.rinit, rproc=ms.rproc,
        dmeas=ms.dmeas, rmeas=ms.rmeas,
        par_trans=ParTrans(to_est=ms.to_est, from_est=ms.from_est),
    )

# ── 1. Model A precision pfilter ──────────────────────────────────
print("\n1. Model A precision pfilter (J=5000, 10 reps)...")
model_A = pypomp.UKMeasles.Pomp(
    unit=[UNIT], theta=theta0, model='001b',
    first_year=FIRST_YEAR, last_year=LAST_YEAR)
model_A.pfilter(J=5000, key=jax.random.key(532), reps=10)
pf_A = model_A.results_history.last()
lls_A = np.array(pf_A.logLiks).flatten()
ll_A = float(np.mean(lls_A))
ll_A_se = float(np.std(lls_A,ddof=1)/np.sqrt(len(lls_A)))
print(f"  Model A log L = {ll_A:.3f} +/- {ll_A_se:.3f}")

# ── 2. Load previous 10 start results ─────────────────────────────
# The 10 IF2 starts from last_attempt.py
# bs values from the output log:
all_bs_values = [
    [0.176,  0.156,  0.024,  0.167,  0.126, -0.278],  # start 1: -3906.76
    [0.502,  0.321, -0.003,  0.118,  0.460, -0.241],  # start 2: -3940.40
    [0.363,  0.271, -0.182,  0.068,  0.453, -0.121],  # start 3: -3894.72
    [0.188,  0.211,  0.055, -0.017,  0.366, -0.133],  # start 4: -3870.51 **
    [0.346,  0.034,  0.032, -0.043,  0.453, -0.178],  # start 5: -3898.64
    [0.179,  0.215,  0.076, -0.080,  0.324, -0.453],  # start 6: -3932.91
    [0.478,  0.337,  0.052,  0.040,  0.300, -0.160],  # start 7: -3913.28
    [0.202, -0.038, -0.042,  0.055,  0.209, -0.005],  # start 8: -3868.95 **
    [0.307,  0.247, -0.225,  0.204,  0.504, -0.023],  # start 9: -3914.42
    [0.028,  0.172,  0.125,  0.049,  0.536, -0.138],  # start 10:-3924.35
]
prev_lls = [-3906.76,-3940.40,-3894.72,-3870.51,-3898.64,
            -3932.91,-3913.28,-3868.95,-3914.42,-3924.35]
prev_ses = [3.34,3.08,2.69,1.08,3.63,1.68,0.22,0.55,0.92,1.00]

# ── 3. High-precision pfilter on top 3 starts ─────────────────────
print("\n2. High-precision pfilter on top 3 starts (J=5000, 10 reps)...")
# top 3 by pfilter ll: start 8 (-3868.95), start 4 (-3870.51), start 3 (-3894.72)
top3_idx = [7, 3, 2]  # 0-indexed
top3_lls = []
top3_ses = []
top3_thetas = []

for rank, idx in enumerate(top3_idx):
    bs = all_bs_values[idx]
    theta_i = theta_spline.copy()
    for j, k in enumerate(['bs1','bs2','bs3','bs4','bs5','bs6']):
        theta_i[k] = bs[j]
    top3_thetas.append(theta_i)

    model_i = make_model_B(theta_i)
    model_i.pfilter(J=5000, key=jax.random.key(800+rank), reps=10)
    pf_i = model_i.results_history.last()
    lls_i = np.array(pf_i.logLiks).flatten()
    ll_i = float(np.mean(lls_i))
    se_i = float(np.std(lls_i,ddof=1)/np.sqrt(len(lls_i)))
    top3_lls.append(ll_i)
    top3_ses.append(se_i)
    print(f"  Start {idx+1} (prev ll={prev_lls[idx]:.2f}): "
          f"precise ll = {ll_i:.3f} +/- {se_i:.3f}")
    print(f"    bs: {bs}")

best_B_ll = max(top3_lls)
best_B_se = top3_ses[top3_lls.index(best_B_ll)]
best_B_theta = top3_thetas[top3_lls.index(best_B_ll)]
print(f"\n  Best Model B log L (J=5000) = {best_B_ll:.3f} +/- {best_B_se:.3f}")

# ── 4. 1D likelihood slices ───────────────────────────────────────
print("\n3. 1D likelihood slices along bs1 and bs2 (J=2000, 3 reps)...")
# slice along bs1 holding others at best theta
bs_grid = np.linspace(-1.5, 1.5, 13)
slice_lls_bs1 = []
slice_lls_bs2 = []

print("  Slicing bs1...")
for bs1_val in bs_grid:
    theta_i = best_B_theta.copy()
    theta_i['bs1'] = float(bs1_val)
    model_i = make_model_B(theta_i)
    model_i.pfilter(J=2000, key=jax.random.key(900), reps=3)
    pf_i = model_i.results_history.last()
    ll_i = float(np.mean(np.array(pf_i.logLiks).flatten()))
    slice_lls_bs1.append(ll_i)
    print(f"    bs1={bs1_val:.2f}: ll={ll_i:.2f}")

print("  Slicing bs2...")
for bs2_val in bs_grid:
    theta_i = best_B_theta.copy()
    theta_i['bs2'] = float(bs2_val)
    model_i = make_model_B(theta_i)
    model_i.pfilter(J=2000, key=jax.random.key(901), reps=3)
    pf_i = model_i.results_history.last()
    ll_i = float(np.mean(np.array(pf_i.logLiks).flatten()))
    slice_lls_bs2.append(ll_i)
    print(f"    bs2={bs2_val:.2f}: ll={ll_i:.2f}")

# ── 5. Save results ───────────────────────────────────────────────
n_extra = 5
lrt = 2*(best_B_ll - ll_A)
pval = 1 - stats.chi2.cdf(lrt, df=n_extra)

results = {
    'll_A': ll_A, 'll_A_se': ll_A_se,
    'top3_lls': top3_lls, 'top3_ses': top3_ses,
    'best_B_ll': best_B_ll, 'best_B_se': best_B_se,
    'prev_lls': prev_lls, 'prev_ses': prev_ses,
    'slice_bs1_grid': bs_grid.tolist(),
    'slice_bs1_lls': slice_lls_bs1,
    'slice_bs2_lls': slice_lls_bs2,
    'lrt': lrt, 'df': n_extra, 'pval': pval,
    'best_B_theta': best_B_theta,
    'wall_time_s': time.time()-t0_wall,
}
with open(os.path.join(OUTDIR,'results_precision.json'),'w') as f:
    json.dump(results, f, indent=2)

# ── 6. Figures ────────────────────────────────────────────────────
print("\n4. Generating diagnostic figures...")

# Fig 1: All 10 starts — pfilter log L with error bars
fig, ax = plt.subplots(figsize=(10, 4.5))
starts = np.arange(1, 11)
colors = ['#d6604d' if ll == min(prev_lls) else
          '#f4a582' if ll < -3895 else
          '#cccccc' for ll in prev_lls]
bars = ax.bar(starts, prev_lls, color=colors,
              edgecolor='white', alpha=0.9, width=0.6)
ax.errorbar(starts, prev_lls, yerr=[2*s for s in prev_ses],
            fmt='none', color='#333333', capsize=4, lw=1.5)
ax.axhline(ll_A, color='#2166ac', ls='--', lw=1.8,
           label=f'Model A log L = {ll_A:.1f}')
ax.axhline(best_B_ll, color='#d6604d', ls=':', lw=1.5,
           label=f'Model B best (J=5000) = {best_B_ll:.1f}')
ax.set_xlabel('IF2 start number')
ax.set_ylabel('Log-likelihood (pfilter J=2000)')
ax.set_title('Model B: pfilter log L across 10 IF2 starts\n'
             '(frozen epidemic parameters, bs1-bs6 only)')
ax.set_xticks(starts)
ax.legend(fontsize=9)
ax.set_ylim(min(prev_lls)-100, ll_A+50)
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig_global_search.png', bbox_inches='tight', dpi=150)
plt.close()

# Fig 2: 1D likelihood slices
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
best_bs1 = float(best_B_theta.get('bs1', 0))
best_bs2 = float(best_B_theta.get('bs2', 0))

ax = axes[0]
ax.plot(bs_grid, slice_lls_bs1, 'o-', color='#d6604d',
        lw=2, ms=6, label='Profile log L')
ax.axvline(best_bs1, color='#d6604d', ls='--', lw=1.2,
           label=f'Best bs1 = {best_bs1:.3f}')
ax.axhline(ll_A, color='#2166ac', ls=':', lw=1.5,
           label=f'Model A = {ll_A:.1f}')
ax.set_xlabel('bs1 (sin 2πt coefficient)')
ax.set_ylabel('Log-likelihood (pfilter J=2000)')
ax.set_title('1D likelihood slice: bs1\n(all other params fixed at best theta)')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(bs_grid, slice_lls_bs2, 's-', color='#4393c3',
        lw=2, ms=6, label='Profile log L')
ax.axvline(best_bs2, color='#4393c3', ls='--', lw=1.2,
           label=f'Best bs2 = {best_bs2:.3f}')
ax.axhline(ll_A, color='#2166ac', ls=':', lw=1.5,
           label=f'Model A = {ll_A:.1f}')
ax.set_xlabel('bs2 (cos 2πt coefficient)')
ax.set_ylabel('Log-likelihood (pfilter J=2000)')
ax.set_title('1D likelihood slice: bs2\n(all other params fixed at best theta)')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig_likelihood_slices.png', bbox_inches='tight', dpi=150)
plt.close()

# Fig 3: precision comparison — top 3 starts
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [f'Start {top3_idx[i]+1}\n(best of 10)' if i==0
          else f'Start {top3_idx[i]+1}'
          for i in range(3)]
colors_p = ['#d6604d', '#f4a582', '#fddbc7']
bars = ax.bar(labels, top3_lls, color=colors_p,
              edgecolor='white', alpha=0.9, width=0.5)
ax.errorbar(labels, top3_lls, yerr=[2*s for s in top3_ses],
            fmt='none', color='#333333', capsize=5, lw=1.5)
ax.axhline(ll_A, color='#2166ac', ls='--', lw=1.8,
           label=f'Model A log L = {ll_A:.2f}')
for bar, val in zip(bars, top3_lls):
    ax.text(bar.get_x()+bar.get_width()/2, val+1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
ax.set_ylim(min(top3_lls)-50, ll_A+30)
ax.set_ylabel('Log-likelihood (pfilter J=5000, 10 reps)')
ax.set_title(f'High-precision evaluation: top 3 Model B starts\n'
             f'LRT stat={lrt:.2f}, df={n_extra}, p={pval:.4f}')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig_precision_comparison.png', bbox_inches='tight', dpi=150)
plt.close()

# Fig 4: final seasonal comparison with precise numbers
t_fine = np.linspace(0, 1, 365)
bs_best = np.array([float(best_B_theta.get(f'bs{i}',0)) for i in range(1,7)])
def fourier_np(t, bs):
    p = 2*np.pi*t
    b = np.array([np.sin(p),np.cos(p),np.sin(2*p),
                  np.cos(2*p),np.sin(3*p),np.cos(3*p)])
    return np.exp(np.dot(bs, b))
fourier_curve = np.array([fourier_np(t, bs_best) for t in t_fine])
amp = theta0.get('amplitude', 0.554)
t_days = t_fine*365.25
seas_A = np.where(
    ((t_days>=7)&(t_days<=100))|((t_days>=115)&(t_days<=199))|
    ((t_days>=252)&(t_days<=300))|((t_days>=308)&(t_days<=356)),
    1.0+amp*0.2411/0.7589, 1-amp)
month_labels=['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
month_ticks = np.array([0,31,59,90,120,151,181,212,243,273,304,334])/365

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(t_fine, fourier_curve/fourier_curve.mean(),
        color='#d6604d', lw=2.5, label=f'Model B: Fourier basis (log L={best_B_ll:.1f})')
ax.plot(t_fine, seas_A/seas_A.mean(),
        color='#2166ac', lw=1.8, ls='--',
        label=f'Model A: school-term step (log L={ll_A:.1f})')
ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.7)
for s, e in [(7,100),(115,199),(252,300),(308,356)]:
    ax.axvspan(s/365, e/365, alpha=0.08, color='#2166ac')
ax.set_xticks(month_ticks)
ax.set_xticklabels(month_labels)
ax.set_xlabel('Month of year')
ax.set_ylabel('Relative transmission (normalised to mean=1)')
ax.set_title(f'Fitted seasonal transmission: Fourier basis vs school-term\n'
             f'Gap = {best_B_ll-ll_A:.1f} log-likelihood units  |  '
             f'ΔAIC = {2*(best_B_ll-ll_A)-2*n_extra:.1f}  |  '
             f'p = {pval:.3f}')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig_seasonal_final.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nAll figures saved.")
print(f"Total wall time: {(time.time()-t0_wall)/60:.1f} min")
print("\n"+"="*65)
print("PRECISION SUMMARY")
print("="*65)
print(f"  Model A log L (J=5000, 10 reps) = {ll_A:.3f} +/- {ll_A_se:.3f}")
print(f"  Model B best  (J=5000, 10 reps) = {best_B_ll:.3f} +/- {best_B_se:.3f}")
print(f"  Gap = {best_B_ll-ll_A:.2f} log-likelihood units")
print(f"  AIC difference = {2*(best_B_ll-ll_A)-2*n_extra:.2f} (+ favors A)")
print(f"  LRT: stat={lrt:.2f}, df={n_extra}, p={pval:.4f}")
print(f"  1D slice bs1: max = {max(slice_lls_bs1):.2f} "
      f"at bs1={bs_grid[slice_lls_bs1.index(max(slice_lls_bs1))]:.2f}")
print(f"  1D slice bs2: max = {max(slice_lls_bs2):.2f} "
      f"at bs2={bs_grid[slice_lls_bs2.index(max(slice_lls_bs2))]:.2f}")
