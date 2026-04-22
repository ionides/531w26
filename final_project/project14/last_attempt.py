"""
Last attempt: freeze ALL epidemic params at theta0
Only optimize bs1-bs6, J=2000 throughout
This eliminates particle collapse risk entirely
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
print("LAST ATTEMPT: freeze epidemic params, J=2000, bs only")
print("="*65)

raw = pypomp.UKMeasles.subset(units=[UNIT], clean=False)
measles = raw['measles']
mles = pypomp.UKMeasles.AK_mles()
theta0 = mles[UNIT].to_dict()

# Model A
model_A = pypomp.UKMeasles.Pomp(
    unit=[UNIT], theta=theta0, model='001b',
    first_year=FIRST_YEAR, last_year=LAST_YEAR)
print("\nModel A pfilter (J=2000, 5 reps)...")
model_A.pfilter(J=2000, key=jax.random.key(532), reps=5)
pf_A = model_A.results_history.last()
lls_A = np.array(pf_A.logLiks).flatten()
ll_A = float(np.mean(lls_A))
ll_A_se = float(np.std(lls_A,ddof=1)/np.sqrt(len(lls_A)))
print(f"  Model A log L = {ll_A:.2f} +/- {ll_A_se:.2f}")

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

# pfilter at init
model_B = make_model_B(theta_spline)
print("\nModel B pfilter at init (J=2000, 5 reps)...")
model_B.pfilter(J=2000, key=jax.random.key(533), reps=5)
pf_Bi = model_B.results_history.last()
lls_Bi = np.array(pf_Bi.logLiks).flatten()
ll_Bi = float(np.mean(lls_Bi))
ll_Bi_se = float(np.std(lls_Bi,ddof=1)/np.sqrt(len(lls_Bi)))
print(f"  Model B init log L = {ll_Bi:.2f} +/- {ll_Bi_se:.2f}")

# KEY: freeze ALL epidemic params, only perturb bs
# rw_sd=0.0 means frozen, only bs1-bs6 are free
rw_frozen = RWSigma(
    sigmas={
        'R0': 0.0, 'sigma': 0.0, 'gamma': 0.0, 'iota': 0.0,
        'rho': 0.0, 'sigmaSE': 0.0, 'psi': 0.0, 'cohort': 0.0,
        'bs1': 0.02, 'bs2': 0.02, 'bs3': 0.02,
        'bs4': 0.02, 'bs5': 0.02, 'bs6': 0.02,
        'S_0': 0.0, 'E_0': 0.0, 'I_0': 0.0, 'R_0': 0.0,
    },
    init_names=['S_0','E_0','I_0','R_0'],
)

# Multiple IF2 runs from different bs starts, J=2000
print("\nIF2 frozen epidemic params, bs only (J=2000, M=200)...")
rng = np.random.default_rng(531)
best_ll_overall = ll_Bi
best_theta_overall = theta_spline
all_results = []

# start 0: school-term approximating init
starts = [theta_spline.copy()]

# starts 1-9: random bs in [-1, 1]
for i in range(9):
    t = theta_spline.copy()
    for k in ('bs1','bs2','bs3','bs4','bs5','bs6'):
        t[k] = float(rng.uniform(-1.0, 1.0))
    starts.append(t)

for i, theta_i in enumerate(starts):
    print(f"\n  Start {i+1}/10...")
    model_Bi = make_model_B(theta_i)
    model_Bi.mif(J=2000, M=200, rw_sd=rw_frozen, a=0.97,
                 key=jax.random.key(600+i), theta=theta_i)
    mif = model_Bi.results_history.last()
    traces = np.array(mif.traces_da)[0]
    var_names = list(mif.traces_da.coords['variable'].values)
    ll_idx = var_names.index('logLik')
    lls_mif = traces[:, ll_idx]
    best_mif = float(np.nanmax(lls_mif))
    best_iter = int(np.nanargmax(lls_mif))
    best_theta = {v: float(traces[best_iter, j])
                  for j, v in enumerate(var_names) if v != 'logLik'}
    print(f"    IF2 best approx = {best_mif:.2f} (iter {best_iter})")

    # pfilter at best — epidemic params are frozen so no collapse risk
    model_best = make_model_B(best_theta)
    model_best.pfilter(J=2000, key=jax.random.key(700+i), reps=3)
    pf_best = model_best.results_history.last()
    lls_best = np.array(pf_best.logLiks).flatten()
    ll_pf = float(np.mean(lls_best))
    se_pf = float(np.std(lls_best,ddof=1)/np.sqrt(len(lls_best)))
    print(f"    pfilter = {ll_pf:.2f} +/- {se_pf:.2f}")
    print(f"    bs: {[round(float(best_theta.get(f'bs{k}',0)),3) for k in range(1,7)]}")
    all_results.append((ll_pf, se_pf, best_theta))

    if ll_pf > best_ll_overall:
        best_ll_overall = ll_pf
        best_theta_overall = best_theta
        print(f"    *** NEW BEST: {ll_pf:.2f} ***")

print(f"\nBest log L found: {best_ll_overall:.2f}")
print(f"Init log L was:   {ll_Bi:.2f}")
print(f"Improvement:      {best_ll_overall - ll_Bi:.2f}")

# final pfilter at best with 5 reps
print("\nFinal pfilter at best theta (J=2000, 5 reps)...")
model_final = make_model_B(best_theta_overall)
model_final.pfilter(J=2000, key=jax.random.key(999), reps=5)
pf_final = model_final.results_history.last()
lls_final = np.array(pf_final.logLiks).flatten()
ll_final = float(np.mean(lls_final))
ll_final_se = float(np.std(lls_final,ddof=1)/np.sqrt(len(lls_final)))
print(f"  Final Model B log L = {ll_final:.2f} +/- {ll_final_se:.2f}")
print(f"  SE = {ll_final_se:.2f} (should be < 3 for stable filter)")

# use best of init vs final
ll_B_report = max(ll_Bi, ll_final)
ll_B_report_se = ll_Bi_se if ll_Bi >= ll_final else ll_final_se
best_report_theta = theta_spline if ll_Bi >= ll_final else best_theta_overall

n_extra = 5
lrt = 2*(ll_B_report - ll_A)
pval = 1 - stats.chi2.cdf(lrt, df=n_extra)

print(f"\nFINAL SUMMARY")
print(f"="*65)
print(f"  Model A log L      = {ll_A:.2f} +/- {ll_A_se:.2f}")
print(f"  Model B init log L = {ll_Bi:.2f} +/- {ll_Bi_se:.2f}")
print(f"  Model B best log L = {ll_B_report:.2f} +/- {ll_B_report_se:.2f}")
print(f"  Gap                = {ll_B_report - ll_A:.2f} units")
print(f"  AIC difference     = {2*(ll_B_report-ll_A) - 2*n_extra:.2f} (+ = favors A)")
print(f"  LRT: stat={lrt:.2f}, df={n_extra}, p={pval:.4f}")
if pval < 0.05:
    print("  => EVIDENCE for Fourier seasonality!")
elif lrt > 0:
    print("  => No significant evidence (p >= 0.05)")
else:
    print("  => IF2 did not beat init")

# save
results = {
    'll_A': ll_A, 'll_A_se': ll_A_se,
    'll_B_init': ll_Bi, 'll_B_init_se': ll_Bi_se,
    'll_B_final': ll_B_report, 'll_B_final_se': ll_B_report_se,
    'all_start_results': [(r[0], r[1]) for r in all_results],
    'lrt': lrt, 'df': n_extra, 'pval': pval,
    'best_theta': best_report_theta,
    'theta0': theta0,
    'wall_time_s': time.time()-t0_wall,
}
with open(os.path.join(OUTDIR,'results5.json'),'w') as f:
    json.dump(results, f, indent=2)

# figures
def dec2date(t):
    year=int(t); day=int((t-year)*365.25)
    return pd.Timestamp(year=year,month=1,day=1)+pd.Timedelta(days=day)

t_fine = np.linspace(0,1,365)
bs = np.array([float(best_report_theta.get(f'bs{i}',0)) for i in range(1,7)])
def fourier_np(t,bs):
    p=2*np.pi*t
    b=np.array([np.sin(p),np.cos(p),np.sin(2*p),np.cos(2*p),np.sin(3*p),np.cos(3*p)])
    return np.exp(np.dot(bs,b))
fourier_curve = np.array([fourier_np(t,bs) for t in t_fine])
amp = theta0.get('amplitude',0.554)
t_days = t_fine*365.25
seas_A = np.where(
    ((t_days>=7)&(t_days<=100))|((t_days>=115)&(t_days<=199))|
    ((t_days>=252)&(t_days<=300))|((t_days>=308)&(t_days<=356)),
    1.0+amp*0.2411/0.7589, 1-amp)
month_labels=['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
month_ticks=np.array([0,31,59,90,120,151,181,212,243,273,304,334])/365

fig,ax=plt.subplots(figsize=(10,4.5))
ax.plot(t_fine,fourier_curve/fourier_curve.mean(),
        color='#d6604d',lw=2.5,label='Model B: Fourier basis (best)')
ax.plot(t_fine,seas_A/seas_A.mean(),
        color='#2166ac',lw=1.8,ls='--',label='Model A: school-term step')
ax.axhline(1.0,color='gray',ls=':',lw=0.8,alpha=0.7)
for s,e in [(7,100),(115,199),(252,300),(308,356)]:
    ax.axvspan(s/365,e/365,alpha=0.08,color='#2166ac')
ax.set_xticks(month_ticks); ax.set_xticklabels(month_labels)
ax.set_xlabel('Month of year')
ax.set_ylabel('Relative transmission (normalised to mean=1)')
ax.set_title(f'Seasonal transmission: Fourier basis vs school-term\n'
             f'Model A={ll_A:.1f}  |  Model B={ll_B_report:.1f}  |  '
             f'AIC diff={2*(ll_B_report-ll_A)-2*n_extra:.1f}')
ax.legend(); plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig_seasonal.png',bbox_inches='tight')
plt.close()
print(f"\nFigures saved. Total wall time: {(time.time()-t0_wall)/60:.1f} min")
