"""
Targeted IF2 based on 1D likelihood slice diagnostics.
Slices showed bs2=0.25 gives ll=-3862.91, better than our best.
Start IF2 from this region and search carefully.
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

OUTDIR = os.path.expanduser('~/final_out6')
FIGDIR = os.path.join(OUTDIR, 'figs')
os.makedirs(FIGDIR, exist_ok=True)

UNIT = 'London'
FIRST_YEAR, LAST_YEAR = 1950, 1963
t0_wall = time.time()

print("="*65)
print("Targeted IF2 from slice-informed starting point")
print("="*65)

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

# Model A
model_A = pypomp.UKMeasles.Pomp(
    unit=[UNIT], theta=theta0, model='001b',
    first_year=FIRST_YEAR, last_year=LAST_YEAR)
model_A.pfilter(J=5000, key=jax.random.key(532), reps=10)
pf_A = model_A.results_history.last()
ll_A = float(np.mean(np.array(pf_A.logLiks).flatten()))
ll_A_se = float(np.std(np.array(pf_A.logLiks).flatten(),ddof=1)/np.sqrt(10))
print(f"Model A log L = {ll_A:.3f} +/- {ll_A_se:.3f}")

# slice-informed starting points
# bs2=0.25 gave -3862.91 (best slice point)
# bs1=0.25 gave -3871.61
# combine: start near (bs1=0.20, bs2=0.25)
starts = [
    # informed by slice peaks
    {'bs1': 0.20,  'bs2': 0.25,  'bs3': -0.042, 'bs4': 0.055,  'bs5': 0.209, 'bs6': -0.005},
    {'bs1': 0.25,  'bs2': 0.25,  'bs3': -0.042, 'bs4': 0.055,  'bs5': 0.209, 'bs6': -0.005},
    {'bs1': 0.20,  'bs2': 0.20,  'bs3': -0.042, 'bs4': 0.055,  'bs5': 0.209, 'bs6': -0.005},
    {'bs1': 0.15,  'bs2': 0.25,  'bs3':  0.000, 'bs4': 0.000,  'bs5': 0.200, 'bs6':  0.000},
    {'bs1': 0.202, 'bs2':-0.038, 'bs3': -0.042, 'bs4': 0.055,  'bs5': 0.209, 'bs6': -0.005},
    # previous best start 8
    {'bs1': 0.188, 'bs2': 0.211, 'bs3':  0.055, 'bs4':-0.017,  'bs5': 0.366, 'bs6': -0.133},
    # previous best start 4
    {'bs1': 0.25,  'bs2': 0.00,  'bs3':  0.000, 'bs4': 0.000,  'bs5': 0.200, 'bs6':  0.000},
    {'bs1': 0.20,  'bs2': 0.30,  'bs3': -0.100, 'bs4': 0.050,  'bs5': 0.150, 'bs6': -0.050},
]

rw_frozen = RWSigma(
    sigmas={
        'R0': 0.0, 'sigma': 0.0, 'gamma': 0.0, 'iota': 0.0,
        'rho': 0.0, 'sigmaSE': 0.0, 'psi': 0.0, 'cohort': 0.0,
        'bs1': 0.01, 'bs2': 0.01, 'bs3': 0.01,
        'bs4': 0.01, 'bs5': 0.01, 'bs6': 0.01,
        'S_0': 0.0, 'E_0': 0.0, 'I_0': 0.0, 'R_0': 0.0,
    },
    init_names=['S_0','E_0','I_0','R_0'],
)

best_ll = -np.inf
best_theta = None
all_lls = []
all_ses = []

print("\nTargeted IF2 (J=2000, M=300, rw_sd=0.01)...")
for i, bs_start in enumerate(starts):
    theta_i = theta_spline.copy()
    theta_i.update(bs_start)

    # pfilter at start first
    model_i = make_model_B(theta_i)
    model_i.pfilter(J=2000, key=jax.random.key(100+i), reps=3)
    pf_start = model_i.results_history.last()
    ll_start = float(np.mean(np.array(pf_start.logLiks).flatten()))

    # IF2
    model_i.mif(J=2000, M=300, rw_sd=rw_frozen, a=0.98,
                key=jax.random.key(200+i), theta=theta_i)
    mif = model_i.results_history.last()
    traces = np.array(mif.traces_da)[0]
    var_names = list(mif.traces_da.coords['variable'].values)
    ll_idx = var_names.index('logLik')
    lls_mif = traces[:, ll_idx]
    best_mif = float(np.nanmax(lls_mif))
    best_iter = int(np.nanargmax(lls_mif))
    best_th = {v: float(traces[best_iter, j])
               for j, v in enumerate(var_names) if v != 'logLik'}

    # pfilter at IF2 best
    model_best = make_model_B(best_th)
    model_best.pfilter(J=5000, key=jax.random.key(300+i), reps=5)
    pf_best = model_best.results_history.last()
    lls_best = np.array(pf_best.logLiks).flatten()
    ll_pf = float(np.mean(lls_best))
    se_pf = float(np.std(lls_best,ddof=1)/np.sqrt(len(lls_best)))
    all_lls.append(ll_pf)
    all_ses.append(se_pf)

    flag = " *** NEW BEST ***" if ll_pf > best_ll else ""
    print(f"  Start {i+1}: init={ll_start:.2f} -> IF2={best_mif:.2f} "
          f"-> pfilter={ll_pf:.3f}+/-{se_pf:.3f}{flag}")
    print(f"    bs: {[round(float(best_th.get(f'bs{k}',0)),3) for k in range(1,7)]}")

    if ll_pf > best_ll:
        best_ll = ll_pf
        best_theta = best_th

print(f"\nBest Model B log L = {best_ll:.3f}")
print(f"Model A log L      = {ll_A:.3f}")
print(f"Gap                = {best_ll-ll_A:.3f}")

# final high-precision evaluation
print("\nFinal precision pfilter (J=5000, 10 reps)...")
model_final = make_model_B(best_theta)
model_final.pfilter(J=5000, key=jax.random.key(999), reps=10)
pf_final = model_final.results_history.last()
lls_final = np.array(pf_final.logLiks).flatten()
ll_final = float(np.mean(lls_final))
ll_final_se = float(np.std(lls_final,ddof=1)/np.sqrt(len(lls_final)))
print(f"  FINAL Model B log L = {ll_final:.3f} +/- {ll_final_se:.3f}")
print(f"  SE = {ll_final_se:.3f}")

n_extra = 5
lrt = 2*(ll_final - ll_A)
pval = 1 - stats.chi2.cdf(lrt, df=n_extra)
print(f"  LRT: stat={lrt:.3f}, df={n_extra}, p={pval:.4f}")
if pval < 0.05:
    print("  => EVIDENCE for Fourier seasonality!")
elif lrt > 0:
    print("  => No significant evidence (p >= 0.05)")
else:
    print(f"  => Model B still below Model A by {abs(ll_final-ll_A):.2f} units")

# save
results = {
    'll_A': ll_A, 'll_A_se': ll_A_se,
    'll_B_final': ll_final, 'll_B_final_se': ll_final_se,
    'all_lls': all_lls, 'all_ses': all_ses,
    'lrt': lrt, 'df': n_extra, 'pval': pval,
    'best_theta': best_theta,
    'wall_time_s': time.time()-t0_wall,
}
with open(os.path.join(OUTDIR,'results6.json'),'w') as f:
    json.dump(results, f, indent=2)

# seasonal figure
t_fine = np.linspace(0,1,365)
bs = np.array([float(best_theta.get(f'bs{i}',0)) for i in range(1,7)])
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
        color='#d6604d',lw=2.5,
        label=f'Model B: Fourier basis (log L={ll_final:.1f})')
ax.plot(t_fine,seas_A/seas_A.mean(),
        color='#2166ac',lw=1.8,ls='--',
        label=f'Model A: school-term (log L={ll_A:.1f})')
ax.axhline(1.0,color='gray',ls=':',lw=0.8,alpha=0.7)
for s,e in [(7,100),(115,199),(252,300),(308,356)]:
    ax.axvspan(s/365,e/365,alpha=0.08,color='#2166ac')
ax.set_xticks(month_ticks); ax.set_xticklabels(month_labels)
ax.set_xlabel('Month of year')
ax.set_ylabel('Relative transmission (normalised to mean=1)')
ax.set_title(f'Seasonal transmission: Fourier basis vs school-term\n'
             f'Gap={ll_final-ll_A:.1f} log-L units  |  '
             f'ΔAIC={2*(ll_final-ll_A)-2*n_extra:.1f}  |  p={pval:.3f}')
ax.legend(); plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig_seasonal_final.png',bbox_inches='tight',dpi=150)
plt.close()
print(f"\nDone. Wall time: {(time.time()-t0_wall)/60:.1f} min")
print("\n"+"="*65)
print("FINAL SUMMARY")
print("="*65)
print(f"  Model A log L = {ll_A:.3f} +/- {ll_A_se:.3f}")
print(f"  Model B log L = {ll_final:.3f} +/- {ll_final_se:.3f}")
print(f"  Gap           = {ll_final-ll_A:.3f} units")
print(f"  AIC diff      = {2*(ll_final-ll_A)-2*n_extra:.2f}")
print(f"  LRT p         = {pval:.4f}")
