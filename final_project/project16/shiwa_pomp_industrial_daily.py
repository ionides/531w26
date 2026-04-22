import os
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
import pandas as pd

from pypomp.pomp_class import Pomp
from pypomp.ParTrans_class import ParTrans

STATENAMES = ['X_I']

PARAM_NAMES = [
    'alpha_T',
    'alpha_H',
    'alpha_W',
    'E',
    'delta_c',
    'delta_s',
    'sigma',
    'tau',
]

X_I_0_FIXED = 10.0

COVAR_NAMES = ['temperature', 'relative_humidity', 'wind_speed', 'pressure', 'day_cos', 'day_sin']


def to_est(theta: dict) -> dict:
    return {
        'alpha_T': jnp.log(theta['alpha_T']),
        'alpha_H': theta['alpha_H'],
        'alpha_W': theta['alpha_W'],
        'E':       jnp.log(theta['E']),
        'delta_c': theta['delta_c'],
        'delta_s': theta['delta_s'],
        'sigma':   jnp.log(theta['sigma']),
        'tau':     jnp.log(theta['tau']),
    }

def from_est(theta: dict) -> dict:
    return {
        'alpha_T': jnp.exp(theta['alpha_T']),
        'alpha_H': theta['alpha_H'],
        'alpha_W': theta['alpha_W'],
        'E':       jnp.exp(theta['E']),
        'delta_c': theta['delta_c'],
        'delta_s': theta['delta_s'],
        'sigma':   jnp.exp(theta['sigma']),
        'tau':     jnp.exp(theta['tau']),
    }

par_trans = ParTrans(to_est=to_est, from_est=from_est)


def rinit(theta_, key, covars, t0):
    return {'X_I': jnp.array(X_I_0_FIXED)}

def rproc(X_, theta_, key, covars, t, dt):
    T       = covars['temperature']
    H       = covars['relative_humidity']
    W       = covars['wind_speed']
    day_cos = covars['day_cos']
    day_sin = covars['day_sin']

    alpha_T = theta_['alpha_T']
    alpha_H = theta_['alpha_H']
    alpha_W = theta_['alpha_W']
    E       = theta_['E']
    delta_c = theta_['delta_c']
    delta_s = theta_['delta_s']
    sigma   = theta_['sigma']

    pbl_mixing = jnp.log1p(jnp.exp(T - 5.0))
    decay = jnp.exp(
        -alpha_T * pbl_mixing
        -alpha_H * (H / 100.0)
        -alpha_W * W
    )

    E_t = E * jnp.exp(delta_c * day_cos + delta_s * day_sin)

    eps   = jax.random.normal(key)
    new_X = (X_['X_I'] * decay + E_t) * jnp.exp(sigma * eps)
    return {'X_I': new_X}

def rmeas(X_, theta_, key, covars, t):
    tau = theta_['tau']
    Y = X_['X_I'] * jnp.exp(tau * jax.random.normal(key))
    return jnp.array([Y])

def dmeas(Y_, X_, theta_, covars, t):
    tau = theta_['tau']
    mu  = jnp.log(X_['X_I'])
    return jstats.norm.logpdf(jnp.log(Y_['Y_I']), mu, tau)


DEFAULT_THETA = {
    'alpha_T': 0.5,
    'alpha_H': 0.5,
    'alpha_W': 0.5,
    'E':       5.0,
    'delta_c': 0.0,
    'delta_s': 0.0,
    'sigma':   0.5,
    'tau':     0.5,
}


def load_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

    ind = pd.read_csv(data_dir + 'shiwa_industrial_daily.csv',
                      index_col='time_stamp', parse_dates=True)

    ys = pd.DataFrame({'Y_I': ind['pm2.5'].clip(lower=0.01)})
    ys.index = np.arange(len(ys), dtype=float)

    doy = ind.index.dayofyear.values.astype(float)
    day_cos = np.cos(2.0 * np.pi * doy / 365.0)
    day_sin = np.sin(2.0 * np.pi * doy / 365.0)

    covars = pd.DataFrame({
        'temperature':       ind['temperature'].values,
        'relative_humidity': ind['relative_humidity'].values,
        'wind_speed':        ind['wind_speed'].values,
        'pressure':          ind['air_pressure'].values,
        'day_cos':           day_cos,
        'day_sin':           day_sin,
    })
    covars.index = np.arange(len(covars), dtype=float)

    return ys, covars


def build_pomp(theta=None, data_dir=None):
    if theta is None:
        theta = DEFAULT_THETA
    ys, covars = load_data(data_dir)
    model = Pomp(
        ys=ys,
        theta=theta,
        statenames=STATENAMES,
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ydim=1,
        covars=covars,
        par_trans=par_trans,
        nstep=1,
    )
    return model


if __name__ == '__main__':
    print('Building daily industrial model...')
    model = build_pomp()
    print('Model built successfully.')
    print(f'Observations: {model.ys.shape}')
    print(f'Covariates:   {model.covars.shape}')
