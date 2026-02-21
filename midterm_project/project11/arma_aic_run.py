import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv("data/DAX_2010-2020.csv")

# Clean date
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data = data.sort_values('Date')

# Clean Open column
data['Open'] = (
    data['Open']
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

data.set_index('Date', inplace=True)
data = data.loc[:'2018-12-31']
data = data.asfreq('B')  # Business day frequency
data['Open'] = data['Open'].ffill()

data['returns'] = np.log(data['Open']) - np.log(data['Open'].shift(1))

returns = data['returns'].dropna()

model = ARIMA(returns, order=(0, 0, 0))
result = model.fit()

# ARMA(4,0,i) model training
for i in range(0, 5):
    model = ARIMA(returns, order=(4, 2, i))
    model_fit = model.fit()
    print("AIC:", model_fit.aic)


# import itertools
# import warnings
# warnings.filterwarnings("ignore")

# p = range(0, 5)
# d = range(0, 3)
# q = range(0, 5)

# best_aic = float("inf")
# best_order = None

# for order in itertools.product(p, d, q):
#     try:
#         model = ARIMA(returns, order=order)
#         model_fit = model.fit()
        
#         if model_fit.aic < best_aic:
#             best_aic = model_fit.aic
#             best_order = order
            
#     except:
#         continue

# print("Best ARIMA order:", best_order)
# print("Best AIC:", best_aic)