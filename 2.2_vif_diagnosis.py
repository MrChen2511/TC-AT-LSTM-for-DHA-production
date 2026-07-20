import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import data_loader as dl

MID_DATA = __import__('pathlib').Path(__file__).resolve().parent.parent / 'Mid_Data'

FEATURE_NAMES = ['Stirrer', 'Feed_Rate', 'Kla', 'pH', 'DO', 'CER', 'OUR', 'RQ']

print('Loading preprocessed data...')
data = dl.load_preprocessed()

X_raw_list = data['X_raw_list']
X_all = np.vstack(X_raw_list)
print(f'  Total samples: {X_all.shape}')

print('Computing Variance Inflation Factor (VIF)...')
vif_values = {}
for j, name in enumerate(FEATURE_NAMES):
    vif = variance_inflation_factor(X_all, j)
    vif_values[name] = vif
    print(f'  {name:12s}  VIF = {vif:.2f}')

vif_arr = np.array([vif_values[name] for name in FEATURE_NAMES])
np.save(MID_DATA / 'vif_values.npy', vif_arr)
print(f'  Saved vif_values.npy ({vif_arr.shape})')

print('\n2.2 VIF diagnosis complete.')