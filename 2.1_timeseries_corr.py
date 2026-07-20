import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import json
import data_loader as dl

MID_DATA = __import__('pathlib').Path(__file__).resolve().parent.parent / 'Mid_Data'

FEATURE_NAMES = ['Stirrer', 'Feed_Rate', 'Kla', 'pH', 'DO', 'CER', 'OUR', 'RQ']
VAR_NAMES = ['Stirrer', 'Feed', 'KLa', 'pH', 'DO', 'CER', 'OUR', 'RQ', 'GCR']

print('Loading preprocessed data...')
data = dl.load_preprocessed()

X_raw_list = data['X_raw_list']
y_raw_list = data['y_raw_list']

n_batches = len(X_raw_list)
n_times = X_raw_list[0].shape[0]
n_features = len(FEATURE_NAMES)
print(f'  Batches: {n_batches}, Time points: {n_times}, Features: {n_features}')

all_data_list = []
batch_ts_data = np.zeros((n_batches, n_times, len(VAR_NAMES)))

for i in range(n_batches):
    X_i = X_raw_list[i]
    y_i = y_raw_list[i].reshape(-1, 1)
    combined = np.hstack([X_i, y_i])
    batch_ts_data[i] = combined
    all_data_list.append(combined)

all_data = np.vstack(all_data_list)

print('Computing time-series statistics...')
mean_ts = batch_ts_data.mean(axis=0)
max_ts = batch_ts_data.max(axis=0)
min_ts = batch_ts_data.min(axis=0)

timeseries_stats = np.zeros((len(VAR_NAMES), n_times, 3))
timeseries_stats[:, :, 0] = mean_ts.T
timeseries_stats[:, :, 1] = max_ts.T
timeseries_stats[:, :, 2] = min_ts.T

np.save(MID_DATA / 'timeseries_stats.npy', timeseries_stats)
print(f'  Saved timeseries_stats.npy ({timeseries_stats.shape})')

PULSE_BATCHES = [14, 15, 17, 18, 19]
CONT_BATCHES = [i for i in range(20) if i not in PULSE_BATCHES]

pulse_data = batch_ts_data[PULSE_BATCHES]
cont_data = batch_ts_data[CONT_BATCHES]

timeseries_grouped = np.zeros((len(VAR_NAMES), n_times, 3, 2))

for g_idx, (group_data, label) in enumerate([(pulse_data, 'Pulse'), (cont_data, 'Continuous')]):
    m = group_data.mean(axis=0)
    mx = group_data.max(axis=0)
    mn = group_data.min(axis=0)
    timeseries_grouped[:, :, 0, g_idx] = m.T
    timeseries_grouped[:, :, 1, g_idx] = mx.T
    timeseries_grouped[:, :, 2, g_idx] = mn.T
    print(f'  {label} feed (n={group_data.shape[0]}): '
          f'Feed peak={m[:, 1].max():.1f}, GCR peak={m[:, 8].max():.2f}')

np.save(MID_DATA / 'timeseries_grouped.npy', timeseries_grouped)
np.save(MID_DATA / 'batch_groups.npy',
        np.array([0 if i in CONT_BATCHES else 1 for i in range(20)]))
print(f'  Saved timeseries_grouped.npy ({timeseries_grouped.shape})')
print(f'  Saved batch_groups.npy (pulse=1, continuous=0)')

print('Computing Pearson correlation matrix...')
corr_matrix = np.corrcoef(all_data, rowvar=False)
np.fill_diagonal(corr_matrix, 1.0)
np.save(MID_DATA / 'corr_matrix.npy', corr_matrix)
print(f'  Saved corr_matrix.npy ({corr_matrix.shape})')
print(f'  r(OUR, CER) = {corr_matrix[6, 5]:.4f}')
print(f'  r(OUR, GCR) = {corr_matrix[6, 8]:.4f}')
print(f'  r(Feed, GCR) = {corr_matrix[1, 8]:.4f}')

with open(MID_DATA / 'var_names.json', 'w') as f:
    json.dump(VAR_NAMES, f, ensure_ascii=False)
print(f'  Saved var_names.json')

print('\n2.1 Time-series analysis complete.')