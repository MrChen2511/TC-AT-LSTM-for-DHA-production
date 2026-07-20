import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import pickle
import csv
import time
import warnings
from pathlib import Path

import torch
import training
from training import DEFAULT_CONFIG
from models import get_dl_model
import data_loader

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
MID_DATA = ROOT / 'Mid_Data'

OPTIMAL_CONFIG = {'attn_window': 5, 'hidden_dim': 128, 'lambda_traj': 0.002}
CONTINUOUS_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]
BEESWARM_BATCHES = [4, 5, 8, 13, 17]
PHASE_SPLIT_HOUR = 40
PHASE_SPLIT_IDX = PHASE_SPLIT_HOUR - 1
WINDOW_SIZE = 5
N_BACKGROUND = 50
RANDOM_SEED = 42
FEATURE_NAMES = ['Stirrer', 'KLa', 'pH', 'DO', 'CER', 'OUR', 'RQ', 'Feed']


def build_windows(X_std_list, batch_ids):
    windows, time_idx, batch_idx = [], [], []
    for bid in batch_ids:
        X_seq = X_std_list[bid - 1]
        for t in range(WINDOW_SIZE - 1, 120):
            windows.append(X_seq[t - WINDOW_SIZE + 1:t + 1])
            time_idx.append(t)
            batch_idx.append(bid)
    return np.array(windows), np.array(time_idx), np.array(batch_idx)


def train_model(data):
    X_std_list = data['X_std_list']
    y_std_list = data['y_std_list']
    scaler_y = data['scaler_y']
    D_raw_list = data.get('D_list', None)
    Glu_raw_list = data.get('Glu_list', None)

    seqs = training._prepare_sequences(X_std_list, y_std_list, CONTINUOUS_BATCHES)
    config = DEFAULT_CONFIG.copy()
    config['hidden_dim'] = OPTIMAL_CONFIG['hidden_dim']
    config['lambda_traj'] = OPTIMAL_CONFIG['lambda_traj']

    model = get_dl_model('M-TCAL', input_dim=config['input_dim'],
                         hidden_dim=OPTIMAL_CONFIG['hidden_dim'],
                         num_layers=config['num_layers'],
                         attn_window=OPTIMAL_CONFIG['attn_window'])

    model = training.train_dl_sequential(model, seqs,
                                          X_std_list, y_std_list, [1],
                                          lambda_traj=OPTIMAL_CONFIG['lambda_traj'],
                                          D_raw_list=D_raw_list, Glu_raw_list=Glu_raw_list,
                                          scaler_y=scaler_y, train_bids=CONTINUOUS_BATCHES,
                                          config=config)
    model.eval()
    return model


def aggregate_shap(sv_2d, n_windows):
    sv_3d = sv_2d.reshape(n_windows, WINDOW_SIZE, 8)
    return sv_3d.sum(axis=1)


def main():
    print('=' * 70)
    print('  5.1 Temporal-SHAP (sliding window, 40-dim, nsamples=auto)')
    print(f'  aw={OPTIMAL_CONFIG["attn_window"]}, hd={OPTIMAL_CONFIG["hidden_dim"]}, '
          f'lambda_traj={OPTIMAL_CONFIG["lambda_traj"]}')
    print(f'  Batches: {len(CONTINUOUS_BATCHES)} continuous feed')
    print('=' * 70)

    print('\n[1/3] Loading data + training TC-AT-LSTM ...')
    data = data_loader.load_preprocessed(MID_DATA / 'preprocessed_data.pkl')
    t_start = time.time()
    model = train_model(data)
    print(f'  Training complete in {(time.time() - t_start) / 60:.1f} min')

    print(f'\n[2/3] Building sliding windows (window={WINDOW_SIZE}) ...')
    all_windows, all_time_idx, all_batch_idx = build_windows(data['X_std_list'],
                                                               CONTINUOUS_BATCHES)
    windows_2d = all_windows.reshape(len(all_windows), -1)
    print(f'  Total windows: {len(all_windows)}')

    print(f'  Background (k-means {N_BACKGROUND} centroids) ...')
    import shap
    bg_2d = shap.kmeans(windows_2d, N_BACKGROUND)
    bg_2d = np.array(bg_2d.data)
    print(f'  Background: {bg_2d.shape}')

    print(f'\n[3/3] Computing SHAP (KernelExplainer, nsamples=auto) ...')

    def predict_fn(x_2d):
        x_3d = x_2d.reshape(-1, WINDOW_SIZE, 8)
        x_t = torch.tensor(x_3d, dtype=torch.float32)
        with torch.no_grad():
            out = model(x_t)
            return out[:, -1, :].cpu().numpy()

    e = shap.KernelExplainer(predict_fn, bg_2d)
    t_shap = time.time()

    sv_all = e.shap_values(windows_2d)
    if isinstance(sv_all, list):
        sv_all = sv_all[0]
    sv_feat = aggregate_shap(np.array(sv_all), len(all_windows))

    elapsed = (time.time() - t_shap) / 60
    print(f'  SHAP complete in {elapsed:.1f} min')

    shap_stack = []
    for bid in CONTINUOUS_BATCHES:
        mask = np.array(all_batch_idx) == bid
        shap_stack.append(sv_feat[mask])
    shap_stack = np.stack(shap_stack, axis=0)

    shap_mean = np.mean(shap_stack, axis=0)
    shap_std = np.std(shap_stack, axis=0)
    shap_abs_mean = np.mean(np.abs(shap_stack), axis=0)

    times = np.arange(5, 121)
    mask_p1 = times <= PHASE_SPLIT_HOUR
    mask_p2 = times > PHASE_SPLIT_HOUR
    p1_imp = np.mean(shap_abs_mean[mask_p1], axis=0)
    p2_imp = np.mean(shap_abs_mean[mask_p2], axis=0)
    global_imp = np.mean(shap_abs_mean, axis=0)
    rank_p1 = np.argsort(np.argsort(-p1_imp)) + 1
    rank_p2 = np.argsort(np.argsort(-p2_imp)) + 1

    print(f'\n  Global |SHAP| ranking:')
    for rank, idx in enumerate(np.argsort(-global_imp)):
        print(f'    {rank + 1}. {FEATURE_NAMES[idx]:>8}: {global_imp[idx]:.4f}  '
              f'(Phase I: {p1_imp[idx]:.4f} [#{rank_p1[idx]}], '
              f'Phase II: {p2_imp[idx]:.4f} [#{rank_p2[idx]}])')

    shap_per_batch = {}
    for bi, bid in enumerate(CONTINUOUS_BATCHES):
        shap_per_batch[bid] = shap_stack[bi]

    output = {
        'shap_mean': shap_mean, 'shap_std': shap_std, 'shap_abs_mean': shap_abs_mean,
        'phase1_importance': p1_imp, 'phase2_importance': p2_imp,
        'global_importance': global_imp, 'rank_phase1': rank_p1, 'rank_phase2': rank_p2,
        'shap_per_batch': shap_per_batch, 'feature_names': FEATURE_NAMES,
        'phase_split_hour': PHASE_SPLIT_HOUR, 'time_points': times, 'config': OPTIMAL_CONFIG,
        'n_batches': len(CONTINUOUS_BATCHES), 'window_size': WINDOW_SIZE,
        'method': f'KernelExplainer (sliding window, nsamples=auto, k-means {N_BACKGROUND})',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(MID_DATA / 'shap_results.pkl', 'wb') as f:
        pickle.dump(output, f)
    print(f'\nSaved: {MID_DATA / "shap_results.pkl"}')

    csv_path = MID_DATA / 'shap_data.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'time_h', 'phase', 'SHAP_mean', 'SHAP_std',
                    'abs_SHAP_mean', 'phase1_importance', 'phase2_importance',
                    'global_importance', 'rank_phase1', 'rank_phase2'])
        for fi, fname in enumerate(FEATURE_NAMES):
            for ti, t in enumerate(times):
                phase = 'Phase I (0-40h)' if t <= PHASE_SPLIT_HOUR else 'Phase II (40-120h)'
                w.writerow([fname, t, phase,
                           round(shap_mean[ti, fi], 6), round(shap_std[ti, fi], 6),
                           round(shap_abs_mean[ti, fi], 6),
                           round(p1_imp[fi], 6), round(p2_imp[fi], 6),
                           round(global_imp[fi], 6), rank_p1[fi], rank_p2[fi]])
    print(f'CSV (global mean): {csv_path}')

    X_std_list = data['X_std_list']
    beeswarm_path = MID_DATA / 'shap_beeswarm.csv'
    with open(beeswarm_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['batch_id', 'feature', 'time_h', 'phase', 'SHAP_value', 'feature_value'])
        for bid in BEESWARM_BATCHES:
            bid_mask = np.array(all_batch_idx) == bid
            bid_windows_2d = windows_2d[bid_mask]
            feat_vals = bid_windows_2d[:, 32:40]
            bid_shap = shap_per_batch[bid]
            for fi, fname in enumerate(FEATURE_NAMES):
                for ti, t in enumerate(times):
                    phase = 'Phase I (0-40h)' if t <= PHASE_SPLIT_HOUR else 'Phase II (40-120h)'
                    w.writerow([bid, fname, t, phase,
                               round(float(bid_shap[ti, fi]), 6),
                               round(float(feat_vals[ti, fi]), 6)])
    print(f'CSV (beeswarm): {beeswarm_path} ({5 * 116 * 8} rows)')
    print('=' * 70)
    print('5.1 SHAP computation complete.')


if __name__ == '__main__':
    main()