import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import pickle
import time
import csv
import warnings
from pathlib import Path

import training
from training import (train_dl_sequential, _predict_dl_sequential, _prepare_sequences,
                       compute_metrics, DEFAULT_CONFIG, _split_folds_stratified)
from models import get_dl_model, compute_trajectory
import data_loader

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
MID_DATA = ROOT / 'Mid_Data'

LAMBDA_LIST = [0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
TRAJECTORY_START_HOUR = 21
TRAJECTORY_START_IDX = TRAJECTORY_START_HOUR - 1


def evaluate_fold(model, X_std_list, y_std_list, val_bids, scaler_y,
                  D_list, Glu_list, S_init_list):
    pred_std = _predict_dl_sequential(model, X_std_list, val_bids)
    pred_std_all = np.concatenate([p for p in pred_std if len(p) > 0], axis=0)
    true_std_all = np.concatenate([y_std_list[b - 1].reshape(-1, 1) for b in val_bids], axis=0)

    pred_raw_all = scaler_y.inverse_transform(pred_std_all).ravel()
    pred_raw_all = np.maximum(pred_raw_all, 0.0)
    true_raw_all = scaler_y.inverse_transform(true_std_all).ravel()

    r2 = compute_metrics(true_raw_all, pred_raw_all)['R2']

    mse_traj_vals, endpoint_vals = [], []
    for bidx, bid in enumerate(val_bids):
        pred_std_batch = pred_std[bidx]
        pred_raw = scaler_y.inverse_transform(pred_std_batch).ravel()
        pred_raw = np.maximum(pred_raw, 0.0)
        Glu_seq = Glu_list[bid - 1]
        S_hat, mse = compute_trajectory(pred_raw, D_list[bid - 1], Glu_seq,
                                        S_init_list[bid - 1], C_feed=800.0,
                                        start_idx=TRAJECTORY_START_IDX)
        mse_traj_vals.append(mse)
        endpoint_vals.append(abs(S_hat[-1] - Glu_seq[-1]))

    return (float(np.mean(mse_traj_vals)), float(np.mean(endpoint_vals)), r2)


def main():
    print('=' * 70)
    print('  4.1 lambda_traj Fine Optimization (5-fold avg, MSE_traj guided)')
    print(f'  Architecture: aw=5, hd=128 (Fig 4 unified)')
    print(f'  Search range: {LAMBDA_LIST}')
    print('=' * 70)

    print('\n[1/3] Loading data ...')
    data = data_loader.load_preprocessed(MID_DATA / 'preprocessed_data.pkl')
    fold_splits = _split_folds_stratified()

    X_std_list = data['X_std_list']
    y_std_list = data['y_std_list']
    scaler_y = data['scaler_y']
    D_list = data['D_list']
    Glu_list = data['Glu_list']
    S_init_list = data['S_init_list']

    print(f'  5-fold splits:')
    for fi, (tr, va) in enumerate(fold_splits):
        print(f'    Fold {fi + 1}: train={tr}, val={va}')

    print(f'\n[2/3] Searching lambda_traj ({len(LAMBDA_LIST)} values x 5 folds) ...')
    t_start = time.time()

    results = []

    for lidx, lam in enumerate(LAMBDA_LIST):
        tag = '(ablation baseline)' if lam == 0 else ''
        print(f'\n  [{lidx + 1}/{len(LAMBDA_LIST)}] lambda_traj = {lam} {tag}')

        fold_mse_vals, fold_endp_vals, fold_r2_vals = [], [], []

        for fold_idx, (train_bids, val_bids) in enumerate(fold_splits):
            train_seqs = _prepare_sequences(X_std_list, y_std_list, train_bids)

            config = DEFAULT_CONFIG.copy()
            config['hidden_dim'] = 128
            config['lambda_traj'] = lam

            model = get_dl_model('M-TCAL', input_dim=config['input_dim'],
                                 hidden_dim=128, num_layers=config['num_layers'],
                                 attn_window=5)

            model = train_dl_sequential(model, train_seqs,
                                         X_std_list, y_std_list, val_bids,
                                         lambda_traj=lam,
                                         D_raw_list=D_list, Glu_raw_list=Glu_list,
                                         scaler_y=scaler_y, train_bids=train_bids,
                                         config=config)

            mse_fold, endp_fold, r2_fold = evaluate_fold(
                model, X_std_list, y_std_list, val_bids, scaler_y,
                D_list, Glu_list, S_init_list)
            fold_mse_vals.append(mse_fold)
            fold_endp_vals.append(endp_fold)
            fold_r2_vals.append(r2_fold)
            print(f'    Fold {fold_idx + 1}: MSE_traj={mse_fold:.2f}, '
                  f'AEE={endp_fold:.2f}, R2={r2_fold:.4f}')

        results.append({
            'lambda_traj': lam,
            'MSE_mean': float(np.mean(fold_mse_vals)),
            'MSE_std': float(np.std(fold_mse_vals)),
            'ENDP_mean': float(np.mean(fold_endp_vals)),
            'ENDP_std': float(np.std(fold_endp_vals)),
            'R2_mean': float(np.mean(fold_r2_vals)),
            'R2_std': float(np.std(fold_r2_vals)),
        })
        print(f'    => 5-fold: MSE_traj = {results[-1]["MSE_mean"]:.2f} +/- '
              f'{results[-1]["MSE_std"]:.2f}, AEE = {results[-1]["ENDP_mean"]:.2f} +/- '
              f'{results[-1]["ENDP_std"]:.2f}')

    elapsed = (time.time() - t_start) / 60
    print(f'\n  Search complete in {elapsed:.1f} min')

    best_idx = np.argmin([r['MSE_mean'] for r in results])
    best = results[best_idx]
    baseline = results[0]
    improvement = (baseline['MSE_mean'] - best['MSE_mean']) / baseline['MSE_mean'] * 100

    print(f'\n[3/3] Optimal Results')
    print('=' * 80)
    print(f'  {"lambda":<10} {"MSE_mean":>10} {"MSE_std":>8} {"AEE_mean":>10} {"Note"}')
    print('  ' + '-' * 60)
    for r in results:
        m = ' <-- best' if r['lambda_traj'] == best['lambda_traj'] else ''
        b = ' (baseline)' if r['lambda_traj'] == 0 else ''
        print(f'  {r["lambda_traj"]:<10.4f} {r["MSE_mean"]:>10.2f} {r["MSE_std"]:>8.2f} '
              f'{r["ENDP_mean"]:>10.2f}{m}{b}')
    print(f'\n  Best lambda_traj = {best["lambda_traj"]}')
    print(f'  MSE_traj = {best["MSE_mean"]:.2f} +/- {best["MSE_std"]:.2f}')
    print(f'  R2 = {best["R2_mean"]:.4f} +/- {best["R2_std"]:.4f}')
    print(f'  MSE_traj improvement = {improvement:.1f}% (vs lambda=0)')

    output = {
        'lambda_list': LAMBDA_LIST, 'results': results,
        'best_lambda': best['lambda_traj'], 'best_mse_traj': best['MSE_mean'],
        'improvement_pct': improvement,
        'config': {'attn_window': 5, 'hidden_dim': 128, 'n_folds': len(fold_splits),
                   'eval_window': f'{TRAJECTORY_START_HOUR}-120 h',
                   'metric': 'MSE_traj (5-fold avg)'},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(MID_DATA / 'lambda_opt_results.pkl', 'wb') as f:
        pickle.dump(output, f)
    print(f'\nSaved: {MID_DATA / "lambda_opt_results.pkl"}')

    csv_path = MID_DATA / 'fig4a_lambda_data.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lambda_traj', 'MSE_mean', 'MSE_std', 'ENDP_mean', 'ENDP_std',
                     'R2_mean', 'R2_std'])
        for r in results:
            w.writerow([r['lambda_traj'], r['MSE_mean'], r['MSE_std'],
                       r['ENDP_mean'], r['ENDP_std'], r['R2_mean'], r['R2_std']])
    print(f'CSV: {csv_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()