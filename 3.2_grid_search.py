import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import pickle
import time
import warnings
from pathlib import Path

import training
from training import (train_dl_sequential, _predict_dl_sequential, _prepare_sequences,
                       compute_metrics, DEFAULT_CONFIG, _split_folds_stratified)
from models import get_dl_model
import data_loader

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
MID_DATA = ROOT / 'Mid_Data'

ATTN_WINDOW_LIST = [3, 5, 10, 15]
HIDDEN_DIM_LIST = [32, 64, 96, 128]
LAMBDA_TRAJ_LIST = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
DEFAULT_LAMBDA_TRAJ = 0.001


def evaluate_fold(model, X_std_list, y_std_list, val_bids, scaler_y):
    pred_std = _predict_dl_sequential(model, X_std_list, val_bids)
    pred_std_all = np.concatenate([p for p in pred_std if len(p) > 0], axis=0)
    true_std_all = np.concatenate([y_std_list[b - 1].reshape(-1, 1) for b in val_bids], axis=0)

    pred_raw = scaler_y.inverse_transform(pred_std_all).ravel()
    pred_raw = np.maximum(pred_raw, 0.0)
    true_raw = scaler_y.inverse_transform(true_std_all).ravel()

    metrics = compute_metrics(true_raw, pred_raw)
    return metrics, pred_raw, true_raw


def run_arch_search_5fold(model_name, data, fold_splits, attn_list, hd_list, lambda_traj=0.0):
    X_std_list = data['X_std_list']
    y_std_list = data['y_std_list']
    scaler_y = data['scaler_y']
    D_raw_list = data.get('D_list', None)
    Glu_raw_list = data.get('Glu_list', None)
    config_base = DEFAULT_CONFIG.copy()

    grid_results = []
    best_r2 = -np.inf
    best_config = None
    n_total = len(attn_list) * len(hd_list)
    n_done = 0

    for aw in attn_list:
        for hd in hd_list:
            n_done += 1
            tcl_str = f'lambda_traj={lambda_traj}' if model_name == 'M-TCAL' else ''
            print(f'\n  [{model_name}] ({n_done}/{n_total}) '
                  f'attn_window={aw}, hidden_dim={hd}  {tcl_str}')

            fold_r2_vals, fold_mae_vals = [], []
            is_default_config = (aw == 5 and hd == 64)
            fold_preds, fold_trues = [], []

            for fold_idx, (train_bids, val_bids) in enumerate(fold_splits):
                train_seqs = _prepare_sequences(X_std_list, y_std_list, train_bids)

                config = config_base.copy()
                config['hidden_dim'] = hd
                config['lambda_traj'] = lambda_traj

                model = get_dl_model(model_name, input_dim=config['input_dim'],
                                     hidden_dim=hd, num_layers=config['num_layers'],
                                     attn_window=aw)

                model = train_dl_sequential(model, train_seqs,
                                             X_std_list, y_std_list, val_bids,
                                             lambda_traj=lambda_traj,
                                             D_raw_list=D_raw_list,
                                             Glu_raw_list=Glu_raw_list,
                                             scaler_y=scaler_y,
                                             train_bids=train_bids,
                                             config=config)

                metrics_fold, pred_raw, true_raw = evaluate_fold(
                    model, X_std_list, y_std_list, val_bids, scaler_y)
                fold_r2_vals.append(metrics_fold['R2'])
                fold_mae_vals.append(metrics_fold['MAE'])

                if is_default_config:
                    fold_preds.append(pred_raw)
                    fold_trues.append(true_raw)

                print(f'    Fold {fold_idx + 1}: R2={metrics_fold["R2"]:.4f}')

            mean_r2 = float(np.mean(fold_r2_vals))
            std_r2 = float(np.std(fold_r2_vals))
            mean_mae = float(np.mean(fold_mae_vals))
            std_mae = float(np.std(fold_mae_vals))

            result = {
                'attn_window': aw, 'hidden_dim': hd, 'lambda_traj': lambda_traj,
                'R2_mean': mean_r2, 'R2_std': std_r2, 'R2_per_fold': fold_r2_vals,
                'MAE_mean': mean_mae, 'MAE_std': std_mae, 'MAE_per_fold': fold_mae_vals,
            }
            if is_default_config:
                result['predictions'] = fold_preds
                result['targets'] = fold_trues
            grid_results.append(result)
            print(f'    => 5-fold avg R2 = {mean_r2:.4f} +/- {std_r2:.4f}, MAE = {mean_mae:.4f}')

            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config = result.copy()

    print(f'\n  -> {model_name} best: attn_window={best_config["attn_window"]}, '
          f'hidden_dim={best_config["hidden_dim"]}, '
          f'5-fold avg R2={best_config["R2_mean"]:.4f}')
    return grid_results, best_config


def run_lambda_sweep_5fold(data, fold_splits, attn_window, hidden_dim, lambda_list):
    X_std_list = data['X_std_list']
    y_std_list = data['y_std_list']
    scaler_y = data['scaler_y']
    D_raw_list = data.get('D_list', None)
    Glu_raw_list = data.get('Glu_list', None)
    config_base = DEFAULT_CONFIG.copy()

    lambda_results = []

    for lidx, lam in enumerate(lambda_list):
        tag = 'ablation baseline' if lam == 0 else f'lambda_traj={lam}'
        print(f'\n  [M-TCAL lambda sweep] ({lidx + 1}/{len(lambda_list)}) {tag}')

        fold_r2_vals, fold_mae_vals = [], []

        for fold_idx, (train_bids, val_bids) in enumerate(fold_splits):
            train_seqs = _prepare_sequences(X_std_list, y_std_list, train_bids)

            config = config_base.copy()
            config['hidden_dim'] = hidden_dim
            config['lambda_traj'] = lam

            model = get_dl_model('M-TCAL', input_dim=config['input_dim'],
                                 hidden_dim=hidden_dim, num_layers=config['num_layers'],
                                 attn_window=attn_window)

            model = train_dl_sequential(model, train_seqs,
                                         X_std_list, y_std_list, val_bids,
                                         lambda_traj=lam,
                                         D_raw_list=D_raw_list,
                                         Glu_raw_list=Glu_raw_list,
                                         scaler_y=scaler_y,
                                         train_bids=train_bids,
                                         config=config)

            metrics_fold, _, _ = evaluate_fold(model, X_std_list, y_std_list, val_bids, scaler_y)
            fold_r2_vals.append(metrics_fold['R2'])
            fold_mae_vals.append(metrics_fold['MAE'])
            print(f'    Fold {fold_idx + 1}: R2={metrics_fold["R2"]:.4f}, MAE={metrics_fold["MAE"]:.4f}')

        result = {
            'attn_window': attn_window, 'hidden_dim': hidden_dim, 'lambda_traj': lam,
            'R2_mean': float(np.mean(fold_r2_vals)), 'R2_std': float(np.std(fold_r2_vals)),
            'MAE_mean': float(np.mean(fold_mae_vals)), 'MAE_std': float(np.std(fold_mae_vals)),
            'R2_per_fold': fold_r2_vals, 'physical_ratio': 1.0,
        }
        lambda_results.append(result)
        print(f'    => 5-fold avg R2 = {result["R2_mean"]:.4f} +/- {result["R2_std"]:.4f}')

    return lambda_results


def main():
    print('=' * 70)
    print('  3.2 M-ATL and M-TCAL Grid Search (5-fold averaged)')
    print(f'  M-TCAL Step 1 fixed lambda_traj = {DEFAULT_LAMBDA_TRAJ}')
    print('=' * 70)

    print('\nLoading data ...')
    data = data_loader.load_preprocessed(MID_DATA / 'preprocessed_data.pkl')
    fold_splits = _split_folds_stratified()
    print(f'  5-fold splits:')
    for fi, (tr, va) in enumerate(fold_splits):
        print(f'    Fold {fi + 1}: train={tr}, val={va}')

    t_start = time.time()

    print(f'\n{"=" * 70}')
    print(f'  [Step 1a] M-ATL architecture search (5-fold avg)')
    print(f'           attn_window={ATTN_WINDOW_LIST}, hidden_dim={HIDDEN_DIM_LIST}')
    print(f'{"=" * 70}')

    atl_grid, best_atl = run_arch_search_5fold(
        'M-ATL', data, fold_splits, ATTN_WINDOW_LIST, HIDDEN_DIM_LIST, lambda_traj=0.0)

    print(f'\n{"=" * 70}')
    print(f'  [Step 1b] M-TCAL architecture search (5-fold avg, lambda_traj={DEFAULT_LAMBDA_TRAJ})')
    print(f'           attn_window={ATTN_WINDOW_LIST}, hidden_dim={HIDDEN_DIM_LIST}')
    print(f'{"=" * 70}')

    tcal_step1_grid, best_tcal_step1 = run_arch_search_5fold(
        'M-TCAL', data, fold_splits, ATTN_WINDOW_LIST, HIDDEN_DIM_LIST,
        lambda_traj=DEFAULT_LAMBDA_TRAJ)
    aw_star = best_tcal_step1['attn_window']
    hd_star = best_tcal_step1['hidden_dim']

    print(f'\n{"=" * 70}')
    print(f'  [Step 2] M-TCAL lambda_traj coarse search (5-fold avg)')
    print(f'           aw={aw_star}, hd={hd_star}')
    print(f'           lambda_traj={LAMBDA_TRAJ_LIST}')
    print(f'{"=" * 70}')

    tcal_step2_sweep = run_lambda_sweep_5fold(
        data, fold_splits, aw_star, hd_star, LAMBDA_TRAJ_LIST)

    elapsed = (time.time() - t_start) / 60

    grid_results = {
        'ATL_grid': atl_grid, 'best_ATL': best_atl,
        'TCAL_step1_grid': tcal_step1_grid, 'best_TCAL_step1': best_tcal_step1,
        'TCAL_step2_lambda_sweep': tcal_step2_sweep,
        'fold_splits': [(list(tr), list(va)) for tr, va in fold_splits],
        'config': {
            'attn_window_list': ATTN_WINDOW_LIST, 'hidden_dim_list': HIDDEN_DIM_LIST,
            'lambda_traj_list': LAMBDA_TRAJ_LIST, 'default_lambda_traj': DEFAULT_LAMBDA_TRAJ,
            'n_folds': len(fold_splits),
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    output_path = MID_DATA / 'grid_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(grid_results, f)
    print(f'\nSaved: {output_path}')

    print('\n' + '=' * 70)
    print('  Grid Search Summary (5-fold avg)')
    print('=' * 70)
    print(f'  M-ATL best:   aw={best_atl["attn_window"]}, hd={best_atl["hidden_dim"]}, '
          f'R2={best_atl["R2_mean"]:.4f}+/-{best_atl["R2_std"]:.4f}')
    print(f'  M-TCAL best:  aw={best_tcal_step1["attn_window"]}, '
          f'hd={best_tcal_step1["hidden_dim"]}, '
          f'R2={best_tcal_step1["R2_mean"]:.4f}+/-{best_tcal_step1["R2_std"]:.4f}')
    print(f'  lambda_traj coarse range: {LAMBDA_TRAJ_LIST}')
    print(f'  Total time: {elapsed:.1f} min')
    print('=' * 70)
    print('3.2 Grid search complete.')


if __name__ == '__main__':
    main()