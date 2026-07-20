import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import pickle
import time
import warnings
from pathlib import Path
from scipy import stats as scipy_stats

import training
from training import (train_dl_sequential, _predict_dl_sequential, _prepare_sequences,
                       compute_metrics, DEFAULT_CONFIG, _split_folds_stratified)
from models import get_dl_model, compute_trajectory
import data_loader

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
MID_DATA = ROOT / 'Mid_Data'

PULSE_BATCHES = [15, 16, 18, 19, 20]
CONTINUOUS_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]
TRAJECTORY_START_HOUR = 21
TRAJECTORY_START_IDX = TRAJECTORY_START_HOUR - 1


def compute_trajectory_metrics(r_S_hat_raw, D_seq, Glu_seq, S_init, C_feed=800.0,
                               start_hour=TRAJECTORY_START_HOUR):
    start_idx = start_hour - 1
    S_hat, MSE_traj = compute_trajectory(r_S_hat_raw, D_seq, Glu_seq, S_init,
                                         C_feed, start_idx=start_idx)
    endpoint_error = float(S_hat[-1] - Glu_seq[-1])
    valid_mask = ~(np.isnan(Glu_seq) | np.isnan(S_hat))
    Glu_valid = Glu_seq[valid_mask]
    S_hat_valid = S_hat[valid_mask]
    if len(Glu_valid) > 1:
        MAE_traj = np.mean(np.abs(Glu_valid - S_hat_valid))
        ss_res = np.sum((Glu_valid - S_hat_valid) ** 2)
        ss_tot = np.sum((Glu_valid - Glu_valid.mean()) ** 2)
        R2_traj = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    else:
        MAE_traj, R2_traj = np.nan, np.nan
    return {'S_hat': S_hat, 'MSE_traj': MSE_traj, 'MAE_traj': MAE_traj,
            'R2_traj': R2_traj, 'endpoint_error': endpoint_error,
            'eval_window': f'{start_hour}-120 h', 'eval_n_points': len(Glu_valid)}


def classify_batch(bid):
    if bid in PULSE_BATCHES: return 'pulse'
    elif bid in CONTINUOUS_BATCHES: return 'continuous'
    return 'unknown'


def group_summary(values, name):
    arr = np.array(values)
    return {f'{name}_mean': float(np.mean(arr)), f'{name}_std': float(np.std(arr)),
            f'{name}_median': float(np.median(arr)), f'{name}_min': float(np.min(arr)),
            f'{name}_max': float(np.max(arr)), 'count': len(arr)}


def run_trajectory_cv(model_name, data, fold_splits, config_opt, start_hour=TRAJECTORY_START_HOUR):
    X_std_list = data['X_std_list']
    y_std_list = data['y_std_list']
    scaler_y = data['scaler_y']
    D_list = data['D_list']
    Glu_list = data['Glu_list']
    S_init_list = data['S_init_list']

    hidden_dim = config_opt['hidden_dim']
    num_layers = DEFAULT_CONFIG['num_layers']
    lambda_traj = config_opt.get('lambda_traj', 0.0)
    attn_window = config_opt['attn_window']

    fold_results = []
    all_fold_models = []
    all_per_batch = []

    for fold_idx, (train_bids, val_bids) in enumerate(fold_splits):
        print(f'\n  Fold {fold_idx + 1}/{len(fold_splits)}: '
              f'train={train_bids}, val={val_bids}')
        if model_name == 'M-TCAL' and lambda_traj > 0:
            print(f'    TC-AT-LSTM: lambda_traj = {lambda_traj}')

        train_seqs = _prepare_sequences(X_std_list, y_std_list, train_bids)

        model = get_dl_model(model_name, input_dim=DEFAULT_CONFIG['input_dim'],
                             hidden_dim=hidden_dim, num_layers=num_layers,
                             attn_window=attn_window)

        config_train = DEFAULT_CONFIG.copy()
        config_train['hidden_dim'] = hidden_dim
        config_train['lambda_traj'] = lambda_traj
        model = train_dl_sequential(model, train_seqs,
                                     X_std_list, y_std_list, val_bids,
                                     lambda_traj=lambda_traj,
                                     D_raw_list=D_list, Glu_raw_list=Glu_list,
                                     scaler_y=scaler_y, train_bids=train_bids,
                                     config=config_train)

        model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        all_fold_models.append({
            'fold': fold_idx + 1, 'train_bids': list(train_bids),
            'val_bids': list(val_bids), 'state_dict': model_state,
            'lambda_traj': lambda_traj,
        })

        pred_std = _predict_dl_sequential(model, X_std_list, val_bids)
        fold_batch_results = []
        fold_preds_raw_all, fold_trues_raw_all = [], []

        for bidx, bid in enumerate(val_bids):
            pred_std_batch = pred_std[bidx]
            pred_raw = scaler_y.inverse_transform(pred_std_batch).ravel()
            pred_raw = np.maximum(pred_raw, 0.0)

            D_seq = D_list[bid - 1]
            Glu_seq = Glu_list[bid - 1]
            S_init = S_init_list[bid - 1]
            y_true_raw = scaler_y.inverse_transform(
                y_std_list[bid - 1].reshape(-1, 1)).ravel()

            rS_metrics = compute_metrics(y_true_raw, pred_raw)
            traj_metrics = compute_trajectory_metrics(pred_raw, D_seq, Glu_seq, S_init,
                                                       start_hour=start_hour)

            batch_result = {
                'batch_id': bid, 'fold': fold_idx + 1, 'feeding': classify_batch(bid),
                'r_S_hat': pred_raw, 'r_S_true': y_true_raw,
                'S_hat': traj_metrics['S_hat'], 'Glu': Glu_seq,
                'D_seq': D_seq, 'S_init': S_init,
                'rS_MAE': rS_metrics['MAE'], 'rS_MSE': rS_metrics['MSE'],
                'rS_R2': rS_metrics['R2'],
                'MSE_traj': traj_metrics['MSE_traj'], 'MAE_traj': traj_metrics['MAE_traj'],
                'R2_traj': traj_metrics['R2_traj'], 'endpoint_error': traj_metrics['endpoint_error'],
            }
            fold_batch_results.append(batch_result)
            fold_preds_raw_all.append(pred_raw)
            fold_trues_raw_all.append(y_true_raw)

            print(f'    Batch {bid:2d} [{batch_result["feeding"]:>10}]: '
                  f'rS_R2={rS_metrics["R2"]:.4f}, '
                  f'MSE_traj(21-120h)={traj_metrics["MSE_traj"]:.2f}, '
                  f'endpoint_err={traj_metrics["endpoint_error"]:+.2f} g/L')

        preds_all = np.concatenate(fold_preds_raw_all)
        trues_all = np.concatenate(fold_trues_raw_all)
        fold_metrics = compute_metrics(trues_all, preds_all)

        fold_mse_traj_vals = [b['MSE_traj'] for b in fold_batch_results]
        fold_r2_traj_vals = [b['R2_traj'] for b in fold_batch_results]
        fold_endpoint_vals = [b['endpoint_error'] for b in fold_batch_results]

        fold_results.append({
            'fold_idx': fold_idx + 1, 'train_bids': list(train_bids),
            'val_bids': list(val_bids), 'lambda_traj': lambda_traj,
            'rS_MAE': fold_metrics['MAE'], 'rS_MSE': fold_metrics['MSE'],
            'rS_R2': fold_metrics['R2'],
            'MSE_traj_mean': float(np.mean(fold_mse_traj_vals)),
            'MSE_traj_std': float(np.std(fold_mse_traj_vals)),
            'R2_traj_mean': float(np.mean(fold_r2_traj_vals)),
            'R2_traj_std': float(np.std(fold_r2_traj_vals)),
            'endpoint_error_mean': float(np.mean(fold_endpoint_vals)),
            'endpoint_error_std': float(np.std(fold_endpoint_vals)),
        })
        all_per_batch.extend(fold_batch_results)

    all_bids = [b['batch_id'] for b in all_per_batch]
    summary = {}
    for group_name, group_bids in [('all', all_bids), ('continuous', CONTINUOUS_BATCHES),
                                    ('pulse', PULSE_BATCHES)]:
        group_results = [b for b in all_per_batch if b['batch_id'] in group_bids]
        if not group_results:
            continue
        mse_vals = [b['MSE_traj'] for b in group_results]
        mae_vals = [b['MAE_traj'] for b in group_results]
        r2_vals = [b['R2_traj'] for b in group_results]
        ep_vals = [b['endpoint_error'] for b in group_results]
        ep_abs_vals = [abs(e) for e in ep_vals]
        summary[group_name] = {
            **group_summary(mse_vals, 'MSE_traj'), **group_summary(mae_vals, 'MAE_traj'),
            **group_summary(r2_vals, 'R2_traj'), **group_summary(ep_vals, 'endpoint_error'),
            **group_summary(ep_abs_vals, 'endpoint_abs_error'),
        }

    return {'model_name': model_name, 'config': config_opt,
            'fold_details': fold_results, 'per_batch': all_per_batch,
            'summary': summary, 'fold_models': all_fold_models}


def main():
    print('=' * 70)
    print('  4.2 Trajectory Consistency Validation (aw=5, hd=128)')
    print(f'  M-ATL + M-TCAL (TC-AT-LSTM) 5-fold LOBO-CV')
    print(f'  Eval window: {TRAJECTORY_START_HOUR}-120 h')
    print('=' * 70)

    print('\n[1/3] Loading data ...')
    data = data_loader.load_preprocessed(MID_DATA / 'preprocessed_data.pkl')
    fold_splits = _split_folds_stratified()

    configs = {
        'M-ATL': {'attn_window': 5, 'hidden_dim': 128, 'lambda_traj': 0.0},
        'M-TCAL': {'attn_window': 5, 'hidden_dim': 128, 'lambda_traj': 0.002},
    }

    print(f'  Batches: {len(data["batch_ids"])}')
    print(f'  Continuous: {len(CONTINUOUS_BATCHES)}, Pulse: {len(PULSE_BATCHES)}')

    print(f'\n[2/3] Running 5-fold LOBO-CV + trajectory integration ...')
    t_start = time.time()
    all_trajectory = {}

    for model_name in ['M-ATL', 'M-TCAL']:
        cfg = configs[model_name]
        tag = 'TC-AT-LSTM' if model_name == 'M-TCAL' else 'M-ATL'
        print(f'\n{"-" * 60}')
        print(f'  {model_name} ({tag}) aw={cfg["attn_window"]}, hd={cfg["hidden_dim"]}, '
              f'lambda_traj={cfg["lambda_traj"]}')
        print(f'{"-" * 60}')

        cv_results = run_trajectory_cv(model_name, data, fold_splits, cfg)
        all_trajectory[model_name] = cv_results

        for group_name, label in [('continuous', 'Continuous (15 batches)'),
                                   ('pulse', 'Pulse (5 batches)'), ('all', 'All (20 batches)')]:
            if group_name not in cv_results['summary']:
                continue
            s = cv_results['summary'][group_name]
            print(f'\n  {label}:')
            print(f'    MSE_traj = {s["MSE_traj_median"]:.2f} (median)')
            print(f'    Endpoint abs error = {s["endpoint_abs_error_median"]:.2f} g/L (median)')

    elapsed_cv = (time.time() - t_start) / 60
    print(f'\n  LOBO-CV complete in {elapsed_cv:.1f} min')

    print(f'\n[3/3] Paired comparison (M-ATL vs M-TCAL)')
    atl_batches = {b['batch_id']: b for b in all_trajectory['M-ATL']['per_batch']}
    tcal_batches = {b['batch_id']: b for b in all_trajectory['M-TCAL']['per_batch']}
    common_bids = sorted(set(atl_batches.keys()) & set(tcal_batches.keys()))

    for group_name, group_bids, label in [
        ('continuous', [b for b in common_bids if b in CONTINUOUS_BATCHES], 'Continuous'),
        ('pulse', [b for b in common_bids if b in PULSE_BATCHES], 'Pulse'),
        ('all', common_bids, 'All'),
    ]:
        atl_mse = np.array([atl_batches[bid]['MSE_traj'] for bid in group_bids])
        tcal_mse = np.array([tcal_batches[bid]['MSE_traj'] for bid in group_bids])
        t_stat, p_val = scipy_stats.ttest_rel(tcal_mse, atl_mse)
        diff = tcal_mse - atl_mse
        reduction = (atl_mse.mean() - tcal_mse.mean()) / atl_mse.mean() * 100

        print(f'\n  {label}:')
        print(f'    M-ATL MSE_traj = {atl_mse.mean():.2f}')
        print(f'    M-TCAL MSE_traj = {tcal_mse.mean():.2f}')
        print(f'    Reduction = {reduction:+.1f}%')
        print(f'    t = {t_stat:.4f}, p = {p_val:.6f}')

    print(f'\n  Per-batch MSE_traj ({TRAJECTORY_START_HOUR}-120h) comparison:')
    print(f'  {"Batch":<6} {"Feeding":<10} {"M-ATL":>12} {"M-TCAL":>12} {"Delta":>12}')
    print(f'  {"-" * 52}')
    for bid in common_bids:
        feeding = classify_batch(bid)
        mse_a = atl_batches[bid]['MSE_traj']
        mse_t = tcal_batches[bid]['MSE_traj']
        d = mse_t - mse_a
        print(f'  {bid:<6} {feeding:<10} {mse_a:>12.2f} {mse_t:>12.2f} {d:>+12.2f}')

    trajectory_results = {
        'trajectory': all_trajectory,
        'paired_tests': {},
        'config': {'optimal_configs': configs, 'n_folds': len(fold_splits),
                   'pulse_batches': PULSE_BATCHES, 'continuous_batches': CONTINUOUS_BATCHES,
                   'trajectory_start_hour': TRAJECTORY_START_HOUR,
                   'eval_window': f'{TRAJECTORY_START_HOUR}-120 h'},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    output_path = MID_DATA / 'trajectory_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(trajectory_results, f)
    print(f'\nSaved: {output_path}')

    total_elapsed = (time.time() - t_start) / 60
    print(f'\nTotal time: {total_elapsed:.1f} min')
    print('=' * 70)
    print('4.2 Trajectory validation complete.')


if __name__ == '__main__':
    main()