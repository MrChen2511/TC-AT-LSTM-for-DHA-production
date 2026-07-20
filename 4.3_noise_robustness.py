import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import pickle
import time
import warnings
from pathlib import Path
from scipy import stats as scipy_stats

import torch
from training import (_predict_dl_sequential, DEFAULT_CONFIG)
from models import get_dl_model
import data_loader

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
MID_DATA = ROOT / 'Mid_Data'

NOISE_LEVELS = [0.05, 0.10]
N_REPEATS = 10
RANDOM_SEED = 42

PULSE_BATCHES = [15, 16, 18, 19, 20]
CONTINUOUS_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]

OPTIMAL_CONFIGS = {
    'M-ATL':  {'attn_window': 5, 'hidden_dim': 128, 'lambda_traj': 0.0},
    'M-TCAL': {'attn_window': 5, 'hidden_dim': 128, 'lambda_traj': 0.002},
}


def classify_batch(bid):
    if bid in PULSE_BATCHES: return 'pulse'
    elif bid in CONTINUOUS_BATCHES: return 'continuous'
    return 'unknown'


def load_model_from_fold(model_name, fold_model_info):
    config_opt = OPTIMAL_CONFIGS[model_name]
    model = get_dl_model(model_name, input_dim=DEFAULT_CONFIG['input_dim'],
                         hidden_dim=config_opt['hidden_dim'],
                         num_layers=DEFAULT_CONFIG['num_layers'],
                         attn_window=config_opt['attn_window'])
    model.load_state_dict(fold_model_info['state_dict'])
    model.eval()
    return model


def compute_cv_ratio(r_S_clean, r_S_noisy_list):
    clean_std = np.std(r_S_clean)
    if clean_std < 1e-12:
        return np.nan
    noisy_all = np.concatenate([p.ravel() for p in r_S_noisy_list])
    return float(np.std(noisy_all) / clean_std)


def test_noise_on_batch(model, X_std_seq, rng, sigma_noise, n_repeats):
    X_t = torch.tensor(X_std_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        r_S_clean = model(X_t).squeeze(0).cpu().numpy()

    r_S_noisy_list = []
    T = X_std_seq.shape[0]
    for _ in range(n_repeats):
        noise = rng.randn(T, 8).astype(np.float32) * sigma_noise
        X_noisy = X_std_seq + noise
        X_noisy_t = torch.tensor(X_noisy, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            r_S_noisy = model(X_noisy_t).squeeze(0).cpu().numpy()
        r_S_noisy_list.append(r_S_noisy)

    cv = compute_cv_ratio(r_S_clean.ravel(), [p.ravel() for p in r_S_noisy_list])
    return r_S_clean.ravel(), r_S_noisy_list, cv


def run_noise_test(model_name, data, trajectory_data):
    X_std_list = data['X_std_list']
    fold_models = trajectory_data['trajectory'][model_name]['fold_models']
    all_per_batch = []
    rng = np.random.RandomState(RANDOM_SEED)

    for fold_info in fold_models:
        fold_id = fold_info['fold']
        val_bids = fold_info['val_bids']
        print(f'\n  Fold {fold_id}: val={val_bids}')
        model = load_model_from_fold(model_name, fold_info)

        for bid in val_bids:
            X_std_seq = X_std_list[bid - 1]
            feeding = classify_batch(bid)
            r_clean, _, _ = test_noise_on_batch(model, X_std_seq, rng, 0.0, 1)

            batch_result = {
                'batch_id': bid, 'fold': fold_id, 'feeding': feeding,
                'r_S_clean': r_clean, 'noise_levels': {},
            }

            for sigma in NOISE_LEVELS:
                _, r_noisy_list, cv = test_noise_on_batch(
                    model, X_std_seq, rng, sigma, N_REPEATS)
                batch_result['noise_levels'][sigma] = {
                    'sigma': sigma, 'CV_ratio': cv,
                    'CV_increase_pct': (cv - 1.0) * 100.0,
                }
                print(f'    Batch {bid:2d} [{feeding:>10}]  sigma={sigma:.2f}: '
                      f'CV_ratio={cv:.4f}  (+{(cv - 1.0) * 100:.1f}%)')

            all_per_batch.append(batch_result)

    summary = {}
    for group_name, group_bids in [('all', list(range(1, 21))),
                                    ('continuous', CONTINUOUS_BATCHES),
                                    ('pulse', PULSE_BATCHES)]:
        group_results = [b for b in all_per_batch if b['batch_id'] in group_bids]
        if not group_results:
            continue
        group_summary = {}
        for sigma in NOISE_LEVELS:
            cv_vals = [b['noise_levels'][sigma]['CV_ratio'] for b in group_results]
            cv_inc_vals = [b['noise_levels'][sigma]['CV_increase_pct'] for b in group_results]
            group_summary[sigma] = {
                'sigma': sigma,
                'CV_ratio_mean': float(np.mean(cv_vals)),
                'CV_ratio_std': float(np.std(cv_vals)),
                'CV_ratio_median': float(np.median(cv_vals)),
                'CV_increase_pct_mean': float(np.mean(cv_inc_vals)),
                'CV_increase_pct_std': float(np.std(cv_inc_vals)),
                'CV_increase_pct_median': float(np.median(cv_inc_vals)),
            }
        summary[group_name] = group_summary

    return {'model_name': model_name, 'per_batch': all_per_batch, 'summary': summary}


def main():
    print('=' * 70)
    print('  4.3 Noise Robustness Test')
    print(f'  Noise levels: {NOISE_LEVELS} (standardized space)')
    print(f'  Repeats per level: {N_REPEATS}')
    print('=' * 70)

    print('\n[1/3] Loading data ...')
    data = data_loader.load_preprocessed(MID_DATA / 'preprocessed_data.pkl')
    with open(MID_DATA / 'trajectory_results.pkl', 'rb') as f:
        trajectory_data = pickle.load(f)

    for mn in ['M-ATL', 'M-TCAL']:
        cfg = OPTIMAL_CONFIGS[mn]
        print(f'  {mn}: aw={cfg["attn_window"]}, hd={cfg["hidden_dim"]}, '
              f'lambda_traj={cfg.get("lambda_traj", 0)}')

    print(f'\n[2/3] Running noise tests ...')
    t_start = time.time()
    all_noise_results = {}

    for model_name in ['M-ATL', 'M-TCAL']:
        tag = 'TC-AT-LSTM' if model_name == 'M-TCAL' else 'M-ATL'
        print(f'\n{"-" * 60}')
        print(f'  {model_name} ({tag})')
        print(f'{"-" * 60}')

        results = run_noise_test(model_name, data, trajectory_data)
        all_noise_results[model_name] = results

        for group_name, label in [('continuous', 'Continuous'),
                                   ('pulse', 'Pulse'), ('all', 'All')]:
            if group_name not in results['summary']:
                continue
            gs = results['summary'][group_name]
            print(f'\n  {label}:')
            for sigma in NOISE_LEVELS:
                s = gs[sigma]
                print(f'    sigma={sigma:.2f}: CV_ratio={s["CV_ratio_median"]:.4f} (median)  '
                      f'mean={s["CV_ratio_mean"]:.4f}+/-{s["CV_ratio_std"]:.4f}')

    elapsed = (time.time() - t_start) / 60
    print(f'\n  Noise tests complete in {elapsed:.1f} min')

    print(f'\n[3/3] M-ATL vs M-TCAL comparison')
    atl_batches = {b['batch_id']: b for b in all_noise_results['M-ATL']['per_batch']}
    tcal_batches = {b['batch_id']: b for b in all_noise_results['M-TCAL']['per_batch']}

    for group_name, group_bids, label in [
        ('continuous', CONTINUOUS_BATCHES, 'Continuous'), ('all', list(range(1, 21)), 'All')]:
        valid_bids = [b for b in group_bids if b in atl_batches and b in tcal_batches]
        if len(valid_bids) < 2:
            continue
        print(f'\n  {label}:')
        for sigma in NOISE_LEVELS:
            atl_cv = np.array([atl_batches[bid]['noise_levels'][sigma]['CV_ratio']
                              for bid in valid_bids])
            tcal_cv = np.array([tcal_batches[bid]['noise_levels'][sigma]['CV_ratio']
                               for bid in valid_bids])
            _, p_val = scipy_stats.ttest_rel(tcal_cv, atl_cv)
            reduction = (atl_cv.mean() - tcal_cv.mean()) / atl_cv.mean() * 100.0
            print(f'    sigma={sigma:.2f}: M-ATL CV={atl_cv.mean():.4f}, '
                  f'M-TCAL CV={tcal_cv.mean():.4f}, reduction={reduction:+.1f}%, '
                  f'p={p_val:.4f}')

    noise_results = {
        'noise_test': all_noise_results,
        'config': {'noise_levels': NOISE_LEVELS, 'n_repeats': N_REPEATS,
                   'random_seed': RANDOM_SEED},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(MID_DATA / 'noise_results.pkl', 'wb') as f:
        pickle.dump(noise_results, f)
    print(f'\nSaved: {MID_DATA / "noise_results.pkl"}')

    total_elapsed = (time.time() - t_start) / 60
    print(f'\nTotal time: {total_elapsed:.1f} min')
    print('=' * 70)


if __name__ == '__main__':
    main()