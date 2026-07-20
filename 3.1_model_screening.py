import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import numpy as np
import pickle
import time
import warnings
from pathlib import Path

import training
from training import run_lobo_cv, DEFAULT_CONFIG, _split_folds_stratified
import data_loader

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
MID_DATA = ROOT / 'Mid_Data'

MODEL_CONFIGS = [
    ('M-LP',  'mechanistic', {}),
    ('M-RF',  'ml',          {}),
    ('M-XGB', 'ml',          {}),
    ('M-LGB', 'ml',          {}),
    ('M-SVR', 'ml',          {}),
    ('M-RNN', 'dl',          {'attn_window': 5, 'hidden_dim': 64, 'num_layers': 2}),
    ('M-LSTM','dl',          {'attn_window': 5, 'hidden_dim': 64, 'num_layers': 2}),
    ('M-CNNL','dl',          {'attn_window': 5, 'hidden_dim': 64, 'num_layers': 2}),
]


def main():
    print('=' * 70)
    print('  3.1 Model Screening (8 models, 5-fold LOBO-CV)')
    print('=' * 70)

    print('\nLoading data ...')
    data = data_loader.load_preprocessed(MID_DATA / 'preprocessed_data.pkl')
    fold_splits = _split_folds_stratified()

    all_results = {'models': {}, 'fold_splits': fold_splits}
    t_start = time.time()

    for model_name, category, params in MODEL_CONFIGS:
        print(f'\n{"=" * 60}')
        print(f'  {model_name} ({category})')
        print(f'{"=" * 60}')

        config = DEFAULT_CONFIG.copy()
        config.update(params)

        result = run_lobo_cv(model_name, data, config=config, fold_splits=fold_splits)

        all_results['models'][model_name] = {
            'category': category,
            'summary_r2': result['summary_r2'],
            'summary_mae': result['summary_mae'],
            'summary_mse': result['summary_mse'],
            'predictions': result['predictions'],
            'targets': result['targets'],
        }

        r2 = result['summary_r2']
        print(f'  R2 = {r2[0]:.4f} +/- {r2[1]:.4f}')

    elapsed = (time.time() - t_start) / 60
    print(f'\n  Screening complete in {elapsed:.1f} minutes')

    output_path = MID_DATA / 'screening_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved: {output_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()