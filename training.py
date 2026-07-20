import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings

from models import (
    get_dl_model, is_sequential_model, is_sklearn_model, is_mechanistic_model,
    MLP_LP, DEFAULT_CONFIG
)

warnings.filterwarnings('ignore', category=UserWarning)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 2:
        return {'MAE': np.nan, 'MSE': np.nan, 'R2': np.nan}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    return {
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'MSE': np.mean((y_true - y_pred) ** 2),
        'R2': 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan,
    }


def _split_folds(batch_ids, n_folds=5, random_state=42):
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(batch_ids)
    fold_groups = np.array_split(shuffled, n_folds)
    folds = []
    for i, val_bids in enumerate(fold_groups):
        train_bids = [int(b) for j, group in enumerate(fold_groups) if j != i
                      for b in group]
        folds.append((train_bids, [int(b) for b in val_bids]))
    return folds


def _split_folds_stratified(random_state=42):
    pulse = [15, 16, 18, 19, 20]
    continuous = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]

    fold_val_config = [
        (16, [1, 6, 9]), (20, [2, 7, 10]), (18, [3, 8, 11]),
        (15, [4, 12, 14]), (19, [5, 13, 17]),
    ]

    folds = []
    for pulse_bid, cont_bids in fold_val_config:
        val_bids = [pulse_bid] + cont_bids
        train_bids = [b for b in (pulse + continuous) if b not in val_bids]
        folds.append((train_bids, val_bids))
    return folds


def fit_Y_OS(y_train_std, OUR_train_std):
    y = y_train_std.ravel()
    OUR = OUR_train_std.ravel()
    Y_OS = np.dot(OUR, y) / (np.dot(OUR, OUR) + 1e-12)
    return float(Y_OS)


def trajectory_loss(r_S_hat_raw, D_raw, Glu_raw, C_feed=800.0, start_idx=20):
    B, T = r_S_hat_raw.shape
    S_hat_list = []
    S_curr = Glu_raw[:, start_idx].clone()

    for t in range(start_idx, T):
        S_next = S_curr + D_raw[:, t] * (C_feed - S_curr) - r_S_hat_raw[:, t]
        S_hat_list.append(S_next)
        S_curr = S_next

    S_hat = torch.stack(S_hat_list, dim=1)
    return torch.mean((Glu_raw[:, start_idx:] - S_hat) ** 2)


def _prepare_sequences(X_std_list, y_std_list, batch_ids):
    seqs = []
    for bid in batch_ids:
        X_i = X_std_list[bid - 1]
        y_i = y_std_list[bid - 1]
        OUR_i = X_i[:, 6].copy()
        seqs.append((X_i, y_i, OUR_i))
    return seqs


@torch.no_grad()
def _predict_dl_sequential(model, X_std_list, batch_ids):
    model.eval()
    preds = []
    for bid in batch_ids:
        X_i = X_std_list[bid - 1]
        X_t = torch.tensor(X_i, dtype=torch.float32).unsqueeze(0)
        pred = model(X_t).squeeze(0).cpu().numpy()
        preds.append(pred)
    return preds


def train_dl_sequential(model, train_seqs, X_std_list, y_std_list, val_bids,
                         lambda_traj=0.0, D_raw_list=None, Glu_raw_list=None,
                         scaler_y=None, train_bids=None, config=None):
    if config is None:
        config = DEFAULT_CONFIG

    n_train = len(train_seqs)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['learning_rate'],
                                  betas=config['betas'],
                                  eps=config['eps'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler_factor'],
        patience=config['scheduler_patience'])

    best_val_mse = float('inf')
    best_state = None
    patience_counter = 0

    use_traj = (lambda_traj > 0 and D_raw_list is not None
                and Glu_raw_list is not None and scaler_y is not None
                and train_bids is not None)
    if use_traj:
        scale_y = torch.tensor(scaler_y.scale_, dtype=torch.float32)
        mean_y = torch.tensor(scaler_y.mean_, dtype=torch.float32)

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0

        batch_order = np.random.permutation(n_train)
        for idx in batch_order:
            X_i, y_i, OUR_i = train_seqs[idx]
            X_t = torch.tensor(X_i, dtype=torch.float32).unsqueeze(0)
            y_t = torch.tensor(y_i, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            optimizer.zero_grad()
            r_S_hat = model(X_t)
            L_data = torch.mean((y_t - r_S_hat) ** 2)

            if use_traj:
                r_S_hat_raw = r_S_hat.squeeze(0).squeeze(-1) * scale_y + mean_y
                r_S_hat_raw = torch.clamp(r_S_hat_raw, min=0.0)

                bid = train_bids[idx]
                D_i = torch.tensor(D_raw_list[bid - 1], dtype=torch.float32)
                Glu_i = torch.tensor(Glu_raw_list[bid - 1], dtype=torch.float32)

                L_traj = trajectory_loss(
                    r_S_hat_raw.unsqueeze(0), D_i.unsqueeze(0), Glu_i.unsqueeze(0))
                L = L_data + lambda_traj * L_traj
            else:
                L = L_data

            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += L.item()

        val_preds = _predict_dl_sequential(model, X_std_list, val_bids)
        val_pred_all = np.concatenate([p for p in val_preds if len(p) > 0], axis=0)
        val_y_all = np.concatenate([y_std_list[b - 1].reshape(-1, 1) for b in val_bids], axis=0)
        val_mse = np.mean((val_y_all - val_pred_all) ** 2)

        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stop_patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _lobo_cv_mlp(data, fold_splits, batch_ids):
    DCW_dict = data['DCW_dict']
    y_raw_list = data['y_raw_list']
    t_seq = np.arange(1, 121, dtype=np.float64)

    fold_results = []
    all_preds, all_trues = [], []
    all_val_bids = []

    for train_bids, val_bids in fold_splits:
        model = MLP_LP(X0_guess=1.5)
        try:
            model.fit(DCW_dict, train_bids, y_raw_list, t_seq=t_seq)
        except Exception:
            pass

        fold_preds, fold_trues = [], []
        for test_bid in val_bids:
            try:
                r_S_pred_raw = model.predict(DCW_dict, test_bid, t_seq=t_seq)
            except Exception:
                r_S_pred_raw = np.full(120, np.nan)
            r_S_true_raw = y_raw_list[test_bid - 1]
            fold_preds.append(r_S_pred_raw)
            fold_trues.append(r_S_true_raw)

        fold_pred_all = np.concatenate(fold_preds)
        fold_true_all = np.concatenate(fold_trues)
        metrics = compute_metrics(fold_true_all, fold_pred_all)
        fold_results.append(metrics)
        all_preds.append(fold_pred_all)
        all_trues.append(fold_true_all)
        all_val_bids.append(val_bids)

    return fold_results, all_preds, all_trues, all_val_bids


def _create_sklearn_model(model_name):
    if model_name == 'M-RF':
        return RandomForestRegressor(n_estimators=100, max_depth=10,
                                      random_state=42, n_jobs=-1)
    elif model_name == 'M-SVR':
        return SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    elif model_name == 'M-XGB':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, max_depth=6,
                            learning_rate=0.1, random_state=42, verbosity=0)
    elif model_name == 'M-LGB':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, max_depth=6,
                              learning_rate=0.1, random_state=42, verbose=-1)
    else:
        raise ValueError(f'Unknown ML model: {model_name}')


def _lobo_cv_sklearn(model_name, data, fold_splits, batch_ids):
    X_std_list = data['X_std_list']
    y_std_list = data['y_std_list']
    scaler_y = data['scaler_y']

    fold_results = []
    all_preds_raw, all_trues_raw = [], []
    all_val_bids = []

    for train_bids, val_bids in fold_splits:
        train_X = np.concatenate([X_std_list[i - 1] for i in train_bids], axis=0)
        train_y = np.concatenate([y_std_list[i - 1] for i in train_bids], axis=0)

        model = _create_sklearn_model(model_name)
        model.fit(train_X, train_y.ravel())

        fold_preds, fold_trues = [], []
        for test_bid in val_bids:
            test_X = X_std_list[test_bid - 1]
            test_y_std = y_std_list[test_bid - 1]
            pred_std = model.predict(test_X)
            pred_raw = scaler_y.inverse_transform(pred_std.reshape(-1, 1)).ravel()
            true_raw = scaler_y.inverse_transform(test_y_std.reshape(-1, 1)).ravel()
            fold_preds.append(pred_raw)
            fold_trues.append(true_raw)

        fold_pred_all = np.concatenate(fold_preds)
        fold_true_all = np.concatenate(fold_trues)
        metrics = compute_metrics(fold_true_all, fold_pred_all)
        fold_results.append(metrics)
        all_preds_raw.append(fold_pred_all)
        all_trues_raw.append(fold_true_all)
        all_val_bids.append(val_bids)

    return fold_results, all_preds_raw, all_trues_raw, all_val_bids


def _lobo_cv_dl(model_name, data, fold_splits, batch_ids, config=None):
    if config is None:
        config = DEFAULT_CONFIG

    hidden_dim = config.get('hidden_dim', 64)
    num_layers = config.get('num_layers', 2)
    lambda_traj = config.get('lambda_traj', 0.0)

    X_std_list = data['X_std_list']
    y_std_list = data['y_std_list']
    scaler_y = data['scaler_y']
    D_raw_list = data.get('D_list', None)
    Glu_raw_list = data.get('Glu_list', None)

    fold_results = []
    all_preds_raw, all_trues_raw = [], []
    all_val_bids = []

    for fold_idx, (train_bids, val_bids) in enumerate(fold_splits):
        train_seqs = _prepare_sequences(X_std_list, y_std_list, train_bids)

        if model_name in ('M-TCAL', 'M-SCAL') and lambda_traj > 0:
            print(f'    TC-AT-LSTM: lambda_traj = {lambda_traj}')
        elif model_name in ('M-TCAL', 'M-SCAL'):
            print(f'    TC-AT-LSTM: lambda_traj = 0 (ablation baseline)')

        model = get_dl_model(model_name, input_dim=config['input_dim'],
                             hidden_dim=hidden_dim, num_layers=num_layers)

        model = train_dl_sequential(model, train_seqs,
                                     X_std_list, y_std_list, val_bids,
                                     lambda_traj=lambda_traj,
                                     D_raw_list=D_raw_list,
                                     Glu_raw_list=Glu_raw_list,
                                     scaler_y=scaler_y,
                                     train_bids=train_bids,
                                     config=config)

        pred_std = _predict_dl_sequential(model, X_std_list, val_bids)
        pred_std_all = np.concatenate([p for p in pred_std if len(p) > 0], axis=0)
        true_std_list = [y_std_list[b - 1].reshape(-1, 1) for b in val_bids]
        true_std_all = np.concatenate(true_std_list, axis=0)

        pred_raw = scaler_y.inverse_transform(pred_std_all).ravel()
        pred_raw = np.maximum(pred_raw, 0.0)
        true_raw = scaler_y.inverse_transform(true_std_all).ravel()

        metrics = compute_metrics(true_raw, pred_raw)
        fold_results.append(metrics)
        all_preds_raw.append(pred_raw)
        all_trues_raw.append(true_raw)
        all_val_bids.append(val_bids)

        print(f'    [{model_name}] fold {fold_idx + 1}/{len(fold_splits)} '
              f'val_batches={val_bids}  R2={metrics["R2"]:.4f}')

    return fold_results, all_preds_raw, all_trues_raw, all_val_bids


def run_lobo_cv(model_name, data, config=None, n_folds=5, random_state=42,
                 fold_splits=None):
    batch_ids = data['batch_ids']
    if fold_splits is None:
        fold_splits = _split_folds(batch_ids, n_folds=n_folds,
                                   random_state=random_state)

    print(f'\n{"="*60}')
    print(f'  LOBO-CV: {model_name}  ({len(fold_splits)}-fold)')
    print(f'{"="*60}')
    for fi, (tr, va) in enumerate(fold_splits):
        print(f'  Fold {fi + 1}: train={tr}, val={va}')

    if is_mechanistic_model(model_name):
        fold_metrics, preds, truths, val_batches = _lobo_cv_mlp(data, fold_splits, batch_ids)
    elif is_sklearn_model(model_name):
        fold_metrics, preds, truths, val_batches = _lobo_cv_sklearn(model_name, data, fold_splits, batch_ids)
    elif is_sequential_model(model_name):
        fold_metrics, preds, truths, val_batches = _lobo_cv_dl(model_name, data, fold_splits, batch_ids, config)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    mae_vals = [m['MAE'] for m in fold_metrics if not np.isnan(m['MAE'])]
    mse_vals = [m['MSE'] for m in fold_metrics if not np.isnan(m['MSE'])]
    r2_vals = [m['R2'] for m in fold_metrics if not np.isnan(m['R2'])]

    results = {
        'model_name': model_name, 'fold_metrics': fold_metrics,
        'predictions': preds, 'targets': truths,
        'val_batches': val_batches, 'fold_splits': fold_splits,
        'summary_mae': (np.mean(mae_vals), np.std(mae_vals)),
        'summary_mse': (np.mean(mse_vals), np.std(mse_vals)),
        'summary_r2': (np.mean(r2_vals), np.std(r2_vals)),
    }

    print(f'  {n_folds}-fold summary: MAE={results["summary_mae"][0]:.4f}+/-{results["summary_mae"][1]:.4f}, '
          f'MSE={results["summary_mse"][0]:.4f}+/-{results["summary_mse"][1]:.4f}, '
          f'R2={results["summary_r2"][0]:.4f}+/-{results["summary_r2"][1]:.4f}')

    return results