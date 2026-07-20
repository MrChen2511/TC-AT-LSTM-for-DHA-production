import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import warnings

FEATURE_NAMES = ['Stirrer', 'Feed_Rate', 'Kla', 'pH', 'DO', 'CER', 'OUR', 'RQ']
DERIVED_CUMSUM_COLS = ['Feed', 'Acid', 'Base']
INITIAL_VOLUME = 3.0
FEED_GLUCOSE_CONC = 800.0

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / 'Original_Data'
PREPROCESSED_PATH = Path(__file__).resolve().parent.parent / 'Mid_Data' / 'preprocessed_data.pkl'


def load_raw_batches(data_dir=None, n_batches=20):
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    else:
        data_dir = Path(data_dir)

    batches = {}
    for bid in range(1, n_batches + 1):
        fname = data_dir / f'Batch_{bid:02d}.csv'
        if not fname.exists():
            print(f'Warning: {fname} not found, skipping')
            continue
        df = pd.read_csv(fname)
        batches[bid] = df
    return batches


def compute_derived_vars(df):
    df = df.copy()

    df['Feed_Rate'] = np.maximum(0, np.gradient(df['Feed'].values))
    df['Acid_Rate'] = np.maximum(0, np.gradient(df['Acid'].values))
    df['Base_Rate'] = np.maximum(0, np.gradient(df['Base'].values))

    df['Volume'] = (INITIAL_VOLUME
                    + (df['Feed'] + df['Acid'] + df['Base']) / 1000.0)

    df['D'] = (df['Feed_Rate'] / 1000.0) / df['Volume']

    dGlu_dt = np.gradient(df['Glu'].values)
    df['r_S'] = np.maximum(0, df['D'] * (FEED_GLUCOSE_CONC - df['Glu']) - dGlu_dt)

    return df


def extract_features_and_labels(batches_dict):
    X_list, y_list = [], []
    D_list, Glu_list = [], []
    r_S_raw_list, OUR_raw_list = [], []
    S_init_list = []
    DCW_dict = {}
    batch_ids = []

    for bid in sorted(batches_dict.keys()):
        df = batches_dict[bid]
        X = df[FEATURE_NAMES].values.astype(np.float64)
        y = df['r_S'].values.astype(np.float64)
        D = df['D'].values.astype(np.float64)
        Glu = df['Glu'].values.astype(np.float64)
        r_S_raw = y.copy()
        OUR_raw = df['OUR'].values.astype(np.float64)
        S_init = float(df['Glu'].iloc[0])

        X_list.append(X)
        y_list.append(y)
        D_list.append(D)
        Glu_list.append(Glu)
        r_S_raw_list.append(r_S_raw)
        OUR_raw_list.append(OUR_raw)
        S_init_list.append(S_init)
        batch_ids.append(bid)

        dcw_mask = df['DCW'].notna().values
        if dcw_mask.any():
            DCW_dict[bid] = df.loc[dcw_mask, ['Time', 'DCW', 'TL', 'DHA']].values.astype(np.float64)
        else:
            DCW_dict[bid] = np.empty((0, 4))

    aux = {
        'D_list': D_list, 'Glu_list': Glu_list,
        'r_S_raw_list': r_S_raw_list, 'OUR_raw_list': OUR_raw_list,
        'S_init_list': S_init_list, 'DCW_dict': DCW_dict,
        'batch_ids': batch_ids,
    }
    return X_list, y_list, aux


def zscore_normalize(X_list, y_list):
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_std_all = scaler_X.fit_transform(X_all)
    y_std_all = scaler_y.fit_transform(y_all.reshape(-1, 1)).ravel()

    X_std_list, y_std_list = [], []
    idx = 0
    for X_i in X_list:
        T_i = X_i.shape[0]
        X_std_list.append(X_std_all[idx:idx + T_i])
        y_std_list.append(y_std_all[idx:idx + T_i])
        idx += T_i

    return X_std_list, y_std_list, scaler_X, scaler_y


def build_sliding_windows(X_std_list, y_std_list, OUR_std_list, tau=5):
    X_windows, y_windows, OUR_windows = [], [], []
    for X_i, y_i, our_i in zip(X_std_list, y_std_list, OUR_std_list):
        T_i = X_i.shape[0]
        n_win = T_i - tau + 1
        if n_win <= 0:
            continue
        wins_X = np.stack([X_i[t:t + tau] for t in range(n_win)], axis=0)
        wins_y = y_i[tau - 1:T_i].reshape(-1, 1)
        wins_OUR = our_i[tau - 1:T_i].reshape(-1, 1)
        X_windows.append(wins_X)
        y_windows.append(wins_y)
        OUR_windows.append(wins_OUR)
    return X_windows, y_windows, OUR_windows


def rS_to_feedrate(r_S, V, C_feed=FEED_GLUCOSE_CONC):
    return r_S * V * 1000.0 / C_feed


def feedrate_to_rS_supply(Feed_Rate, V, C_feed=FEED_GLUCOSE_CONC):
    return Feed_Rate * C_feed / (1000.0 * V)


def save_preprocessed(data_dict, path=None):
    if path is None:
        path = PREPROCESSED_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Saved: {path}')


def load_preprocessed(path=None):
    if path is None:
        path = PREPROCESSED_PATH
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_preprocessing(data_dir=None, tau_list=(3, 5, 10)):
    print('Data preprocessing pipeline')
    print('=' * 60)

    print('[1/6] Loading raw CSV files ...')
    batches_raw = load_raw_batches(data_dir)
    print(f'      Loaded {len(batches_raw)} batches')

    print('[2/6] Computing derived variables ...')
    batches_derived = {bid: compute_derived_vars(df) for bid, df in batches_raw.items()}

    print('[3/6] Extracting feature matrices and labels ...')
    X_list, y_list, aux = extract_features_and_labels(batches_derived)

    print('[4/6] Z-score standardization (global fit) ...')
    X_std_list, y_std_list, scaler_X, scaler_y = zscore_normalize(X_list, y_list)

    OUR_std_list = [X_i[:, 6] for X_i in X_std_list]

    print(f'[5/6] Building sliding windows tau={list(tau_list)} ...')
    windows_dict = {}
    for tau in tau_list:
        X_w, y_w, OUR_w = build_sliding_windows(X_std_list, y_std_list, OUR_std_list, tau=tau)
        windows_dict[tau] = {'X': X_w, 'y': y_w, 'OUR': OUR_w}
        total_win = sum(w.shape[0] for w in X_w)
        print(f'      tau={tau}: {total_win} windows (20 batches)')

    print('[6/6] Saving preprocessed_data.pkl ...')
    data_dict = {
        'X_raw_list': X_list, 'y_raw_list': y_list,
        'X_std_list': X_std_list, 'y_std_list': y_std_list,
        'windows_dict': windows_dict,
        'D_list': aux['D_list'], 'Glu_list': aux['Glu_list'],
        'OUR_raw_list': aux['OUR_raw_list'], 'r_S_raw_list': aux['r_S_raw_list'],
        'S_init_list': aux['S_init_list'], 'DCW_dict': aux['DCW_dict'],
        'batch_ids': aux['batch_ids'],
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'feature_names': FEATURE_NAMES, 'tau_list': list(tau_list),
    }
    save_preprocessed(data_dict)

    print('Preprocessing complete.')
    print(f'  Batches: {len(aux["batch_ids"])}')
    print(f'  Features: {len(FEATURE_NAMES)}')
    print(f'  Total samples: {sum(len(y) for y in y_list)}')
    print(f'  Output: {PREPROCESSED_PATH}')
    return data_dict


if __name__ == '__main__':
    run_preprocessing()