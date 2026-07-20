import numpy as np
from scipy.optimize import curve_fit
import torch
import torch.nn as nn

DEFAULT_CONFIG = {
    'input_dim': 8, 'hidden_dim': 64, 'num_layers': 2, 'tau': 5,
    'batch_size': 32, 'learning_rate': 0.001, 'betas': (0.9, 0.999),
    'eps': 1e-8, 'weight_decay': 1e-5, 'max_epochs': 200,
    'early_stop_patience': 20, 'lr_scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 5, 'scheduler_factor': 0.1,
    'lambda_traj': 0.0, 'grad_clip_norm': 1.0,
}


class MLP_LP:
    def __init__(self, X0_guess=1.5):
        self.X0_guess = X0_guess
        self.logistic_params = {}
        self.Y_XS = None
        self.m_S = None
        self._fitted = False

    @staticmethod
    def _logistic(t, X_max, X_0, mu_max):
        return X_max / (1.0 + ((X_max - X_0) / X_0) * np.exp(-mu_max * t))

    @staticmethod
    def _logistic_derivative(t, X_max, X_0, mu_max):
        X = MLP_LP._logistic(t, X_max, X_0, mu_max)
        return mu_max * X * (1.0 - X / X_max)

    def fit_logistic_batch(self, t_dcw, dcw):
        p0 = [dcw.max() * 1.2, self.X0_guess, 0.05]
        bounds = ([dcw.max() * 0.5, 0.1, 0.001],
                  [dcw.max() * 5.0, 10.0, 0.5])
        try:
            popt, _ = curve_fit(self._logistic, t_dcw, dcw, p0=p0,
                                bounds=bounds, maxfev=10000)
        except Exception:
            popt = p0
        return popt[0], popt[1], popt[2]

    def fit(self, DCW_dict, train_batch_ids, y_raw_list, t_seq=None):
        for bid in train_batch_ids:
            dcw_data = DCW_dict.get(bid)
            if dcw_data is None or dcw_data.shape[0] < 4:
                continue
            t_dcw = dcw_data[:, 0]
            dcw = dcw_data[:, 1]
            self.logistic_params[bid] = self.fit_logistic_batch(t_dcw, dcw)

        all_dX, all_X, all_rS = [], [], []
        if t_seq is None:
            t_seq = np.arange(1, 121, dtype=np.float64)

        for bid in train_batch_ids:
            if bid not in self.logistic_params:
                continue
            X_max, X_0, mu_max = self.logistic_params[bid]
            X_fit = self._logistic(t_seq, X_max, X_0, mu_max)
            dX_fit = self._logistic_derivative(t_seq, X_max, X_0, mu_max)
            rS_batch = y_raw_list[bid - 1]
            all_dX.append(dX_fit)
            all_X.append(X_fit)
            all_rS.append(rS_batch)

        dX_all = np.concatenate(all_dX)
        X_all = np.concatenate(all_X)
        rS_all = np.concatenate(all_rS)

        A = np.column_stack([dX_all, X_all])
        coeff, _, _, _ = np.linalg.lstsq(A, rS_all, rcond=None)
        a, b = coeff[0], coeff[1]
        self.Y_XS = 1.0 / a if abs(a) > 1e-8 else np.inf
        self.m_S = b
        self._fitted = True

    def predict(self, DCW_dict, batch_id, t_seq=None):
        if not self._fitted:
            raise RuntimeError('M-LP not fitted. Call fit() first.')
        if t_seq is None:
            t_seq = np.arange(1, 121, dtype=np.float64)
        dcw_data = DCW_dict.get(batch_id)
        if dcw_data is None or dcw_data.shape[0] < 4:
            return np.full_like(t_seq, np.nan, dtype=np.float64)
        t_dcw = dcw_data[:, 0]
        dcw = dcw_data[:, 1]
        X_max, X_0, mu_max = self.fit_logistic_batch(t_dcw, dcw)
        X_fit = self._logistic(t_seq, X_max, X_0, mu_max)
        dX_fit = self._logistic_derivative(t_seq, X_max, X_0, mu_max)
        r_S_pred = (1.0 / self.Y_XS) * dX_fit + self.m_S * X_fit
        return np.maximum(r_S_pred, 0.0)


class SimpleRNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers,
                          batch_first=True, nonlinearity='tanh')
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.regressor(rnn_out)


class BiLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.regressor(lstm_out)


class CNNLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2,
                 conv_out=32, kernel_size=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_out, kernel_size,
                              padding=kernel_size - 1)
        self.lstm = nn.LSTM(conv_out, hidden_dim, num_layers,
                            batch_first=True)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv[:, :, :x.shape[1]]
        x_conv = torch.relu(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_conv)
        return self.regressor(lstm_out)


class ATLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2,
                 attn_dropout=0.1, attn_window=5):
        super().__init__()
        self.attn_window = attn_window
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        T = lstm_out.shape[1]
        H = lstm_out.shape[2]

        lstm_normed = self.norm1(lstm_out)
        Q = self.query_proj(lstm_normed)
        K = self.key_proj(lstm_normed)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (H ** 0.5)

        rows = torch.arange(T, device=x.device).unsqueeze(1)
        cols = torch.arange(T, device=x.device).unsqueeze(0)
        w = self.attn_window
        local_mask = (cols <= rows) & (cols >= rows - w + 1)
        scores = scores.masked_fill(~local_mask.unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.bmm(attn_weights, lstm_out)

        context = self.norm2(attn_out + lstm_out)
        return self.regressor(context)


TCATLSTM = ATLSTM
SCATLSTM = ATLSTM


def stoichiometric_loss(r_S_hat, OUR_batch, Y_OS):
    return torch.mean((r_S_hat - Y_OS * OUR_batch) ** 2)


def compute_trajectory(r_S_hat_seq, D_seq, Glu_seq, S_init, C_feed=800.0, start_idx=0):
    r_S_hat_seq = np.asarray(r_S_hat_seq).ravel()
    D_seq = np.asarray(D_seq).ravel()
    Glu_seq = np.asarray(Glu_seq).ravel()

    seq_len = len(r_S_hat_seq)
    S_hat = np.full(seq_len, np.nan)

    if start_idx > 0:
        S_hat[start_idx] = Glu_seq[start_idx]
        S_curr = Glu_seq[start_idx]
        for t in range(start_idx + 1, seq_len):
            S_next = S_curr + D_seq[t] * (C_feed - S_curr) - r_S_hat_seq[t]
            S_hat[t] = S_next
            S_curr = max(S_next, 0.0)
        valid_mask = ~np.isnan(S_hat)
        MSE_traj = np.mean((Glu_seq[valid_mask] - S_hat[valid_mask]) ** 2)
    else:
        S_curr = S_init
        for t in range(seq_len):
            S_next = S_curr + D_seq[t] * (C_feed - S_curr) - r_S_hat_seq[t]
            S_hat[t] = S_next
            S_curr = max(S_next, 0.0)
        MSE_traj = np.mean((Glu_seq - S_hat) ** 2)

    return S_hat, MSE_traj


_DL_MODEL_REGISTRY = {
    'M-RNN': (SimpleRNN, True), 'M-LSTM': (BiLSTM, True),
    'M-CNNL': (CNNLSTM, True), 'M-ATL': (ATLSTM, True),
    'M-TCAL': (TCATLSTM, True), 'M-SCAL': (SCATLSTM, True),
}

_ML_MODEL_NAMES = {'M-RF', 'M-XGB', 'M-LGB', 'M-SVR'}
_MECH_MODEL_NAMES = {'M-LP'}

ALL_MODEL_NAMES = list(_DL_MODEL_REGISTRY.keys()) + list(_ML_MODEL_NAMES) + list(_MECH_MODEL_NAMES)


def get_dl_model(model_name, input_dim=8, hidden_dim=64, num_layers=2, **kwargs):
    if model_name not in _DL_MODEL_REGISTRY:
        raise ValueError(f'Unknown DL model: {model_name}. Options: {list(_DL_MODEL_REGISTRY.keys())}')
    model_cls, _ = _DL_MODEL_REGISTRY[model_name]
    return model_cls(input_dim=input_dim, hidden_dim=hidden_dim,
                     num_layers=num_layers, **kwargs)


def is_sequential_model(model_name):
    return model_name in _DL_MODEL_REGISTRY


def is_sklearn_model(model_name):
    return model_name in _ML_MODEL_NAMES


def is_mechanistic_model(model_name):
    return model_name in _MECH_MODEL_NAMES