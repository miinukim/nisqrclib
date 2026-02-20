from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Protocol, Sequence, Tuple

import numpy as np

from ..utils.linear import ridge_regression_fit, ridge_regression_predict


class ReservoirProtocol(Protocol):
    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        ...


@dataclass
class ChannelEqualizationConfig:
    T_total: int = 3000
    washout: int = 200
    train_len: int = 1800
    test_len: int = 800
    delay: int = 2
    input_seed: int = 2026
    ridge_l2: float = 1e-6
    # Causal channel taps (current -> older samples)
    taps: Tuple[float, ...] = (0.08, 0.12, -0.18, -0.10)
    # Cubic nonlinearity coefficients
    nonlin2: float = 0.036
    nonlin3: float = -0.011
    noise_std: float = 0.02
    metric: Literal["ber", "mse"] = "ber"


def generate_channel_equalization_data(cfg: ChannelEqualizationConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.input_seed)
    s = rng.choice([-1.0, 1.0], size=cfg.T_total).astype(float)
    maxlag = max(0, len(cfg.taps) - 1)

    v = np.zeros_like(s)
    for i, a in enumerate(cfg.taps):
        if i == 0:
            v += a * s
        else:
            v[i:] += a * s[:-i]

    y = v + cfg.nonlin2 * (v**2) + cfg.nonlin3 * (v**3)
    if cfg.noise_std > 0.0:
        y += rng.normal(0.0, cfg.noise_std, size=cfg.T_total)

    # Delay is clipped to guarantee a valid causal target index.
    d = int(np.clip(cfg.delay, 0, cfg.T_total - 1))
    target = np.zeros_like(s)
    target[d:] = s[:-d] if d > 0 else s
    # Early target positions are undefined due to delay; keep them but washout should discard.
    return y.astype(float), target.astype(float)


class ChannelEqualizationTaskRunner:
    def __init__(self, reservoir: ReservoirProtocol, cfg: ChannelEqualizationConfig):
        self.res = reservoir
        self.cfg = cfg

    def generate_io(self) -> Tuple[np.ndarray, np.ndarray]:
        return generate_channel_equalization_data(self.cfg)

    def run(self) -> Dict[str, float]:
        cfg = self.cfg
        u, target = self.generate_io()
        X = self.res.run_stream(u.tolist())

        t0 = max(cfg.washout, cfg.delay, len(cfg.taps) - 1)
        t_train_end = t0 + cfg.train_len
        t_test_end = t_train_end + cfg.test_len
        tr = np.arange(t0, t_train_end)
        te = np.arange(t_train_end, t_test_end)

        w = ridge_regression_fit(X[tr], target[tr], l2=cfg.ridge_l2)
        yhat_tr = ridge_regression_predict(X[tr], w)
        yhat_te = ridge_regression_predict(X[te], w)

        pred_tr = np.where(yhat_tr >= 0.0, 1.0, -1.0)
        pred_te = np.where(yhat_te >= 0.0, 1.0, -1.0)

        ber_tr = float(np.mean(pred_tr != target[tr]))
        ber_te = float(np.mean(pred_te != target[te]))
        mse_tr = float(np.mean((yhat_tr - target[tr]) ** 2))
        mse_te = float(np.mean((yhat_te - target[te]) ** 2))

        return {
            "train_ber": ber_tr,
            "test_ber": ber_te,
            "train_mse": mse_tr,
            "test_mse": mse_te,
            "train_score": -ber_tr if cfg.metric == "ber" else -mse_tr,
            "test_score": -ber_te if cfg.metric == "ber" else -mse_te,
        }
