from __future__ import annotations

import numpy as np

from nisqrclib.baselines import SoftmaxReadoutConfig, run_channel_equalization_symbol_logistic
from nisqrclib.tasks.channel_equalization import (
    ChannelEqualizationDatasetConfig,
    generate_channel_equalization_dataset,
)


def test_generate_channel_equalization_dataset_shapes() -> None:
    cfg = ChannelEqualizationDatasetConfig(n_train=4, n_test=3, n_symb=12, snr_db=10.0, input_seed=7)
    out = generate_channel_equalization_dataset(cfg)

    assert out["train_messages"].shape == (4, 12)
    assert out["train_observed"].shape == (4, 12)
    assert out["test_messages"].shape == (3, 12)
    assert out["test_observed"].shape == (3, 12)
    assert np.all(np.isin(out["train_messages"], np.array(cfg.symbols, dtype=float)))


def test_run_channel_equalization_symbol_logistic_returns_valid_error_rate() -> None:
    cfg = ChannelEqualizationDatasetConfig(n_train=6, n_test=5, n_symb=15, snr_db=15.0, input_seed=11)
    dataset = generate_channel_equalization_dataset(cfg)

    out = run_channel_equalization_symbol_logistic(
        train_observed=dataset["train_observed"],
        train_messages=dataset["train_messages"],
        test_observed=dataset["test_observed"],
        test_messages=dataset["test_messages"],
        readout_cfg=SoftmaxReadoutConfig(fit_intercept=True, l2=1e-6, max_iter=100, tol=1e-9),
        n_lags=1,
    )

    assert np.isfinite(out["train_error_rate"])
    assert np.isfinite(out["test_error_rate"])
    assert 0.0 <= out["train_error_rate"] <= 1.0
    assert 0.0 <= out["test_error_rate"] <= 1.0
