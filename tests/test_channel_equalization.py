from __future__ import annotations

import numpy as np

from nisqrclib.baselines.esn import ESNConfig, run_channel_equalization_esn
from nisqrclib.channel_map import ChannelMapReservoir, ChannelMapReservoirConfig
from nisqrclib.tasks.channel_equalization import (
    ChannelEqualizationConfig,
    ChannelEqualizationTaskRunner,
    generate_channel_equalization_data,
)


def test_channel_equalization_quantum_and_esn_run() -> None:
    cfg = ChannelEqualizationConfig(
        T_total=600,
        washout=50,
        train_len=350,
        test_len=150,
        delay=2,
        input_seed=11,
        ridge_l2=1e-6,
        metric="ber",
    )
    qres = ChannelMapReservoir(
        ChannelMapReservoirConfig(
            n_system=4,
            n_ancilla=2,
            tau=1.0,
            seed=17462,
            include_bias=True,
            use_shot_noise=False,
        )
    )
    qout = ChannelEqualizationTaskRunner(qres, cfg).run()

    observed, target = generate_channel_equalization_data(cfg)
    eout = run_channel_equalization_esn(
        observed=observed,
        target=target,
        washout=max(cfg.washout, cfg.delay, len(cfg.taps) - 1),
        train_len=cfg.train_len,
        test_len=cfg.test_len,
        esn_cfg=ESNConfig(n_res=100, spectral_radius=0.8, input_scale=0.1, leak_rate=0.3, seed=2),
        ridge_l2=cfg.ridge_l2,
        metric=cfg.metric,
    )

    for out in (qout, eout):
        assert np.isfinite(out["test_ber"])
        assert np.isfinite(out["test_mse"])
        assert 0.0 <= out["test_ber"] <= 1.0
