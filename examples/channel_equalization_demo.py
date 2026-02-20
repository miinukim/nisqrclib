from __future__ import annotations

import numpy as np

from nisqrclib.baselines.esn import ESNConfig, run_channel_equalization_esn
from nisqrclib.channel_map import ChannelMapReservoir, ChannelMapReservoirConfig
from nisqrclib.tasks.channel_equalization import (
    ChannelEqualizationConfig,
    ChannelEqualizationTaskRunner,
    generate_channel_equalization_data,
)

np.seterr(all="raise")


def main() -> None:
    qcfg = ChannelMapReservoirConfig(
        n_system=4,
        n_ancilla=2,
        tau=1.0,
        input_scale=1.0,
        include_bias=True,
        use_shot_noise=False,
        init_state="maximally_mixed",
        seed=17462,
    )
    qres = ChannelMapReservoir(qcfg)

    ce_cfg = ChannelEqualizationConfig(
        T_total=3000,
        washout=200,
        train_len=1800,
        test_len=800,
        delay=2,
        input_seed=2026,
        ridge_l2=1e-6,
        metric="ber",
    )

    qrunner = ChannelEqualizationTaskRunner(qres, ce_cfg)
    qout = qrunner.run()

    observed, target = generate_channel_equalization_data(ce_cfg)
    esn_cfg = ESNConfig(
        n_res=200,
        spectral_radius=0.8,
        input_scale=0.1,
        leak_rate=0.3,
        ridge_l2=1e-4,
        seed=2,
    )
    eout = run_channel_equalization_esn(
        observed=observed,
        target=target,
        washout=max(ce_cfg.washout, ce_cfg.delay, len(ce_cfg.taps) - 1),
        train_len=ce_cfg.train_len,
        test_len=ce_cfg.test_len,
        esn_cfg=esn_cfg,
        ridge_l2=ce_cfg.ridge_l2,
        metric=ce_cfg.metric,
    )

    print("Channel Equalization (test)")
    print(f"Quantum BER: {qout['test_ber']:.6f} | MSE: {qout['test_mse']:.6f}")
    print(f"ESN BER:     {eout['test_ber']:.6f} | MSE: {eout['test_mse']:.6f}")
    print(f"Quantum score: {qout['test_score']:.6f}")
    print(f"ESN score:     {eout['test_score']:.6f}")


if __name__ == "__main__":
    main()
