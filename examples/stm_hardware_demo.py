from __future__ import annotations

import numpy as np

from nisqrclib.baselines.esn import ESNConfig, run_stm_esn
from nisqrclib.hardware import HardwareTrajectoryReservoir, HardwareTrajectoryReservoirConfig
from nisqrclib.tasks.stm import STMConfig, STMTaskRunner

np.seterr(all="raise")


def main() -> None:
    qcfg = HardwareTrajectoryReservoirConfig(
        n_system=4,
        n_ancilla=2,
        tau=1.0,
        input_scale=1.0,
        include_bias=True,
        init_state="maximally_mixed",
        shots=512,
        seed=17462,
    )
    res = HardwareTrajectoryReservoir(qcfg)

    stm_cfg = STMConfig(
        T_total=400,
        washout=50,
        train_len=200,
        test_len=100,
        delays=tuple(range(1, 11)),
        ridge_l2=1e-6,
        input_seed=2026,
        metric="r2",
    )
    runner = STMTaskRunner(res, stm_cfg)
    qres = runner.run()
    mc_q = runner.memory_capacity(qres, use_test=True)

    esn_cfg = ESNConfig(n_res=200, spectral_radius=0.8, input_scale=0.1, leak_rate=0.3, ridge_l2=1e-4, seed=2)
    eres = run_stm_esn(
        T_total=stm_cfg.T_total,
        washout=stm_cfg.washout,
        train_len=stm_cfg.train_len,
        test_len=stm_cfg.test_len,
        delays=stm_cfg.delays,
        input_dist=stm_cfg.input_dist,
        input_seed=stm_cfg.input_seed,
        esn_cfg=esn_cfg,
        metric=stm_cfg.metric,
    )
    mc_e = runner.memory_capacity(eres, use_test=True)

    print("Hardware-trajectory Quantum MC:", mc_q)
    print("ESN MC:", mc_e)


if __name__ == "__main__":
    main()
