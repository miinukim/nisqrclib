from __future__ import annotations
from .channel_map import ChannelMapReservoir, ChannelMapReservoirConfig
from .tasks.stm import STMConfig, STMTaskRunner
from .baselines.esn import ESNConfig, run_stm_esn

def stm_demo() -> None:
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
    res = ChannelMapReservoir(qcfg)

    stm_cfg = STMConfig(
        T_total=400, washout=50, train_len=200, test_len=100,
        delays=tuple(range(1, 11)), ridge_l2=1e-6, input_seed=2026, metric="r2"
    )
    runner = STMTaskRunner(res, stm_cfg)
    qres = runner.run()
    mc_q = runner.memory_capacity(qres, use_test=True)

    # stable ESN defaults
    esn_cfg = ESNConfig(n_res=200, spectral_radius=0.8, input_scale=0.1, leak_rate=0.3, ridge_l2=1e-4, seed=2)
    eres = run_stm_esn(
        T_total=stm_cfg.T_total, washout=stm_cfg.washout,
        train_len=stm_cfg.train_len, test_len=stm_cfg.test_len,
        delays=stm_cfg.delays, input_dist=stm_cfg.input_dist,
        input_seed=stm_cfg.input_seed,
        esn_cfg=esn_cfg,
        metric=stm_cfg.metric,
    )
    mc_e = runner.memory_capacity(eres, use_test=True)

    print("Quantum STM memory capacity (sum R^2 over delays):", mc_q)
    print("ESN STM memory capacity (sum R^2 over delays):", mc_e)
    print("Per-delay test R^2 (quantum):", {d: round(v["test_score"], 3) for d, v in qres.items()})
    print("Per-delay test R^2 (ESN):    ", {d: round(v["test_score"], 3) for d, v in eres.items()})
