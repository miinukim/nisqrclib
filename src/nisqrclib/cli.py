from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .baselines import ESNConfig, run_channel_equalization_esn, run_stm_esn
from .channel_map import ChannelMapReservoir, ChannelMapReservoirConfig
from .hardware import HardwareTrajectoryReservoir, HardwareTrajectoryReservoirConfig
from .tasks import (
    ChannelEqualizationConfig,
    ChannelEqualizationTaskRunner,
    STMConfig,
    STMTaskRunner,
    generate_channel_equalization_data,
)


ReservoirFactory = Callable[[], Any]


def _make_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def _output_dir_str() -> str:
    if HydraConfig.initialized():
        return str(_make_output_dir())
    return "N/A"


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_builtin(payload), fh, indent=2, sort_keys=True)


def _save_arrays(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_run_artifacts(
    cfg: DictConfig,
    metrics: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    summary: Dict[str, Any],
) -> None:
    outdir = _make_output_dir()
    if bool(cfg.logging.save_resolved_config):
        OmegaConf.save(cfg, outdir / "resolved_config.yaml", resolve=True)
    if bool(cfg.logging.save_metrics):
        _save_json(outdir / "metrics.json", metrics)
    if bool(cfg.logging.save_summary):
        _save_json(outdir / "run_summary.json", summary)
    if bool(cfg.logging.save_arrays) and arrays:
        _save_arrays(outdir / "arrays" / "results.npz", arrays)


def _print_metrics(metrics: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def _build_reservoir_factory(cfg: DictConfig) -> ReservoirFactory:
    kind = str(cfg.reservoir.kind)

    if kind == "channel_map":
        def factory() -> ChannelMapReservoir:
            rcfg = ChannelMapReservoirConfig(
                n_system=int(cfg.reservoir.n_system),
                n_ancilla=int(cfg.reservoir.n_ancilla),
                tau=float(cfg.reservoir.tau),
                input_scale=float(cfg.reservoir.input_scale),
                include_bias=bool(cfg.reservoir.include_bias),
                use_shot_noise=bool(cfg.reservoir.use_shot_noise),
                shots=int(cfg.reservoir.shots),
                init_state=str(cfg.reservoir.init_state),
                seed=int(cfg.reservoir.seed),
            )
            return ChannelMapReservoir(rcfg)

        return factory

    if kind == "hardware_trajectory":
        def factory() -> HardwareTrajectoryReservoir:
            rcfg = HardwareTrajectoryReservoirConfig(
                n_system=int(cfg.reservoir.n_system),
                n_ancilla=int(cfg.reservoir.n_ancilla),
                tau=float(cfg.reservoir.tau),
                input_scale=float(cfg.reservoir.input_scale),
                include_bias=bool(cfg.reservoir.include_bias),
                init_state=str(cfg.reservoir.init_state),
                shots=int(cfg.reservoir.shots),
                seed=int(cfg.reservoir.seed),
            )
            return HardwareTrajectoryReservoir(rcfg)

        return factory

    raise ValueError(f"Unsupported reservoir kind: {kind}")


def _esn_config(cfg: DictConfig) -> ESNConfig:
    return ESNConfig(
        n_res=int(cfg.baseline.n_res),
        spectral_radius=float(cfg.baseline.spectral_radius),
        input_scale=float(cfg.baseline.input_scale),
        leak_rate=float(cfg.baseline.leak_rate),
        ridge_l2=float(cfg.baseline.ridge_l2),
        seed=int(cfg.baseline.seed),
        state_clip=float(cfg.baseline.state_clip),
        power_iter=int(cfg.baseline.power_iter),
    )


def _run_stm(cfg: DictConfig, reservoir_factory: ReservoirFactory) -> tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    task_cfg = STMConfig(
        T_total=int(cfg.task.T_total),
        washout=int(cfg.task.washout),
        train_len=int(cfg.task.train_len),
        test_len=int(cfg.task.test_len),
        delays=tuple(int(d) for d in cfg.task.delays),
        input_dist=str(cfg.task.input_dist),
        input_seed=int(cfg.task.input_seed),
        ridge_l2=float(cfg.task.ridge_l2),
        metric=str(cfg.task.metric),
    )

    analysis_runner = STMTaskRunner(reservoir_factory(), task_cfg)
    inputs = analysis_runner.generate_inputs()
    features = reservoir_factory().run_stream(inputs.tolist())

    quantum_runner = STMTaskRunner(reservoir_factory(), task_cfg)
    qres = quantum_runner.run()
    quantum_mc = quantum_runner.memory_capacity(qres, use_test=True)

    esn_cfg = _esn_config(cfg)
    eres = run_stm_esn(
        T_total=task_cfg.T_total,
        washout=task_cfg.washout,
        train_len=task_cfg.train_len,
        test_len=task_cfg.test_len,
        delays=task_cfg.delays,
        input_dist=task_cfg.input_dist,
        input_seed=task_cfg.input_seed,
        esn_cfg=esn_cfg,
        metric=task_cfg.metric,
    )
    esn_mc = quantum_runner.memory_capacity(eres, use_test=True)

    delays = np.asarray(task_cfg.delays, dtype=int)
    q_train = np.asarray([qres[d]["train_score"] for d in task_cfg.delays], dtype=float)
    q_test = np.asarray([qres[d]["test_score"] for d in task_cfg.delays], dtype=float)
    e_train = np.asarray([eres[d]["train_score"] for d in task_cfg.delays], dtype=float)
    e_test = np.asarray([eres[d]["test_score"] for d in task_cfg.delays], dtype=float)

    metrics = {
        "task": "stm",
        "reservoir_kind": str(cfg.reservoir.kind),
        "quantum_memory_capacity": quantum_mc,
        "esn_memory_capacity": esn_mc,
        "quantum_best_test_score": float(np.max(q_test)),
        "esn_best_test_score": float(np.max(e_test)),
    }
    arrays = {
        "inputs": inputs.astype(float),
        "reservoir_features": np.asarray(features, dtype=float),
        "delays": delays,
        "quantum_train_scores": q_train,
        "quantum_test_scores": q_test,
        "esn_train_scores": e_train,
        "esn_test_scores": e_test,
    }
    summary = {
        "task": "stm",
        "metric": task_cfg.metric,
        "n_delays": len(task_cfg.delays),
        "output_dir": _output_dir_str(),
    }
    return metrics, arrays, summary


def _run_channel_equalization(
    cfg: DictConfig,
    reservoir_factory: ReservoirFactory,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    task_cfg = ChannelEqualizationConfig(
        T_total=int(cfg.task.T_total),
        washout=int(cfg.task.washout),
        train_len=int(cfg.task.train_len),
        test_len=int(cfg.task.test_len),
        delay=int(cfg.task.delay),
        input_seed=int(cfg.task.input_seed),
        ridge_l2=float(cfg.task.ridge_l2),
        taps=tuple(float(v) for v in cfg.task.taps),
        nonlin2=float(cfg.task.nonlin2),
        nonlin3=float(cfg.task.nonlin3),
        noise_std=float(cfg.task.noise_std),
        metric=str(cfg.task.metric),
    )

    observed, target = generate_channel_equalization_data(task_cfg)
    features = reservoir_factory().run_stream(observed.tolist())

    quantum_runner = ChannelEqualizationTaskRunner(reservoir_factory(), task_cfg)
    qres = quantum_runner.run()

    esn_cfg = _esn_config(cfg)
    eres = run_channel_equalization_esn(
        observed=observed,
        target=target,
        washout=task_cfg.washout,
        train_len=task_cfg.train_len,
        test_len=task_cfg.test_len,
        esn_cfg=esn_cfg,
        ridge_l2=task_cfg.ridge_l2,
        metric=task_cfg.metric,
    )

    metrics = {
        "task": "channel_equalization",
        "reservoir_kind": str(cfg.reservoir.kind),
        "quantum_test_score": float(qres["test_score"]),
        "quantum_test_ber": float(qres["test_ber"]),
        "quantum_test_mse": float(qres["test_mse"]),
        "esn_test_score": float(eres["test_score"]),
        "esn_test_ber": float(eres["test_ber"]),
        "esn_test_mse": float(eres["test_mse"]),
    }
    arrays = {
        "observed": observed.astype(float),
        "target": target.astype(float),
        "reservoir_features": np.asarray(features, dtype=float),
    }
    summary = {
        "task": "channel_equalization",
        "metric": task_cfg.metric,
        "delay": task_cfg.delay,
        "output_dir": _output_dir_str(),
    }
    return metrics, arrays, summary


def run_experiment_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    reservoir_factory = _build_reservoir_factory(cfg)
    task_name = str(cfg.task.name)

    if task_name == "stm":
        metrics, arrays, summary = _run_stm(cfg, reservoir_factory)
    elif task_name == "channel_equalization":
        metrics, arrays, summary = _run_channel_equalization(cfg, reservoir_factory)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    _save_run_artifacts(cfg, metrics=metrics, arrays=arrays, summary=summary)
    _print_metrics(metrics)
    print(f"output_dir: {_make_output_dir()}")
    return metrics


def stm_demo() -> None:
    cfg = OmegaConf.create(
        {
            "reservoir": {
                "kind": "channel_map",
                "n_system": 4,
                "n_ancilla": 2,
                "tau": 1.0,
                "input_scale": 1.0,
                "include_bias": True,
                "use_shot_noise": False,
                "shots": 4096,
                "init_state": "maximally_mixed",
                "seed": 17462,
            },
            "task": {
                "name": "stm",
                "T_total": 400,
                "washout": 50,
                "train_len": 200,
                "test_len": 100,
                "delays": list(range(1, 11)),
                "input_dist": "uniform_pm1",
                "input_seed": 2026,
                "ridge_l2": 1e-6,
                "metric": "r2",
            },
            "baseline": {
                "n_res": 200,
                "spectral_radius": 0.8,
                "input_scale": 0.1,
                "leak_rate": 0.3,
                "ridge_l2": 1e-4,
                "seed": 2,
                "state_clip": 5.0,
                "power_iter": 200,
            },
        }
    )
    metrics, _, _ = _run_stm(cfg, _build_reservoir_factory(cfg))
    _print_metrics(metrics)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    run_experiment_from_cfg(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def stm_hydra(cfg: DictConfig) -> None:
    run_experiment_from_cfg(cfg)
