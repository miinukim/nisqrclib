from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nisqrclib.baselines import ESNConfig, EchoStateNetwork, SoftmaxReadoutConfig, fit_softmax_readout, predict_softmax_readout
from nisqrclib.channel_map import ChannelMapReservoir, ChannelMapReservoirConfig
from nisqrclib.reservoir_params import ReservoirParams
from nisqrclib.tasks.channel_equalization import (
    ChannelEqualizationDatasetConfig,
    collect_channel_equalization_reservoir_features,
    generate_channel_equalization_dataset,
)

np.seterr(all="raise")


def _nearest_symbol(values: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)[..., None]
    symbols = np.asarray(symbols, dtype=float)
    return symbols[np.argmin(np.abs(values - symbols[None, ...]), axis=-1)]


def _evaluate_qrc(
    reservoir: ChannelMapReservoir,
    dataset: dict[str, np.ndarray],
    readout_cfg: SoftmaxReadoutConfig,
    initial_state: np.ndarray,
) -> dict[str, np.ndarray | float]:
    train_features = collect_channel_equalization_reservoir_features(
        reservoir,
        dataset["train_observed"],
        initial_state=initial_state,
    )
    test_features = collect_channel_equalization_reservoir_features(
        reservoir,
        dataset["test_observed"],
        initial_state=initial_state,
    )
    X_train = train_features.reshape(-1, train_features.shape[-1])
    X_test = test_features.reshape(-1, test_features.shape[-1])
    y_train = dataset["train_messages"].reshape(-1)
    y_test = dataset["test_messages"].reshape(-1)

    model = fit_softmax_readout(X_train, y_train, readout_cfg)
    train_pred = predict_softmax_readout(X_train, model)
    test_pred = predict_softmax_readout(X_test, model)
    return {
        "train_error_rate": float(np.mean(train_pred != y_train)),
        "test_error_rate": float(np.mean(test_pred != y_test)),
        "train_pred": train_pred.reshape(dataset["train_messages"].shape),
        "test_pred": test_pred.reshape(dataset["test_messages"].shape),
    }


def _evaluate_raw_logistic(
    dataset: dict[str, np.ndarray],
    readout_cfg: SoftmaxReadoutConfig,
) -> dict[str, np.ndarray | float]:
    X_train = dataset["train_observed"].reshape(-1, 1)
    X_test = dataset["test_observed"].reshape(-1, 1)
    y_train = dataset["train_messages"].reshape(-1)
    y_test = dataset["test_messages"].reshape(-1)

    model = fit_softmax_readout(X_train, y_train, readout_cfg)
    train_pred = predict_softmax_readout(X_train, model)
    test_pred = predict_softmax_readout(X_test, model)
    return {
        "train_error_rate": float(np.mean(train_pred != y_train)),
        "test_error_rate": float(np.mean(test_pred != y_test)),
        "train_pred": train_pred.reshape(dataset["train_messages"].shape),
        "test_pred": test_pred.reshape(dataset["test_messages"].shape),
    }


def _collect_esn_features(esn_cfg: ESNConfig, observed_messages: np.ndarray) -> np.ndarray:
    observed_messages = np.asarray(observed_messages, dtype=float)
    features = []
    for message in observed_messages:
        esn = EchoStateNetwork(esn_cfg)
        message_features = esn.collect_states(message.tolist())
        features.append(np.asarray(message_features, dtype=float))
    return np.stack(features, axis=0)


def _evaluate_esn(
    dataset: dict[str, np.ndarray],
    esn_cfg: ESNConfig,
    readout_cfg: SoftmaxReadoutConfig,
) -> dict[str, np.ndarray | float]:
    train_features = _collect_esn_features(esn_cfg, dataset["train_observed"])
    test_features = _collect_esn_features(esn_cfg, dataset["test_observed"])
    X_train = train_features.reshape(-1, train_features.shape[-1])
    X_test = test_features.reshape(-1, test_features.shape[-1])
    y_train = dataset["train_messages"].reshape(-1)
    y_test = dataset["test_messages"].reshape(-1)

    model = fit_softmax_readout(X_train, y_train, readout_cfg)
    train_pred = predict_softmax_readout(X_train, model)
    test_pred = predict_softmax_readout(X_test, model)
    return {
        "train_error_rate": float(np.mean(train_pred != y_train)),
        "test_error_rate": float(np.mean(test_pred != y_test)),
        "train_pred": train_pred.reshape(dataset["train_messages"].shape),
        "test_pred": test_pred.reshape(dataset["test_messages"].shape),
    }


def _evaluate_naive_rounding(dataset: dict[str, np.ndarray]) -> dict[str, float]:
    symbols = dataset["symbols"]
    train_pred = _nearest_symbol(dataset["train_observed"], symbols)
    test_pred = _nearest_symbol(dataset["test_observed"], symbols)
    return {
        "train_error_rate": float(np.mean(train_pred != dataset["train_messages"])),
        "test_error_rate": float(np.mean(test_pred != dataset["test_messages"])),
    }


def _plot_results(
    outdir: Path,
    snr_list: np.ndarray,
    qrc_errors: np.ndarray,
    esn_errors: np.ndarray,
    logistic_errors: np.ndarray,
    naive_errors: np.ndarray,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(snr_list, naive_errors, "k-.", linewidth=2.5, label="Naive rounding")
    ax.plot(snr_list, logistic_errors, color="#bcbd22", marker="+", markersize=14, linewidth=2.5, label="Logistic")
    ax.plot(snr_list, esn_errors, color="#1f77b4", marker="s", markersize=8, linewidth=2.4, label="ESN")
    ax.plot(
        snr_list,
        qrc_errors,
        color="#d62728",
        marker="o",
        markersize=10,
        linewidth=2.8,
        label="QRC",
    )
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(snr_list)), float(np.max(snr_list)))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test Symbol Error Rate")
    ax.set_title("Channel Equalization")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = outdir / "channel_equalization_snr.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def main() -> None:
    reservoir_params = ReservoirParams(
        n_system=2,
        n_ancilla=4,
        tau=1.0,
        seed=17462,
        hx0_base=1.0,
        hx0_std=1.0,
        hx0_scale=2.0,
        hz1_base=1.0,
        hz1_std=1.0,
        hz1_scale=0.5,
        J_scale=1.0,
        graph_kind="full",
    ).generate()
    qcfg = ChannelMapReservoirConfig(
        n_system=2,
        n_ancilla=4,
        tau=1.0,
        input_scale=1.0,
        include_bias=False,
        use_shot_noise=False,
        init_state="zero",
        hx0_vec=reservoir_params["hx0_vec"],
        hz1_vec=reservoir_params["hz1_vec"],
        J_mat=reservoir_params["J_mat"],
        seed=17462,
    )
    reservoir = ChannelMapReservoir(qcfg)
    rho_fp = reservoir.fixed_point()
    esn_cfg = ESNConfig(
        n_res=200,
        spectral_radius=0.8,
        input_scale=0.1,
        leak_rate=0.3,
        ridge_l2=1e-4,
        seed=2,
    )
    readout_cfg = SoftmaxReadoutConfig(fit_intercept=True, l2=1e-6, max_iter=1000, tol=1e-9)

    base_cfg = ChannelEqualizationDatasetConfig(
        n_train=20,
        n_test=20,
        n_symb=100,
        input_seed=17462,
    )
    snr_list = np.linspace(0.0, 25.0, 6)

    qrc_errors = []
    esn_errors = []
    logistic_errors = []
    naive_errors = []

    print("Channel equalization")
    print("SNR(dB) | QRC test SER | ESN test SER | Logistic test SER | Naive test SER")
    print("-" * 78)
    for snr_db in snr_list:
        cfg = ChannelEqualizationDatasetConfig(
            n_train=base_cfg.n_train,
            n_test=base_cfg.n_test,
            n_symb=base_cfg.n_symb,
            snr_db=float(snr_db),
            input_seed=base_cfg.input_seed,
            symbols=base_cfg.symbols,
            taps=base_cfg.taps,
            nonlin2=base_cfg.nonlin2,
            nonlin3=base_cfg.nonlin3,
        )
        dataset = generate_channel_equalization_dataset(cfg)

        qout = _evaluate_qrc(reservoir, dataset, readout_cfg, initial_state=rho_fp)
        eout = _evaluate_esn(dataset, esn_cfg, readout_cfg)
        lout = _evaluate_raw_logistic(dataset, readout_cfg)
        nout = _evaluate_naive_rounding(dataset)

        qrc_errors.append(qout["test_error_rate"])
        esn_errors.append(eout["test_error_rate"])
        logistic_errors.append(lout["test_error_rate"])
        naive_errors.append(nout["test_error_rate"])

        print(
            f"{snr_db:6.1f} | "
            f"{qout['test_error_rate']:.6f} | "
            f"{eout['test_error_rate']:.6f} | "
            f"{lout['test_error_rate']:.6f} | "
            f"{nout['test_error_rate']:.6f}"
        )

    qrc_errors_arr = np.asarray(qrc_errors, dtype=float)
    esn_errors_arr = np.asarray(esn_errors, dtype=float)
    logistic_errors_arr = np.asarray(logistic_errors, dtype=float)
    naive_errors_arr = np.asarray(naive_errors, dtype=float)
    plot_path = _plot_results(
        outdir=Path("outputs") / "channel_equalization_demo",
        snr_list=snr_list,
        qrc_errors=qrc_errors_arr,
        esn_errors=esn_errors_arr,
        logistic_errors=logistic_errors_arr,
        naive_errors=naive_errors_arr,
    )

    print()
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
