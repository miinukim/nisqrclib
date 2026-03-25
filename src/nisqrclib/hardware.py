from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, partial_trace

from .reservoir_params import ReservoirParams


def _single_pauli_label(n: int, q: int, p: str) -> str:
    chars = ["I"] * n
    chars[n - 1 - q] = p
    return "".join(chars)


def _two_pauli_label(n: int, q1: int, p1: str, q2: int, p2: str) -> str:
    chars = ["I"] * n
    chars[n - 1 - q1] = p1
    chars[n - 1 - q2] = p2
    return "".join(chars)


@dataclass
class HardwareTrajectoryReservoirConfig:
    n_system: int = 4
    n_ancilla: int = 2
    tau: float = 1.0
    input_scale: float = 1.0
    include_bias: bool = True
    init_state: str = "zero"  # "maximally_mixed" or "zero"
    shots: int = 1024
    seed: int = 17462
    hx0_vec: Optional[np.ndarray] = None
    hz1_vec: Optional[np.ndarray] = None
    J_mat: Optional[np.ndarray] = None


class HardwareTrajectoryReservoir:
    """
    Shot-trajectory emulator:
    - Each shot carries a pure state trajectory.
    - At each step, ancilla are projectively measured and then reset to |0...0>.
    - Features are empirical ancilla probabilities from finite shots.
    """

    def __init__(self, cfg: HardwareTrajectoryReservoirConfig):
        self.cfg = cfg
        self.nS = cfg.n_system
        self.nA = cfg.n_ancilla
        self.n = self.nS + self.nA
        if self.nA <= 0:
            raise ValueError("n_ancilla must be >= 1 for hardware trajectory readout.")
        if cfg.shots <= 0:
            raise ValueError("shots must be > 0.")
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.hx0_vec is None or cfg.hz1_vec is None or cfg.J_mat is None:
            gen = ReservoirParams(
                n_system=cfg.n_system,
                n_ancilla=cfg.n_ancilla,
                tau=cfg.tau,
                seed=cfg.seed,
            ).generate()
            self.hx0 = np.asarray(gen["hx0_vec"], dtype=float)
            self.hz1 = np.asarray(gen["hz1_vec"], dtype=float)
            self.J = np.asarray(gen["J_mat"], dtype=float)
            self.tau = float(gen["tau"])
        else:
            self.hx0 = np.asarray(cfg.hx0_vec, dtype=float)
            self.hz1 = np.asarray(cfg.hz1_vec, dtype=float)
            self.J = np.asarray(cfg.J_mat, dtype=float)
            self.tau = float(cfg.tau)

        self.H0, self.H1 = self._build_H0_H1()
        self._unitary_cache: Dict[float, Operator] = {}
        self._dS = 2**self.nS
        self._dA = 2**self.nA
        self.rho_gA = DensityMatrix.from_label("0" * self.nA)
        I_S = Operator(np.eye(self._dS, dtype=complex))
        self._projectors = []
        for m in range(self._dA):
            proj = np.zeros((self._dA, self._dA), dtype=complex)
            proj[m, m] = 1.0
            self._projectors.append(Operator(proj).tensor(I_S).data)

    def _build_H0_H1(self) -> tuple[SparsePauliOp, SparsePauliOp]:
        h0_terms = []
        for i in range(self.n):
            h0_terms.append((_single_pauli_label(self.n, i, "X"), float(self.hx0[i])))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Jij = self.J[i, j]
                if Jij != 0.0:
                    h0_terms.append((_two_pauli_label(self.n, i, "Z", j, "Z"), float(Jij)))
        H0 = SparsePauliOp.from_list(h0_terms)

        h1_terms = []
        for i in range(self.n):
            h1_terms.append((_single_pauli_label(self.n, i, "Z"), float(self.hz1[i])))
        H1 = SparsePauliOp.from_list(h1_terms)
        return H0, H1

    def _unitary_for_input(self, ueff: float) -> Operator:
        cached = self._unitary_cache.get(ueff)
        if cached is not None:
            return cached
        H = self.H0 + (ueff * self.H1)
        Hmat = H.to_matrix(sparse=False)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            U = Operator(expm(-1j * self.tau * Hmat))
        if not np.isfinite(U.data).all():
            raise FloatingPointError("Non-finite values in hardware trajectory unitary.")
        self._unitary_cache[ueff] = U
        return U

    def _init_system_density(self) -> DensityMatrix:
        if self.cfg.init_state == "zero":
            return DensityMatrix.from_label("0" * self.nS)
        return DensityMatrix(np.eye(self._dS, dtype=complex) / float(self._dS))

    def run(self, inputs: Sequence[float]) -> np.ndarray:
        T = len(inputs)
        counts = np.zeros((T, self._dA), dtype=int)

        for _ in range(int(self.cfg.shots)):
            rhoSE = self._init_system_density().tensor(self.rho_gA)
            for t, u in enumerate(inputs):
                U = self._unitary_for_input(self.cfg.input_scale * float(u))
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    rhoSE = rhoSE.evolve(U)
                if not np.isfinite(rhoSE.data).all():
                    raise FloatingPointError("Non-finite values in hardware trajectory state update.")

                rhoE = partial_trace(rhoSE, qargs=list(range(self.nS)))
                probs = np.real(np.diag(rhoE.data)).astype(float)
                probs = np.clip(probs, 0.0, 1.0)
                s = float(probs.sum())
                if s <= 1e-18:
                    probs = np.ones(self._dA, dtype=float) / float(self._dA)
                else:
                    probs = probs / s

                m = int(self.rng.choice(np.arange(self._dA), p=probs))
                counts[t, m] += 1

                Pm = self._projectors[m]
                pm = max(float(probs[m]), 1e-18)
                with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                    rho_post = Pm @ rhoSE.data @ Pm.conj().T
                rho_post = rho_post / pm
                rhoSE = DensityMatrix(rho_post)
                rhoS = partial_trace(rhoSE, qargs=list(range(self.nS, self.n)))
                rhoSE = rhoS.tensor(self.rho_gA)

        probs_emp = counts.astype(float) / float(self.cfg.shots)
        X = np.hstack([np.ones((T, 1)), probs_emp]) if self.cfg.include_bias else probs_emp
        if not np.isfinite(X).all():
            raise FloatingPointError("Non-finite features from hardware trajectory reservoir.")
        return X

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        return self.run(inputs)
