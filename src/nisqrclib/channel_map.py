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
class ChannelMapReservoirConfig:
    n_system: int = 4
    n_ancilla: int = 2
    tau: float = 1.0
    input_scale: float = 1.0
    include_bias: bool = True
    use_shot_noise: bool = False
    shots: int = 4096
    init_state: str = "maximally_mixed"  # "maximally_mixed" or "zero"
    hx0_vec: Optional[np.ndarray] = None
    hz1_vec: Optional[np.ndarray] = None
    J_mat: Optional[np.ndarray] = None
    seed: int = 17462
    connectivity_kind: str = "full"


class ChannelMapReservoir:
    """
    Channel-map reservoir:

    rho_S -> rho_SE = rho_S ⊗ |0><0|^{⊗A}
         -> evolve under U(u)=exp(-i tau (H0 + u H1))
         -> readout from ancilla marginal diag(rho_E)
         -> update rho_S = Tr_E(rho_SE), then reattach ancilla |0><0|^{⊗A}
    """

    def __init__(self, cfg: ChannelMapReservoirConfig):
        self.cfg = cfg
        self.nS = cfg.n_system
        self.nA = cfg.n_ancilla
        self.n = self.nS + self.nA
        if self.nA <= 0:
            raise ValueError("n_ancilla must be >= 1 for channel-map readout.")
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.hx0_vec is None or cfg.hz1_vec is None or cfg.J_mat is None:
            gen = ReservoirParams(
                n_system=cfg.n_system,
                n_ancilla=cfg.n_ancilla,
                tau=cfg.tau,
                seed=cfg.seed,
                graph_kind=cfg.connectivity_kind,
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
        self._fixed_point_cache: np.ndarray | None = None
        self.rho_gA = DensityMatrix.from_label("0" * self.nA)
        self.reset()

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

    def reset(self, rhoS0: Optional[np.ndarray] = None) -> None:
        if rhoS0 is None:
            if self.cfg.init_state == "zero":
                self.rhoS = DensityMatrix.from_label("0" * self.nS)
            else:
                dS = 2**self.nS
                self.rhoS = DensityMatrix(np.eye(dS, dtype=complex) / dS)
        else:
            self.rhoS = DensityMatrix(np.asarray(rhoS0, dtype=complex))
        self.rhoSE = self.rhoS.tensor(self.rho_gA)

    def _unitary_for_input(self, ueff: float) -> Operator:
        cached = self._unitary_cache.get(ueff)
        if cached is not None:
            return cached
        H = self.H0 + (ueff * self.H1)
        Hmat = H.to_matrix(sparse=False)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            U = Operator(expm(-1j * self.tau * Hmat))
        if not np.isfinite(U.data).all():
            raise FloatingPointError("Non-finite values in channel-map unitary.")
        self._unitary_cache[ueff] = U
        return U

    def _memory_channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        op_memory = np.asarray(op_memory, dtype=complex)
        if op_memory.shape != (2**self.nS, 2**self.nS):
            raise ValueError("op_memory has incompatible shape.")
        U = self._unitary_for_input(self.cfg.input_scale * float(u)).data
        joint = np.kron(op_memory, self.rho_gA.data)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            evolved = U @ joint @ U.conj().T
        if not np.isfinite(evolved).all():
            raise FloatingPointError("Non-finite values in channel-map operator evolution.")
        dS = 2**self.nS
        dA = 2**self.nA
        reduced = np.trace(evolved.reshape(dS, dA, dS, dA), axis1=1, axis2=3)
        if not np.isfinite(reduced).all():
            raise FloatingPointError("Non-finite values in reduced system operator.")
        return reduced

    def fixed_point(self) -> np.ndarray:
        if self._fixed_point_cache is not None:
            return self._fixed_point_cache.copy()

        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        one_qubit_basis = [I, X, Y, Z]

        pauli_basis = []
        for idx in range(4**self.nS):
            word = np.base_repr(idx, base=4).zfill(self.nS)
            op = np.array([[1]], dtype=complex)
            for char in word:
                op = np.kron(op, one_qubit_basis[int(char)])
            pauli_basis.append(op)

        dim = 2**self.nS
        trans = np.zeros((len(pauli_basis), len(pauli_basis)), dtype=float)
        for col, basis_op in enumerate(pauli_basis):
            out = self._memory_channel(0.0, basis_op)
            for row, measure_op in enumerate(pauli_basis):
                trans[row, col] = float(np.real(np.trace(out @ measure_op)) / dim)

        A = trans[1:, 1:]
        b = trans[1:, 0]
        x = np.linalg.solve(np.eye(A.shape[0]) - A, b)

        rho_tomo = np.zeros((dim, dim), dtype=complex)
        for idx, coeff in enumerate(x, start=1):
            rho_tomo += coeff * pauli_basis[idx]
        rho_fp = (rho_tomo + np.eye(dim, dtype=complex)) / dim
        rho_fp = 0.5 * (rho_fp + rho_fp.conj().T)
        rho_fp /= np.trace(rho_fp)
        if not np.isfinite(rho_fp).all():
            raise FloatingPointError("Non-finite values in channel-map fixed point.")

        self._fixed_point_cache = rho_fp.copy()
        return rho_fp

    def step(self, u: float) -> np.ndarray:
        ueff = self.cfg.input_scale * float(u)
        U = self._unitary_for_input(ueff)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            self.rhoSE = self.rhoSE.evolve(U)
        if not np.isfinite(self.rhoSE.data).all():
            raise FloatingPointError("Non-finite values in channel-map state update.")

        rhoE = partial_trace(self.rhoSE, qargs=list(range(self.nS)))
        probs = np.real(np.diag(rhoE.data)).astype(float)
        probs = np.clip(probs, 0.0, 1.0)
        s = probs.sum()
        if s <= 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs = probs / s
        if self.cfg.use_shot_noise:
            counts = self.rng.multinomial(self.cfg.shots, probs)
            probs = counts.astype(float) / float(self.cfg.shots)

        self.rhoS = partial_trace(self.rhoSE, qargs=list(range(self.nS, self.n)))
        self.rhoSE = self.rhoS.tensor(self.rho_gA)

        if self.cfg.include_bias:
            return np.concatenate([[1.0], probs])
        return probs

    def run(self, inputs: Sequence[float]) -> np.ndarray:
        X = []
        for u in inputs:
            X.append(self.step(float(u)))
        X = np.vstack(X)
        if not np.isfinite(X).all():
            raise FloatingPointError("Non-finite features from channel-map reservoir.")
        return X

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        # Compatibility with STMTaskRunner interface.
        return self.run(inputs)
