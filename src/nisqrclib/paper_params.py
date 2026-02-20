from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PaperReservoirParams:
    """
    Paper-style random parameter generation for channel-map reservoirs.
    """

    n_system: int = 4
    n_ancilla: int = 2
    tau: float = 1.0
    seed: int = 17462
    hx0_base: float = 1.0
    hz1_base: float = 1.0
    hx0_std: float = 0.3
    hz1_std: float = 0.3
    J_scale: float = 1.0

    def n_qubits(self) -> int:
        return self.n_system + self.n_ancilla

    def generate(self) -> dict:
        n = self.n_qubits()
        rs = np.random.RandomState(seed=self.seed)

        hx0 = self.hx0_base + self.hx0_std * rs.randn(n)
        hz1 = self.hz1_base + self.hz1_std * rs.randn(n)

        J_graph = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                J_graph[i, j] = 1.0

        J = self.J_scale * (rs.rand(n, n) * J_graph)
        return {
            "hx0_vec": hx0,
            "hz1_vec": hz1,
            "J_mat": J,
            "tau": float(self.tau),
            "n_system": self.n_system,
            "n_ancilla": self.n_ancilla,
            "seed": self.seed,
        }
