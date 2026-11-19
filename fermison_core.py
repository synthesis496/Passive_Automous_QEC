#CopyRight (c)2026 Chutiphong Bunloed
#All Rights Reserved.

import numpy as np
import scipy.linalg as la

class _CoreEngine:
    def __init__(self, N=32, d=5, gamma=0.1):
        self.N = N
        self.d = d
        self.gamma = gamma
        self.phi = (1 + np.sqrt(5)) / 2
        self.psi = (1 - np.sqrt(5)) / 2
        self._build_operators()

    def _build_operators(self):
        d = self.d
        L = np.zeros((d, d), dtype=complex)
        L[0, 2] = np.sqrt(self.gamma)
        L[1, 3] = np.sqrt(self.gamma)
        L[0, 4] = np.sqrt(self.gamma)
        self.L = L
        self.L_dag = L.conj().T
        self.H_eff = -1j * self.L_dag @ self.L
        self.L_super = self._build_lindblad()
        self.H_super = self._build_hamiltonian()

    def _build_lindblad(self):
        L, L_dag, d = self.L, self.L_dag, self.d
        t1 = np.kron(L, L.conj())
        t2 = np.kron(L_dag @ L, np.eye(d))
        t3 = np.kron(np.eye(d), (L_dag @ L).T)
        return t1 - 0.5 * (t2 + t3)

    def _build_hamiltonian(self):
        H, d = self.H_eff, self.d
        return -1j * (np.kron(H, np.eye(d)) - np.kron(np.eye(d), H.T))

    def evolve(self, rho0, dt, steps):
        L_total = self.H_super + self.L_super
        rho_vec = rho0.flatten()
        rho_t = la.expm(L_total * dt * steps) @ rho_vec
        return rho_t.reshape((self.d, self.d))

    def fibonacci(self, n):
        idx = np.arange(n)
        return np.round(np.real((self.phi**idx - self.psi**idx) / np.sqrt(5))).astype(int)

    def omni_mirror(self):
        A = np.zeros((self.N, self.N))
        idx = np.arange(self.N)
        A[idx, self.N - 1 - idx] = 1
        return A

    def eigen_gap(self):
        vals = la.eigvals(self.L_super)
        sorted_reals = np.sort(np.real(vals))
        return sorted_reals[1] - sorted_reals[0]

    def dark_state_check(self):
        d = self.d
        codespace = np.zeros((d, d), dtype=complex)
        codespace[0, 0] = 1
        codespace[1, 1] = 1
        L_eff = self.L @ codespace @ self.L_dag
        return np.allclose(L_eff, 0)
