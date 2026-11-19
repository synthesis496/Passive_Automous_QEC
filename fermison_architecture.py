#Copyright © 2026 Chutiphong Bunloed
#All Rights Reserved.

import numpy as np
import scipy.linalg as la

class FermisonArchitecture:
    def __init__(self, gamma=0.1, dt=0.01, steps=1000, N=32):
        self.gamma = gamma
        self.dt = dt
        self.steps = steps
        self.N = N
        self.d = 5
        self.phi = (1 + np.sqrt(5)) / 2
        self.psi = (1 - np.sqrt(5)) / 2
        self.L_heal = self._build_l_heal()
        self.U_fold = self._build_u_fold()
        self.H_eff = self._build_h_eff()
        self.L_super = self._build_lindblad_super()
        self.H_super = self._build_hamiltonian_super()
        self.rho0 = self._build_initial_state()

    def _build_l_heal(self):
        L = np.zeros((self.d, self.d), dtype=complex)
        L[0, 2] = np.sqrt(self.gamma)
        L[1, 3] = np.sqrt(self.gamma)
        L[0, 4] = np.sqrt(self.gamma)
        return L

    def _build_u_fold(self):
        U = np.zeros((self.d, self.d), dtype=complex)
        indices = np.arange(self.d)
        mirror_indices = self.d - 1 - indices
        U[indices, mirror_indices] = 1
        return U

    def _build_h_eff(self):
        return -1j * self.L_heal.conj().T @ self.L_heal

    def _build_lindblad_super(self):
        L = self.L_heal
        L_dag = L.conj().T
        term1 = np.kron(L, L.conj())
        term2 = np.kron(L_dag @ L, np.eye(self.d))
        term3 = np.kron(np.eye(self.d), (L_dag @ L).T)
        return term1 - 0.5 * (term2 + term3)

    def _build_hamiltonian_super(self):
        H = self.H_eff
        term1 = np.kron(H, np.eye(self.d))
        term2 = np.kron(np.eye(self.d), H.T)
        return -1j * (term1 - term2)

    def _build_initial_state(self):
        rho = np.zeros((self.d, self.d), dtype=complex)
        rho[2, 2] = 1
        return rho

    def evolve(self):
        L_total = self.H_super + self.L_super
        rho_vec = self.rho0.flatten()
        rho_t = la.expm(L_total * self.dt * self.steps) @ rho_vec
        return rho_t.reshape((self.d, self.d))

    def validate_dark_state(self):
        codespace = np.zeros((self.d, self.d), dtype=complex)
        codespace[0, 0] = 1
        codespace[1, 1] = 1
        L_eff = self.L_heal @ codespace @ self.L_heal.conj().T
        return np.allclose(L_eff, 0)

    def validate_no_traps(self):
        eigenvalues = la.eigvals(self.L_super)
        max_real = np.max(np.real(eigenvalues))
        return max_real < 1e-10

    def validate_convergence(self):
        eigenvalues = la.eigvals(self.L_super)
        sorted_reals = np.sort(np.real(eigenvalues))
        gap = sorted_reals[1] - sorted_reals[0]
        return gap > 0

    def get_fibonacci_sequence(self, n):
        indices = np.arange(n)
        F_n = (self.phi**indices - self.psi**indices) / np.sqrt(5)
        return np.round(np.real(F_n)).astype(int)

    def build_omni_mirror_network(self):
        A = np.zeros((self.N, self.N))
        indices = np.arange(self.N)
        mirror_indices = self.N - 1 - indices
        A[indices, mirror_indices] = 1
        return A

    def build_fibonacci_clock(self):
        n = np.arange(1, self.N + 1)
        F_n = (self.phi**n - self.psi**n) / np.sqrt(5)
        freq = 1.0 / np.round(np.real(F_n)).astype(int)
        return freq

