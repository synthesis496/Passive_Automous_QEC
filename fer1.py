# Copyright (c) 2026 Chutiphong Bunloed
# All Rights Reserved.

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla


class FermisonEngineUltraJAX:
    def __init__(self, N=32, gamma=0.1, tau0=1.0, alpha=0.5):
        if N <= 0:
            raise ValueError("N must be a positive integer.")
        self.N = int(N)
        self.gamma = float(gamma)
        self.tau0 = float(tau0)
        self.alpha = float(alpha)
        self.phi = (1.0 + jnp.sqrt(5.0)) / 2.0
        self.psi = (1.0 - jnp.sqrt(5.0)) / 2.0
        self._initialize()

    def _initialize(self):
        d = self.N
        L = jnp.zeros((d, d), dtype=complex)
        if d >= 3:
            rows = jnp.arange(d - 2)
            L = L.at[rows, rows + 2].set(jnp.sqrt(self.gamma))
        self.L = L
        self.L_dag = jnp.conj(L.T)
        self.H_eff = -1j * (self.L_dag @ self.L)
        self.R_mirror = jnp.fliplr(jnp.eye(d, dtype=complex))

    def _build_dynamic_lindblad(self, current_L):
        d = self.N
        Ld = jnp.conj(current_L.T)
        LdL = Ld @ current_L
        I = jnp.eye(d, dtype=complex)
        t1 = jnp.kron(current_L, jnp.conj(current_L))
        t2 = jnp.kron(LdL, I)
        t3 = jnp.kron(I, LdL.T)
        return t1 - 0.5 * (t2 + t3)

    def _build_dynamic_hamiltonian(self):
        d = self.N
        I = jnp.eye(d, dtype=complex)
        return -1j * (jnp.kron(self.H_eff, I) - jnp.kron(I, self.H_eff.T))

    def fibonacci_sequence(self, n):
        idx = jnp.arange(max(1, int(n)))
        values = (self.phi ** idx - self.psi ** idx) / jnp.sqrt(5.0)
        return jnp.round(jnp.real(values)).astype(int)

    def fibonacci_clock_signal(self, omega, modes=64):
        m = jnp.arange(1, modes + 1)
        amp = self.phi ** (-self.alpha * m)
        phase = self.phi ** m * omega * self.tau0
        return jnp.sum(amp * jnp.cos(phase))

    def mirror_transform(self, x):
        x = jnp.asarray(x)
        if x.ndim == 0:
            return x
        if x.ndim == 1:
            return self.R_mirror @ x
        if x.ndim == 2:
            return self.R_mirror @ x @ self.R_mirror
        return x

    def mirror_cancellation(self, x):
        return x + self.mirror_transform(x)

    def attractor_map(self, z, a=-0.5, beta=1.0):
        z = jnp.asarray(z)
        return jnp.exp((a + 1j * beta) * z)

    def limit_cycle(self, z0, iterations=1000, a=-0.5, beta=1.0):
        z = jnp.asarray(z0)
        for _ in range(max(0, int(iterations))):
            z = self.attractor_map(z, a, beta)
        return z

    @jax.jit
    def evolve_ultra(self, rho0, steps, dt=0.01, omega_clock=1.0):
        rho0 = jnp.asarray(rho0, dtype=complex)
        H_super = self._build_dynamic_hamiltonian()
        rho_vec_init = rho0.reshape(-1)

        def time_step_scan(carry_rho_vec, step_idx):
            t = step_idx * dt
            clock_pulse = self.fibonacci_clock_signal(omega=omega_clock * t)
            gated_L = self.L * clock_pulse
            L_super_dynamic = self._build_dynamic_lindblad(gated_L)
            generator = H_super + L_super_dynamic
            next_rho_vec = jla.expm(generator * dt) @ carry_rho_vec
            return next_rho_vec, next_rho_vec

        final_rho_vec, _ = jax.lax.scan(time_step_scan, rho_vec_init, jnp.arange(int(steps)))
        return final_rho_vec.reshape((self.N, self.N))

    def attractor_fidelity(self, rho0, steps=500, dt=0.01, omega_clock=1.0):
        rho_final = self.evolve_ultra(rho0, steps, dt, omega_clock)
        diagonal = jnp.real(jnp.diag(rho_final))
        if self.N < 2:
            return 0.0
        return float(jnp.sum(diagonal[:2]))

    def residual_heat(self, rho0, steps=500, dt=0.01, omega_clock=1.0):
        rho_final = self.evolve_ultra(rho0, steps, dt, omega_clock)
        diagonal = jnp.real(jnp.diag(rho_final))
        if self.N <= 2:
            return 0.0
        return float(jnp.sum(diagonal[2:]))


if __name__ == "__main__":
    engine = FermisonEngineUltraJAX(N=8, gamma=0.1, tau0=1.0, alpha=0.5)
    rho0 = jnp.eye(engine.N, dtype=complex) / engine.N

    rho_final = engine.evolve_ultra(rho0, steps=10, dt=0.01, omega_clock=1.0)
    fidelity = engine.attractor_fidelity(rho0, steps=10, dt=0.01, omega_clock=1.0)
    heat = engine.residual_heat(rho0, steps=10, dt=0.01, omega_clock=1.0)

    print("Final state shape:", rho_final.shape)
    print("Attractor fidelity:", fidelity)
    print("Residual heat:", heat)
