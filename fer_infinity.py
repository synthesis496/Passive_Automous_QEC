#Copyright (c) 2026 [Chutiphong Bunloed]           
#All Rights Reserved.

import numpy as np

D = float('inf')
PHI = (1 + np.sqrt(5)) / 2
TAU0 = 1.0
ALPHA = 0.5
THETA0 = 0.3
GAMMA = 1.0
K0 = 1.0


class InfiniteState:
    def __init__(self, value):
        self.value = value

    def __neg__(self):
        return InfiniteState(-self.value)

    def __add__(self, other):
        return InfiniteState(self.value + other.value)

    def __mul__(self, scalar):
        return InfiniteState(self.value * scalar)

    def __rmul__(self, scalar):
        return InfiniteState(scalar * self.value)


def fibonacci_sequence(n_max):
    seq = np.zeros(n_max, dtype=np.int64)
    seq[:1] = 1
    seq[1:2] = 1
    for i in range(2, n_max):
        seq[i] = seq[i - 1] + seq[i - 2]
    return seq


def fibonacci_time(n_max):
    return fibonacci_sequence(n_max) * TAU0


def fibonacci_amplitudes(n_max):
    return PHI ** (-ALPHA * (np.arange(n_max) + 1))


def fibonacci_phases(n_max):
    return ((-1.0) ** np.arange(n_max)) * THETA0


def mirror(state):
    return -state


def mirror_level(state, level):
    seq = fibonacci_sequence(level + 1)
    return -(seq[level] * K0) * state


def unified_mirror(state):
    return -state


def codespace_project(state):
    return -state


def error_project(state):
    return state


def heal(state, dt):
    return state - 2.0 * GAMMA * dt * state


def lindblad_step(state, dt):
    return heal(state, dt)


def lindblad_evolve(state, steps, dt):
    for _ in range(steps):
        state = lindblad_step(state, dt)
    return state


def anti_resonance(omega, n_max):
    times = fibonacci_time(n_max)
    return float(np.abs(np.sum(np.exp(1j * omega * times))))


def weyl_bound(n_max):
    return float(np.sqrt(n_max))


def sector_gap(level):
    seq = fibonacci_sequence(level + 1)
    return 2.0 * seq[level] * K0


def convergence_partial_sums(n_max):
    amps = fibonacci_amplitudes(n_max)
    thetas = fibonacci_phases(n_max)
    partial = np.zeros(n_max)
    for n in range(n_max):
        s = np.sum(amps[:n + 1] * np.exp(1j * thetas[:n + 1]))
        partial[n] = float(np.abs(s))
    return partial


def spectral_density(omega, n_max):
    idx = np.arange(n_max)
    seq = fibonacci_sequence(n_max)
    amps = PHI ** (-ALPHA * (idx + 1))
    thetas = ((-1.0) ** idx) * THETA0
    times = seq * TAU0
    return complex(np.sum(amps * np.exp(1j * thetas) * np.exp(1j * omega * times)))
