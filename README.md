# Passive Autonomous QEC

A small experimental Python project exploring a passive, autonomous quantum-error-correction-inspired dynamical model driven by Fibonacci-like temporal structure, mirror symmetry cancellation, and dissipative attractor dynamics.

## Overview

This repository contains two prototype implementations:

- `fer1.py` — a JAX-based simulation of a dissipative, Fibonacci-clock-driven quantum-like system.
- `fer_infinity.py` — a lightweight NumPy prototype implementing the same conceptual ideas in an algebraic and scalar form.

The code is not a full production library or package. It is a research prototype aimed at exploring how:

- Fibonacci time signals can be used as dynamic modulation,
- mirror symmetry / cancellation terms affect state evolution,
- dissipative Lindblad-like dynamics can steer the system toward an attractor,
- residual heat and fidelity-like measures can be tracked numerically.

---

## Repository layout

```text
Passive_Automous_QEC/
├── README.md
├── fer1.py
├── fer_infinity.py
├── fermi_pic_qec.jpg
├── graph_fermiqec.jpg
├── graph_fermiqecb.jpg
├── LICENSE
└── ...
```

### Files

- `fer1.py`
  - Main experimental implementation.
  - Uses JAX and JAX NumPy.
  - Defines a `FermisonEngineUltraJAX` class.

- `fer_infinity.py`
  - Conceptual / scalar prototype.
  - Uses NumPy for simpler computations.
  - Implements Fibonacci time, mirror operations, healing, anti-resonance, and spectral density helpers.

- `fermi_pic_qec.jpg`, `graph_fermiqec.jpg`, `graph_fermiqecb.jpg`
  - Visual outputs or experiment plots for the QEC-style dynamics.

---

## Main implementation: `fer1.py`

`fer1.py` introduces the class:

```python
FermisonEngineUltraJAX
```

It models a dissipative quantum-like system with the following concepts:

### 1. Fibonacci clock modulation
The code builds a signal via:

```python
fibonacci_clock_signal(self, omega, modes=64)
```

This uses a golden-ratio-based frequency structure:

```python
amp = self.phi ** (-self.alpha * m)
phase = self.phi ** m * omega * self.tau0
```

This creates a modulation pattern reminiscent of a Fibonacci-driven temporal clock.

### 2. Mirror cancellation
The class includes:

```python
mirror_transform(self, x)
mirror_cancellation(self, x)
```

These apply a mirror / reversal transformation to the state and combine it with the original state to emulate symmetric cancellation effects.

### 3. Attractor and limit-cycle dynamics
The model includes:

```python
attractor_map(self, z, a=-0.5, beta=1.0)
limit_cycle(self, z0, iterations=1000, a=-0.5, beta=1.0)
```

This implements a complex attractor map of the form:

```python
exp((a + i * beta) * z)
```

representing a dissipative/contractive dynamic flow.

### 4. Lindblad-like evolution
The evolution routine is:

```python
evolve_ultra(self, rho0, steps, dt=0.01, omega_clock=1.0)
```

It constructs a dynamic Liouvillian-like generator using:

- a Hamiltonian contribution,
- a dynamic Lindblad generator,
- a Fibonacci-clock-controlled gate operator.

This mimics a dissipative quantum evolution process in which the clock signal modulates the error/healing dynamics.

### 5. Observables
The class also defines:

- `attractor_fidelity(rho0, steps=500, dt=0.01, omega_clock=1.0)`
- `residual_heat(rho0, steps=500, dt=0.01, omega_clock=1.0)`

These metrics summarize the state’s tendency toward a stable attractor and the leftover non-thermal or residual contribution after evolution.

---

## Secondary prototype: `fer_infinity.py`

This file is a simpler, more algebraic prototype that captures the conceptual ingredients without JAX acceleration.

It defines functions for:

- Fibonacci sequence generation
- Fibonacci time sampling
- amplitude and phase generation
- mirror symmetry operations
- healing and Lindblad-style evolution
- anti-resonance and spectral density calculation
- convergence partial sums

Examples of key functions:

- `fibonacci_sequence(n_max)`
- `fibonacci_time(n_max)`
- `fibonacci_amplitudes(n_max)`
- `mirror(state)`
- `heal(state, dt)`
- `lindblad_step(state, dt)`
- `anti_resonance(omega, n_max)`
- `spectral_density(omega, n_max)`

This file acts more like a symbolic or analytical prototype for the JAX implementation.

---

## Dependencies

### Python packages

- `numpy`
- `jax`
- `jax.numpy`
- `jax.scipy.linalg`

Install with:

```bash
pip install numpy jax jaxlib
```

---

## Example usage

### JAX prototype

```python
from fer1 import FermisonEngineUltraJAX
import jax.numpy as jnp

engine = FermisonEngineUltraJAX(N=32, gamma=0.1, tau0=1.0, alpha=0.5)

rho0 = jnp.eye(32, dtype=complex) / 32

rho_final = engine.evolve_ultra(rho0, steps=200, dt=0.01, omega_clock=1.0)
print(engine.attractor_fidelity(rho0, steps=200, dt=0.01, omega_clock=1.0))
print(engine.residual_heat(rho0, steps=200, dt=0.01, omega_clock=1.0))
```

### NumPy prototype

```python
from fer_infinity import fibonacci_sequence, spectral_density

seq = fibonacci_sequence(20)
print(seq)
print(spectral_density(1.0, 20))
```

---

## Scientific interpretation

The code is best understood as a toy model combining three ideas:

1. Fibonacci-based temporal modulation
2. mirror / parity cancellation structure
3. dissipative attractor dynamics for approximate autonomous stabilization

This is conceptually related to:

- passive error-correction ideas,
- dissipative state stabilization,
- control-inspired quantum dynamics,
- nontrivial time-ordered modulation using Fibonacci-like frequencies.

It is not yet a benchmarked or peer-reviewed QEC implementation; it is an exploratory numerical model.

---

## Notes and limitations

- The repository is a research prototype, not yet a full software package.
- The code is experimental and may need cleanup, validation, and tests before being used as a serious simulation framework.
- The naming suggests a link to autonomous passive QEC, but the implementation should be interpreted as a conceptual prototype rather than a complete QEC system.
- The plots in the repository likely serve as proof-of-concept visuals for the dynamics under study.

---

## License

See [LICENSE](./LICENSE).

---

## Summary

This repository presents a compact numerical exploration of a Fibonacci-modulated, symmetry-aware, dissipative quantum dynamics model. The code is intentionally lightweight and experimental, but it captures a clear conceptual theme: using Fibonacci temporal structure and mirror-cancellation logic to steer a system toward a stable attractor-like state.
