# Usage guide

This project is a small research prototype. The code is intentionally lightweight and meant to be run directly from the repository root.

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

If you prefer to install manually:

```bash
pip install numpy jax jaxlib
```

---

## Running the main prototype

From the repository root:

```bash
python fer1.py
```

This runs the JAX-based experimental engine. The implementation is designed to be adapted and extended rather than treated as a final product.

---

## Running the lightweight NumPy prototype

```bash
python fer_infinity.py
```

This version is useful for quick symbolic or scalar experiments without the heavier JAX runtime.

---

## Example workflow

### 1. Explore the concept quickly

Use `fer_infinity.py` for fast iteration on the mathematical idea.

### 2. Investigate the JAX simulation

Use `fer1.py` to test the full dynamic loop and metrics.

### 3. Inspect outputs

The image files in the root directory can be used as visual reference artifacts for the dynamics the code was designed to explore.

---

## Typical operations

### Build a simulation object

```python
from fer1 import FermisonEngineUltraJAX

engine = FermisonEngineUltraJAX(N=32, gamma=0.1, tau0=1.0, alpha=0.5)
```

### Run an evolution step

```python
rho0 = jnp.eye(32, dtype=complex) / 32
rho_final = engine.evolve_ultra(rho0, steps=200, dt=0.01, omega_clock=1.0)
```

### Compute metrics

```python
fidelity = engine.attractor_fidelity(rho0, steps=200, dt=0.01, omega_clock=1.0)
heat = engine.residual_heat(rho0, steps=200, dt=0.01, omega_clock=1.0)
print(fidelity)
print(heat)
```

---

## Recommended experimentation flow

- Start with `fer_infinity.py`
- Tune the Fibonacci parameters and damping terms
- Move to `fer1.py` for higher-dimensional or more dynamic simulation
- Track how the attractor fidelity evolves over time
- Compare against residual heat and system behavior

---

## Important note

This repository is intentionally not a polished application. It is meant to be a concept exploration tool, so experimentation and iteration are expected.

---

## Related files

- [../README.md](../README.md)
- [../fer1.py](../fer1.py)
- [../fer_infinity.py](../fer_infinity.py)
