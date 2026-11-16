# AcoustiSolar: Simulation-First PV Micro-Crack Detection Platform

[![AcoustiSolar CI](https://github.com/acoustisolar/acoustisolar/actions/workflows/ci.yml/badge.svg)](https://github.com/acoustisolar/acoustisolar/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)

**AcoustiSolar** is a simulation-first platform for detecting, localizing, and classifying solar-panel micro-cracks using Acoustic Emission (AE) stress waves in photovoltaic laminates.

---

## 1. Monorepo Architecture

The repository is structured into modular Python packages with strict boundaries:

```text
acoustisolar/
├── packages/
│   ├── schemas/        # Pydantic v2 data models, config hashing, and JSON-Schema export
│   ├── physics/        # Analytical Lamb waves, multilayer SAFE, viscoelastic EVA, T0 & T1 wave solvers
│   ├── daqtwin/        # PZT transducer response, preamp, analog filter, ADC (jitter/skew), and trigger logic
│   ├── dsp/            # Bandpass filtering, AIC & STA/LTA onset pickers, AE features, TDOA localizer
│   ├── ml/             # Synthetic dataset factory, baseline Gradient Booster, 1D ResNet, held-out evaluation
│   └── sdk/            # Unified Python client (AcoustiSolar) and CLI tool
├── configs/            # YAML configurations for modules, DAQ, and sources
├── validation/         # Automated V&V suite, convergence checks, reciprocity tests, and reports
├── tests/              # Pytest test suite covering all packages (>90% coverage)
├── .github/workflows/  # Continuous integration workflow
├── Dockerfile          # Containerized development & testing environment
├── ASSUMPTIONS.md      # Grounded physical parameters and design assumptions log
└── DECISIONS.md        # Architecture Decision Records (ADRs)
```

---

## 2. Quickstart

### Prerequisites
- Python 3.12 or 3.13
- Core dependencies: `numpy`, `pandas`, `pydantic`, `pyyaml`, `pytest`

### Running Tests
Execute the comprehensive test suite across all packages:
```bash
python -m pytest tests/
```

### Running the V&V Suite
Run automated physics verification, multi-fidelity gap analysis, and generalization tests:
```bash
python validation/run_vv_suite.py
```

### Using the CLI
Run a single simulation and localization:
```bash
python -m acoustisolar.cli simulate --x 520 --y 880 --temp 25.0
```

Batch-generate a synthetic labeled dataset:
```bash
python -m acoustisolar.cli generate --n 50 --output data_events.csv
```

Run a held-out module ML benchmark (evaluating strictly on unseen module geometry):
```bash
python -m acoustisolar.cli benchmark --train-n 40 --test-n 20
```

---

## 3. Python SDK Usage

```python
from acoustisolar import AcoustiSolar

# Initialize client
client = AcoustiSolar()

# 1. Run simulation with DAQ acquisition & hyperbolic TDOA localization
result = client.simulate_event(x_mm=500.0, y_mm=800.0)
print(f"Triggered: {result['triggered']}")
print(f"Estimated Position: ({result['localization']['x_mm']:.1f}, {result['localization']['y_mm']:.1f}) mm")

# 2. Batch-generate bit-reproducible synthetic dataset
waveforms, df_index = client.generate_synthetic_dataset(n_events=20)

# 3. Evaluate on held-out module configuration (zero data leakage)
metrics = client.evaluate_held_out(n_train=40, n_test=20)
print(f"New-fracture F1: {metrics['f1_score']:.3f}, ECE: {metrics['ece']:.3f}")
```

---

## 3a. Research-Driven Revision (2026-11)

A literature review and code evaluation ([docs/research/](docs/research/)) led to a revision
of the physics, signal processing, ML and verification layers.  Highlights:

- **Physics:** fixed a sign error in the multilayer SAFE solver (its dispersion curves were
  unphysical); Tier-0 now uses mode-resolved laminate physics from the configured stack,
  source-dependent A0/S0 excitation, out-of-plane PZT coupling with aperture, frame
  reflections, causal spectral shaping and realistic absolute levels (the old data were 99 %
  ADC-saturated).  Tier-1 is documented as a scalar solver; its sponge, geometry and
  time-indexing bugs are fixed (2nd-order convergence verified).
- **DAQ / DSP:** causal analog filters, multi-hit and floating-threshold triggering,
  vectorised AIC, causal STA/LTA, mode-aware gated TDOA with robust all-sensor solving,
  Delta-T mapping, extended AE features (RA, average frequency, partial powers, early/late
  energy ratio, arrival spread, edge distance) in arrival order.
- **ML:** 10-class labels incl. hail-induced cracks, a real trainable PyTorch residual CNN,
  temperature scaling, conformal prediction sets, Mahalanobis "unknown source" detection,
  GPD-based false-alarm thresholds, hash-verified model loading.
- **Verification:** every V&V check computes a real quantity (`validation/run_vv_suite.py`,
  results in `validation/out/`); the scale benchmark writes
  `validation/kpi_benchmark_summary.json`.  **All results are simulation-only.**

---

## 4. Key Milestones Completed (Phase 0–1 MVP-Sim)

- **M1: Monorepo Skeleton:** Python workspace, CI actions, Dockerfile, devcontainer, Hydra/YAML configs.
- **M2: Schemas Package:** Pydantic v2 data models for twin, sensor, DAQ, source, event record; deterministic SHA-256 config hashing; JSON-Schema export.
- **M3: Physics Core:** Exact Rayleigh-Lamb dispersion for S0/A0 modes, multilayer SAFE eigenvalue solver, temperature/frequency-dependent viscoelastic EVA damping model.
- **M4: Tier-0 Simulator:** Dispersion-aware frequency-domain Green's function ray tracer with geometric spreading, source library (cracks & environmental noise).
- **M5: DAQ Twin:** Resonant PZT transducer, analog front-end, ADC digitization with quantization, jitter, channel skew, and hardware trigger logic.
- **M6: DSP Engine:** AIC (Maeda 1985) and STA/LTA onset pickers, classic AE feature extraction, dispersion-corrected TDOA multilateration with covariance ellipses.
- **M7: Dataset Factory:** Deterministic, seeded batch synthetic generator with domain randomization.
- **M8: ML Baseline & Deep Models:** Gradient Boosted Trees, 1D-ResNet waveform classifier, held-out module configuration evaluation harness (PRD 9.5).
- **M9: Tier-1 Solver:** 2D *scalar* wave solver (self-adjoint flux form, heterogeneous media) with absorbing sponge or reflecting (frame) boundaries and Courant stability compliance.
- **M10: Validation Suite & Report:** Automated V&V suite (`validation/run_vv_suite.py`) computing dispersion accuracy vs an independent reference, SAFE regression, causality, heterogeneous reciprocity, observed grid-convergence order, Tier-0/Tier-1 kinematic agreement and held-out ML metrics (KPI shortfalls reported as known gaps).

