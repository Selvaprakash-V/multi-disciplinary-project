# AcoustiSolar Architecture Decision Records (ADR)

## ADR-001: Monorepo Structure and Package Boundaries
- **Status:** Approved
- **Context:** PRD Section 16.1 defines a modular platform spanning physics, DAQ modeling, DSP, ML, and public SDK.
- **Decision:** Use a clean `src/` layout under `packages/` with explicit import paths:
  - `acoustisolar_schemas`: Pure Pydantic v2 data models and serialization.
  - `acoustisolar_physics`: Pure numerical Lamb wave & multilayer SAFE dispersion solvers, EVA viscoelasticity, wavefield solvers.
  - `acoustisolar_daqtwin`: PZT sensor transfer functions, analog chain, ADC, jitter, and trigger logic.
  - `acoustisolar_dsp`: Filters, AIC & STA/LTA pickers, AE feature extractors, dispersion-corrected TDOA multilateration.
  - `acoustisolar_ml`: Feature dataset preparation, baseline tree models, 1D-CNN deep models, leakage-free evaluation.
  - `acoustisolar`: Unified SDK and CLI.
- **Consequence:** Zero circular dependencies; packages can be imported and tested independently.

## ADR-002: Dependency Independence and Pure Numerical Implementation
- **Status:** Approved
- **Context:** The environment has Python 3.13 and NumPy/Pandas/Pydantic/PyYAML. Heavy or platform-specific external packages may introduce deployment fragility in constrained CI or embedded environments.
- **Decision:** Implement all core physics (Rayleigh-Lamb dispersion root solver, SAFE eigenvalue problem, FDTD elastodynamic grid, AIC picker, IIR filter synthesis, Levenberg-Marquardt TDOA solver) using standard Python and vectorized NumPy.
- **Consequence:** Extremely fast, fully deterministic, CPU-portable everywhere without binary dependency issues.

## ADR-003: Reproducibility and Canonical Config Hashing
- **Status:** Approved
- **Context:** PRD requirement SIM-01 and Section 8.2 mandate that every simulated event and dataset is bit-reproducible from seed, code version, and config hash.
- **Decision:** Implement canonical JSON serialization with sorted keys and SHA-256 hashing in `acoustisolar_schemas.hashing`.
- **Consequence:** Ensures identical config hashes across systems and platforms.

