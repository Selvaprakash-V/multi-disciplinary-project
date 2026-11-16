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


## ADR-004: SciPy as a Core Dependency
- **Status:** Approved (2026-11)
- **Context:** ADR-002 aimed for pure NumPy, but correctness now needs a Hermitian generalized eigen-solver (SAFE), causal IIR filtering (analog front end), Bessel functions (PZT aperture) and a Generalised-Pareto fit (false-alarm tail).  Re-implementing these would be slower and riskier than depending on SciPy, which was already imported by the SAFE solver.
- **Decision:** Declare `scipy` as a core dependency; keep heavy ML libraries (`scikit-learn`, `torch`) optional extras.
- **Consequence:** ADR-002's "pure NumPy" goal is relaxed for numerics; CI installs SciPy.

## ADR-005: Mode-Resolved Laminate Physics in Tier-0
- **Status:** Approved (2026-11)
- **Context:** Tier-0 used bare-glass dispersion, a fixed 75/25 S0/A0 split for every source, equal attenuation for both modes and no frame boundaries.  The literature identifies modal content (S0 vs A0) as a primary discriminator between in-plane fracture and out-of-plane impacts, and edge reflections as a major factor for plate AE.
- **Decision:** Tier-0 builds a `LaminateDispersionModel` from the configured stack (corrected SAFE solver, modal strain-energy attenuation with shear-only polymer loss, hybrid WLF/Arrhenius EVA shift), uses source-dependent modal excitation and out-of-plane receiver coupling with disc aperture, adds image-source frame reflections, minimum-phase (causal) spectral shaping and a causality mute.
- **Consequence:** ~40 ms per event per module variant (cached laminate build ~5 s).  All new parameters are `[CALIBRATE]`; see ASSUMPTIONS.md §5.

## ADR-006: Multi-Class Labels and the Meaning of `is_crack`
- **Status:** Approved (2026-11)
- **Context:** The PRD asks for source classification (crack, hail, rain, wind, thermal, EMI, handling).  Rain and hail behave differently, hail can itself crack cells, and crack *friction* is activity of an existing crack rather than new fracture.
- **Decision:** `event_class` ∈ {crack_initiation, crack_growth, crack_friction, hail_crack, hail, rain, wind_rattle, thermal_tick, emi_burst, handling_tap}.  `is_crack` = new fracture (initiation, growth, hail_crack); `is_crack_activity` additionally includes friction.
- **Consequence:** Binary metrics are not comparable with pre-2026-11 reports, which counted friction as crack.

## ADR-007: Honest Verification Outputs
- **Status:** Approved (2026-11)
- **Context:** The previous V&V suite reported PASS for checks that tested nothing, the MLflow logger recorded constants, the demo echoed ground truth when the model was missing and served hard-coded benchmark numbers.
- **Decision:** Every V&V check computes a real quantity; KPI shortfalls are reported as `KNOWN_GAP`; generated results go to `validation/out/` and `validation/kpi_benchmark_summary.json`; loggers and the demo read only computed files; model files are loaded only after SHA-256 verification.
- **Consequence:** Reported numbers are lower than before, but they are reproducible and traceable.
