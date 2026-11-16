# AcoustiSolar Codebase Evaluation Against the Literature

**Date:** 2026-11-15 · **Commit evaluated:** `cabbab0` (working tree has an uncommitted change to `validation/VV_REPORT.md`)
**Basis:** [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) (reference numbers `[R#]` point there), a full read of `packages/`, `validation/`, `demo/` and `tests/`, and four small experiments run against the code (§7).

Every finding is tagged with how sure it is:
- **VERIFIED**: reproduced by running code; numbers in §7.
- **CODE**: visible directly in the source; line references given.
- **LIT**: a gap against published practice.

---

## 1. Verdict

The **architecture is right** and matches the literature closely. The PRD (§9) already asks for most of what the research recommends: S0/A0 energy ratio, scalogram CNNs, domain adaptation, novelty detection, calibrated probabilities, and module-level splits. The monorepo layout, config hashing, Pydantic schemas, assumption tiers and the honest KPI table in the committed V&V report are strong foundations.

The **implementation lags the design**, and some reported results are not trustworthy:

- **Physics:** one confirmed solver bug. The simulator also leaves out three effects that matter for this problem: frame reflections, source-dependent wave-mode content, and how the PZT senses each mode.
- **Localization:** the current error tail is caused by **wave-mode confusion in the cross-correlation**, not by the solver (verified).
- **ML:** the deep model has no training code; the task is binary while the goal is to name the source; and the "held-out modules" barely differ physically from the training modules.
- **V&V:** parts of the automated suite report PASS without testing anything, and running it overwrote the detailed committed report.

Fix integrity first (§2), then physics fidelity (§3). Everything downstream depends on those two.

---

## 2. P0: Correctness and integrity (fix before any new feature)

### 2.1 SAFE multilayer solver has a sign error that wrecks its dispersion curves. **VERIFIED**
- **Where:** [safe.py:99-104](../../packages/physics/src/acoustisolar_physics/safe.py#L99-L104) and [safe.py:120-121](../../packages/physics/src/acoustisolar_physics/safe.py#L120-L121).
- **Cause:** the cross-coupling entries `[1,2]` and `[3,0]` of the element matrix have the wrong sign. The forced symmetrization `0.5*(K + K^H)` then cancels the coupling between neighbouring nodes instead of exposing the error.
- **Evidence** (single 3.2 mm glass plate, where the exact answer is known):

| k (rad/m) | A0 thin-plate theory | A0 repo | A0 corrected | S0 plate limit | S0 repo | S0 corrected |
|---|---|---|---|---|---|---|
| 50 | 254 m/s | **3416** | 253 | 5501 | **5730** | 5501 |
| 150 | 762 | **3416** | 736 | 5501 | **5730** | 5497 |

  The repo's S0 is faster than the plate-wave limit, which is physically impossible.
- **Full 5-layer stack:** the corrected solver gives S0 = 5324 m/s, about 3% below bare glass, as expected from added mass. The committed V&V report's "−40.43% velocity reduction … interfacial flexural-shear mode" is an artefact of this bug and should be withdrawn.
- **Second issue:** `compute_dispersion` labels modes by sort order ("mode 1 = S0"). In the laminate the second root is a soft-layer mode (about 2927 m/s). Classify modes by polarization (in-plane vs out-of-plane energy), not by order.
- **Fix:** correct the two signs, delete the forced symmetrization, and add `assert np.allclose(ke, ke.conj().T)`. Add a regression test: single-layer SAFE must match `LambDispersionSolver` to within 1%. (The corrected matrix is in `safe_check.py` in the session scratchpad.)

### 2.2 The automated V&V suite reports PASS for tests that test nothing. **CODE**
In [run_vv_suite.py](../../validation/run_vv_suite.py):
- **Multi-fidelity gap:** compares `np.random.normal` noise with itself plus 5% noise, never Tier-0 against Tier-1 ([lines 82-86](../../validation/run_vv_suite.py#L82-L86)). The "0.00% arrival error, 0.999 correlation" in the report is meaningless.
- **Reciprocity and grid convergence:** both are hard-coded `"status": "PASS"` ([lines 58-61](../../validation/run_vv_suite.py#L58-L61), [76-80](../../validation/run_vv_suite.py#L76-L80)).
- **ECE target:** an ECE of 0.291 against a target of ≤0.08 still yields `ALL_PASS`, because the ML block has no `status`.
- **Timestamp:** hard-coded to `"2026-11-15T01:30:00Z"`.
- **Overwrites the report:** the script writes `validation/VV_REPORT.md` unconditionally. That is exactly what the uncommitted diff shows: the curated 199-line report, with its deviation table and honest KPI failures, was replaced by a 27-line summary.
- **Fix:**
  - Restore the committed report.
  - Make the suite write to `validation/out/` instead.
  - Make every block assert a real computed quantity (the real versions already exist in [verify_physics.py](../../validation/verify_physics.py)).
  - Fail CI when a target is missed, or mark the block `KNOWN_GAP` explicitly.

### 2.3 The MLflow logger records hard-coded numbers. **CODE**
[log_mlflow_run.py](../../validation/log_mlflow_run.py) writes literal values (`0.9439`, `0.5608`, `222.77` …) as metrics, both as a fallback and in the `metrics` block. Experiment tracking must log computed values only; read them from `benchmark_summary.json` or fail.

### 2.4 The "1D-ResNet" has no training code and no residual connection. **CODE**
- [deep_model.py](../../packages/ml/src/acoustisolar_ml/deep_model.py) has only a forward pass with random weights, and no skip connection despite the "Residual Block" comment.
- The benchmark reports its F1 of 0.7507 as a model result.
- The demo labels predictions "LightGBM + 1D-ResNet Ensemble" ([server.py:218](../../demo/server.py#L218)).
- When the LightGBM pickle is missing, the demo **fakes predictions from the ground truth** ([server.py:179-182](../../demo/server.py#L179-L182)).
- **Fix:** implement a real trainable model (PyTorch, as the PRD specifies), or remove it from reports and the demo. Make the demo show "model unavailable" instead of echoing the label.

### 2.5 The Tier-1 localization result is explained wrongly. **CODE**
The benchmark attributes the 222.77 mm Tier-1 error to "multipath coda reflections from plate edges". The actual causes:
- **Sensors inside the absorber:** the sensors (60 mm from the edge) sit **inside the 10-cell × 10 mm = 100 mm sponge layer** ([run_scale_kpi_benchmark.py:266-271](../../validation/run_scale_kpi_benchmark.py#L266-L271), [tier1.py:48-58](../../packages/physics/src/acoustisolar_physics/tier1.py#L48-L58)), so the signals are damped and distorted where they are recorded.
- **No edge reflections:** the sponge *removes* them (2.37% reflection), the opposite of a framed module.
- **Under-resolved grid:** at dx = 10 mm and 200 kHz, the grid has only about 2.65 points per S0 wavelength (≥10 is normal).
- **Not elastodynamic:** "Tier-1 elastodynamic" is a scalar membrane wave equation with one speed: no A0, no dispersion.
- **Fix:** rename it a scalar wave solver. Put sensors outside the absorber, or better, use reflecting edges to represent the frame (§3.1). Increase resolution to ≥10 points per wavelength. Re-derive the conclusion.

### 2.6 The "zero-leakage held-out modules" barely differ from the training modules. **CODE**
- **Tier-0 ignores most of the module config:**
  - It takes only `stack[0].thickness_mm` and hard-codes `GLASS_MATERIAL` ([tier0.py:36-38](../../packages/physics/src/acoustisolar_physics/tier0.py#L36-L38)).
  - It ignores layer E/ν, EVA properties, `LayerConfig.temperature_c`, and module width/length (no boundaries).
  - So MOD-5 "aged, 65 °C, high damping" is simulated with the same physics as MOD-1, and its described α ≈ 2.5 Np/m is never applied.
- **Sensors are not at the corners of larger modules:** every module uses the default sensor positions for 1040×1760 mm, including the 1000×2000 and 1134×2278 mm modules. Sources are then drawn up to about 2200 mm, outside the sensor array, which inflates localization error for geometric reasons alone.
- **Cell-ID accuracy is defined wrongly:** the benchmark counts `err < 60 mm` as a match ([run_scale_kpi_benchmark.py:236-237](../../validation/run_scale_kpi_benchmark.py#L236-L237)) instead of mapping both positions to cell indices.
- **Fix:**
  - Have Tier-0 consume the whole `ModuleTwinConfig`.
  - Derive sensor positions from module size.
  - Compute cell-ID from the layout.
  - Restate the generalization claims once these are fixed.

### 2.7 Some cited sources in ASSUMPTIONS.md are wrong. **LIT**
My literature search found **no peer-reviewed AE spectral data for PV modules**, so any parameter claiming a PV-specific AE source deserves scrutiny:
- **"ISO 12716 Acoustic Emission Testing of PV Laminates":** ISO 12716 is the general AE **vocabulary** standard. It is cited for friction and thermal-noise spectra.
- **"Köntges et al. (2014), IEA-PVPS Task 13 … four-point bend testing" for a 180–320 kHz crack frequency centroid:** that review covers failure modes, not AE spectra.
- **ASTM E1106 (sensor calibration) and ASTM E976 (sensor response reproducibility)** are cited for crack rise time and friction duration. They don't contain those values.

**Fix:** re-tag these as `[UNSOURCED]` / `[CALIBRATE]`. This matters because the source library's frequency bands define the classes the ML learns.

### 2.8 Small but real. **CODE**
- **CI never runs on this branch:** it triggers on `main`/`develop`, but the repo's branch is `master` ([ci.yml:4-7](../../.github/workflows/ci.yml#L4-L7)).
- **Undeclared dependencies:** `scipy`, `lightgbm`, `scikit-learn`, `pyarrow` and `fastapi` are used but missing from `pyproject.toml`.
- **ECE binning:** probability = 1.0 falls in no bin ([metrics.py:27](../../packages/ml/src/acoustisolar_ml/metrics.py#L27)).
- **Unsafe model loading:** the demo `pickle.load`s a model file. Prefer `joblib` with a hash check, or LightGBM's native text format.

---

## 3. P1: Physics fidelity gaps that the literature says matter

| # | Gap | Where | Why it matters (literature) | Improvement |
|---|---|---|---|---|
| 3.1 | **No frame or edge reflections in Tier-0.** Open infinite plate | [tier0.py](../../packages/physics/src/acoustisolar_physics/tier0.py) | Edge reflections dominate the later part of real plate AE signals. They break naive cross-correlation, but also carry location information that single-sensor deep learning exploits [R34–R36]. The repo's own measurement found 0.25 full-window correlation between Tier-0 and Tier-1 | Add image-source reflections from the four frame edges (cheap in the frequency domain). Randomize the reflection coefficient to stand in for frame and clamp coupling |
| 3.2 | **Fixed 75/25 S0/A0 split for every source**, noise included | [tier0.py:94-96](../../packages/physics/src/acoustisolar_physics/tier0.py#L94-L96) | Hail, rain and taps are out-of-plane impacts and excite mostly A0; in-plane cleavage excites S0. The S0/A0 ratio is one of the strongest physical cues for telling hail from cracks, and PRD §9.2 already asks for it | Make modal weights depend on the source: in-plane dipole for cracks, normal force for impacts and taps. Add an S0/A0 energy-ratio feature |
| 3.3 | **PZT sensor treated as mode-blind.** No out-of-plane vs in-plane sensitivity, and no aperture effect even though the docstring claims one | [pzt_sensor.py](../../packages/daqtwin/src/acoustisolar_daqtwin/pzt_sensor.py) | Surface PZT discs respond mainly to out-of-plane motion, and S0 has little out-of-plane displacement at low frequency-thickness. Real data will therefore look more A0-heavy than the simulation, which worsens the mode confusion in §4.1. A 10 mm disc is comparable to the A0 wavelength around 200 kHz, so aperture filtering matters | Weight each mode by its surface out-of-plane displacement (from the corrected SAFE). Add a Bessel aperture filter `2J1(ka)/(ka)` |
| 3.4 | **Same attenuation for both modes**, with a fitted constant (`* 300.0`, clipped to 5 Np/m) | [viscoelastic_eva.py:69-71](../../packages/physics/src/acoustisolar_physics/viscoelastic_eva.py#L69-L71) | In laminated glass the polymer interlayer damps antisymmetric modes more than symmetric ones [R37] | Compute attenuation per mode from the imaginary part of the corrected SAFE solution with complex EVA modulus. Keep it `[CALIBRATE]` against pitch-catch data |
| 3.5 | **Arrhenius temperature shift for EVA** near its glass transition (−35 °C) | same file | Near Tg, polymer temperature–frequency shifts follow WLF-type behaviour; Arrhenius underestimates the change in the −20 to 0 °C range common in winter | Use a WLF shift below roughly Tg+50 °C. Randomize Tg and the relaxation frequency in the training data |
| 3.6 | **Tier-0 uses bare-glass dispersion**, not the laminate | [tier0.py:37](../../packages/physics/src/acoustisolar_physics/tier0.py#L37) | The laminate shifts S0 by about 3% (corrected SAFE), enough to bias velocity-based TDOA by about 1–3% of distance | After fixing §2.1, interpolate Tier-0 phase velocities from SAFE for the configured stack |
| 3.7 | **Simulation and filters are unrealistic in time.** Circular FFT with no zero-padding (slow low-frequency A0 wraps around the 2.56 ms window); the source fires at sample 0 so there is no pre-trigger noise; the DAQ and DSP filters are zero-phase (acausal) | [tier0.py:122](../../packages/physics/src/acoustisolar_physics/tier0.py#L122), [analog_chain.py](../../packages/daqtwin/src/acoustisolar_daqtwin/analog_chain.py), [filter.py](../../packages/dsp/src/acoustisolar_dsp/filter.py) | Acausal filters create energy *before* the true arrival. Onset pickers and models trained on that will not transfer to real causal analog hardware. Without a pre-trigger segment, AIC and STA/LTA cannot estimate noise (it also broke my AIC test in §7) | Zero-pad 2×. Place the source at a random time after the pre-trigger samples. Model the analog chain as a causal IIR (Butterworth via `scipy.signal`, with per-channel tolerance) |
| 3.8 | **No ambient noise environment.** Electronic noise only; one isolated event per window | [source_library.py](../../packages/physics/src/acoustisolar_physics/source_library.py) | Field AE noise includes friction, rain, wind and dust [R44]. ASSUMPTIONS.md itself lists inverter IGBT harmonics at 16–32 kHz. False-alarm rate is the hardest KPI and is set by this environment | Add continuous background noise: wind-band noise, Poisson rain-drop trains, inverter switching harmonics. Add **overlapping events**, especially hail impact followed by a crack burst within milliseconds |
| 3.9 | **Rain and hail are one class.** Crack friction is labelled `is_crack=1` | [source.py](../../packages/schemas/src/acoustisolar_schemas/source.py), [generator.py:114](../../packages/ml/src/acoustisolar_ml/generator.py#L114) | Rain (dense trains of weak impacts) and hail (one large impact that can *cause* cracks) behave very differently. Friction comes from an existing crack: useful for health trending, but not a new fracture event, and easily confused with wind rattle | Split `RAIN` and `HAIL`. Add a `HAIL+CRACK` compound label. Make "new fracture" vs "existing-crack activity" an explicit two-level label |

---

## 4. P1: Localization

### 4.1 The current error tail comes from wave-mode confusion in the cross-correlation. **VERIFIED**
On 120 Tier-0 crack events (default 120-cell module):
- **Overall:** median 19.3 mm, P90 196.9 mm; 69% of events within 30 mm.
- **Solver is not the problem:** fed the *true* time differences, `solve_hyperbolic` returns 0.0 mm. Multi-start grid initialization changed nothing, so local minima are not the cause.
- **Time differences are:** the worst 15% of events have errors of 25–180 µs, while good events sit around 5 µs. The cross-correlation in [tdoa.py:40-63](../../packages/dsp/src/acoustisolar_dsp/tdoa.py#L40-L63) runs over the full window, so it locks onto the large, slow, dispersive A0 packet. The solver then converts that delay into distance using the S0 speed (5300 m/s). A 100 µs error × 5.3 mm/µs ≈ 530 mm.
- **Expect worse on real panels:** real PZTs see A0 more strongly than the simulation does (§3.3).

### 4.2 Improvements, in order
1. **Gate the first arrival before correlating.** Correlate only a short window (about 20–40 µs) after each channel's first threshold crossing, which holds the S0 arrival, and band-limit it to the less dispersive part of S0.
2. **Physical consistency checks.** Reject or flag any time difference larger than the sensor separation divided by the speed. Use **all sensor pairs** (6 for 4 sensors) with a robust loss (Huber or RANSAC) instead of only the differences against sensor 0.
3. **Estimate the wave speed with the position** (solve for x, y, v), or use a temperature-compensated S0 speed from SAFE plus a temperature sensor.
4. **AIC-Delta-T mapping as a calibrated baseline** [R32]. Calibrate a grid of arrival-time differences from pencil-lead breaks on the real module; it needs no speed model. The literature reports 32.6 → 3 mm on complex parts. This is also the natural bridge to real hardware.
5. **AIC picking only after §3.7.** My quick AIC + grid-search test did *worse* (68 mm median), mainly because the Tier-0 window has no noise-only pre-trigger region. Vectorize the AIC first: [pickers.py:35-41](../../packages/dsp/src/acoustisolar_dsp/pickers.py#L35-L41) is O(N²) in Python, and a cumulative-sum version is O(N) (in `loc_exp.py` in the scratchpad).
6. **Make STA/LTA causal.** `np.convolve(mode='same')` uses future samples ([pickers.py:64-65](../../packages/dsp/src/acoustisolar_dsp/pickers.py#L64-L65)); use trailing windows so it can run in real time on the edge.
7. **Later:** a learned residual correction on top of classical TDOA (PRD §9.3(c)), and single-sensor or edge-reflection deep learning [R35, R36] once reflections exist in the simulator (§3.1).
8. **Report uncertainty honestly.** With 4 sensors, the covariance has only 1 degree of freedom ([tdoa.py:135](../../packages/dsp/src/acoustisolar_dsp/tdoa.py#L135)). Propagate time-difference uncertainty from the measured noise and SNR instead.

---

## 5. P1: ML pipeline

| # | Gap | Improvement (with literature anchor) |
|---|---|---|
| 5.1 | **Binary crack-vs-noise only**, but PRD §9.1 and the project goal call for naming the source (crack, hail, rain, wind, thermal, EMI, handling, unknown) | Multi-class or multi-label head, with the compound `HAIL+CRACK` class. The two-branch CNN that separates noise, impact and crack growth in rails is the closest published template [R41] |
| 5.2 | **Features are thin.** 7 per channel. The "partial power bands" in the docstring are not implemented. Nothing in time-frequency, no mode features, no context | Add: RA value and average frequency (standard AE crack-mode indicators); partial power in 4–6 bands; S0/A0 energy ratio; CWT or synchrosqueezed scalograms for a 2D CNN [R40, R41]; **distance-corrected source energy** using the estimated location and attenuation model; **context features** (location relative to frame or clamps, dT/dt, wind, rain flag). The `EventRecord` schema already has `temperature_c`, `wind_ms` and `rain_flag`, but nothing uses them |
| 5.3 | **Features depend on channel order** (concatenated per sensor index), so the model learns sensor-specific patterns tied to position | Per-sensor encoder plus permutation-invariant pooling (the PRD's set-transformer / GNN idea), or at least sort channels by arrival order |
| 5.4 | **No real deep model** (§2.4); the baseline booster uses depth-1 stumps with 9 thresholds | PyTorch scalogram CNN plus a 1D waveform branch. Keep LightGBM on engineered features as the fallback that can run on the edge node |
| 5.5 | **No handling of unseen noise.** Real field noise will include things the simulator never produced | Out-of-distribution / open-set head (Mahalanobis distance on embeddings or energy score) that outputs "unknown" instead of forcing a class (PRD §9.3 already names it) |
| 5.6 | **No probability calibration**, while measured ECE is 0.29 on the small held-out run | Temperature scaling on a calibration split, plus conformal prediction sets for alerts (PRD §9.4.5) |
| 5.7 | **Narrow domain randomization:** only temperature, position, energy, rise time and sensor gain | Also randomize EVA modulus, Tg and loss; glass thickness and E; sensor position jitter and bond quality; edge reflection coefficients; background-noise level and type; the trigger threshold. Required for any chance of transfer to real data [R45–R49] |
| 5.8 | **Sim-to-real is not designed in** | Design the pipeline so pretraining on synthetic data, fine-tuning on a small pencil-lead break / LAB-1 set, and domain adaptation are drop-in steps. The literature reports about 20% gains from domain adaptation on imbalanced simulated-to-real AE data [R46] |
| 5.9 | **False-alarm estimate is not meaningful.** "Recall at 1 false alarm per module-day" uses the 99.99th percentile of about 3,200 noise events, which is effectively the sample maximum. The "10,000 triggers/day" figure is unsourced | Estimate the noise score tail with ≥10⁵ noise events or an extreme-value fit. Report false-alarm rate against threshold as a curve. Replace 10,000/day with field logging data once available (ASSUMPTIONS §4A-3) |
| 5.10 | **Severity is not linked to impact** | Add a severity output tied to the literature thresholds (Köntges: <8% inactive cell area → no power loss; 12–50% → roughly linear loss [R1]). Calibrate it later against EL-measured inactive area (PRD §14.1 severity ρ ≥ 0.7) |
| 5.11 | **Edge deployment not started** | TinyML AE classifiers already reach >99% on microcontroller-class hardware in published work [R50]. Plan an int8-quantized model and firmware-in-the-loop test (PRD §10.3) |

**Tested and cleared:** I checked whether the synthetic classes leak through one feature. Best single-feature AUC on 400 events: peak frequency 0.72, counts 0.70, centroid 0.62. So the current task is not trivially separable. The bigger risk is that the hand-designed source waveforms (§2.7) are unvalidated, not that they leak the label.

---

## 6. P2: Validation and real-data roadmap

The literature has **no real PV AE dataset**, so AcoustiSolar's credibility rests on creating one (gaps G1–G7 in the review):

1. **Sensor and propagation calibration (weeks).**
   - Hsu-Nielsen pencil-lead breaks (ASTM E976) on a grid over a real framed module, in a thermal chamber from −20 to 75 °C.
   - Outputs: S0/A0 speeds, per-mode attenuation, reflection behaviour, sensor gain spread.
   - This retires the `[CALIBRATE]` items and gives the Delta-T calibration data.
2. **LAB-1 fracture dataset with EL ground truth.**
   - Controlled cracking: four-point bend on cells, static load per IEC 61215, dynamic load per IEC TS 62782.
   - EL image before and after every load step.
   - This links AE energy and count to cracked or inactive area (gap G5).
3. **NOISE-1 dataset.** IEC 61215 ice-ball hail on a sacrificial module, rain simulator, wind shaker with clamps, thermal cycling chamber, inverter EMI. Include hail that *does* crack cells (compound events).
4. **Sim-to-real benchmark.** Report every KPI three ways: synthetic → synthetic, synthetic → LAB-1, and fine-tuned → LAB-1. Only the last two support product claims.
5. **Keep the honest framing** from the committed V&V report. Present AE as *complementary* to an EL baseline: it detects new fracture events, but dormant pre-existing cracks may never emit [R5].

---

## 7. Experiments run for this evaluation

All scripts are in the session scratchpad (`C:\Users\Suriy\AppData\Local\Temp\claude\e1fbeb8f5ee7c679\51357b2b-642e-4487-9498-6e266eb9970d\scratchpad\`). The repo was not modified.

| Script | What it showed |
|---|---|
| inline SAFE check (`safe_check.py` has the corrected matrix) | Repo SAFE: A0 3416 m/s vs 254–762 m/s theory; S0 5730 m/s, above the 5501 m/s limit. Corrected: 253 / 736 / 5501 / 5497 m/s. 5-layer stack S0 corrected to 5324 m/s |
| `loc_exp.py` | Repo cross-correlation TDOA: median 19.3 mm, P90 196.9 mm. Multi-start: identical. Quick AIC + grid search: median 68.4 mm (worse; see §4.2-5) |
| `tail_diag.py` | Solver with true time differences: 0.0 mm. Tail events: time-difference errors of 25–180 µs; good events about 5 µs, which points to mode confusion |
| `leak.py` | Single-feature AUC ≤ 0.72, so no trivial label leakage |

---

## 8. Suggested order of work

| Step | Items | Size |
|---|---|---|
| 1. Trustworthy baseline | §2.1 SAFE fix and regression test; §2.2 real V&V checks and stop overwriting the report; §2.3 MLflow logs computed values; §2.4 remove or implement the deep model and stop faking demo output; §2.8 CI branch and dependencies | about 2–3 days |
| 2. Honest benchmark | §2.5 Tier-1 setup; §2.6 Tier-0 uses the full module config, sensors scale with module size, real cell-ID metric; §2.7 re-tag the mis-cited sources; re-run and republish KPIs | about 1 week |
| 3. Physics that matters | §3.1 edge reflections; §3.2/3.3 source-dependent modes and mode-aware PZT; §3.4/3.6 laminate dispersion and attenuation from SAFE; §3.7 causal filters, pre-trigger, zero-padding; §3.8 background noise and overlapping events | about 2–3 weeks |
| 4. Localization | §4.2 items 1–3 (gated cross-correlation, consistency checks, speed estimation), then Delta-T mapping | about 1 week |
| 5. ML | §5.1–5.6 multi-class with compound events, richer features, permutation invariance, real CNN, unknown-source rejection, calibration; §5.9 proper false-alarm estimation | about 3–4 weeks |
| 6. Real data | §6 calibration, then LAB-1, NOISE-1 and the sim-to-real benchmark | quarter-scale; drives everything else |

---

## 9. Implementation status (2026-11-16)

Steps 1–5 of §8 were implemented in code; step 6 (real-module calibration, LAB-1, NOISE-1)
needs physical hardware and remains open.  New bugs found during implementation are marked
**[NEW]**.

| Item | Status | Where |
|---|---|---|
| §2.1 SAFE sign error; mode identification by energy location, not frequency order | Done; single-layer SAFE matches exact Lamb within 0.08 % | `physics/safe.py`, `tests/test_physics_fidelity.py` |
| §2.2 V&V suite tests nothing / overwrites report | Done; 7 real checks + ML known-gap reporting, output to `validation/out/` | `validation/run_vv_suite.py` |
| §2.3 MLflow logs constants | Done; reads computed files, fails if missing | `validation/log_mlflow_run.py` |
| §2.4 Untrained "ResNet", demo fakes predictions | Done; trainable PyTorch residual CNN; demo refuses unverified models and never echoes ground truth | `ml/deep_model.py`, `demo/server.py` |
| §2.5 Tier-1 mislabelled, sensors in sponge | Done; documented as scalar solver, sensors in sponge rejected | `physics/tier1.py` |
| **[NEW]** Tier-1 sponge reflected ~74 % (damping 10× too weak; old "2.37 %" claim wrong) | Fixed (now ~11 % for a 1.4-wavelength layer) | `physics/tier1.py` |
| **[NEW]** Tier-1 first-order errors: cell geometry offset by dx/2, sensor sample recorded one step late | Fixed; observed convergence order 1.93 | `physics/tier1.py` |
| **[NEW]** Tier-1 "heterogeneous reciprocity" test never applied its heterogeneity | Fixed; self-adjoint flux form, reciprocity error ~1e-15 with a real inclusion | `physics/tier1.py`, `validation/verify_physics.py` |
| §2.6 Held-out modules not physically different; sensors not at module corners; wrong cell-ID metric | Done; Tier-0 consumes the full stack; `SensorArrayConfig.for_module`; layout-based cell IDs | `physics/tier0.py`, `schemas/sensor.py`, `ml/generator.py` |
| §2.7 Mis-cited sources | Done; re-tagged `[UNSOURCED]`/`[CALIBRATE]` | `ASSUMPTIONS.md` |
| §2.8 CI branch, dependencies, ECE bin, pickle | Done; CI on `master`, deps declared, ECE fixed, SHA-256-verified model loading | `.github/workflows/ci.yml`, `pyproject.toml`, `ml/metrics.py` |
| **[NEW]** 99 % of simulated events saturated the ADC (amplitude features carried no information) | Fixed; absolute level anchored at ~70 dB_AE with a ~30 dB source-level spread | `physics/tier0.py` |
| §3.1 Frame reflections | Done (image sources, randomised reflection coefficient) | `physics/tier0.py` |
| §3.2/3.3 Source-dependent modes, PZT out-of-plane coupling, aperture | Done; crack S0/A0 ≈ 0.84 vs hail ≈ 0.004 at 150 kHz | `physics/laminate.py`, `physics/tier0.py` |
| §3.4/3.6 Mode-dependent attenuation and laminate dispersion | Done (modal strain energy, shear-only polymer loss) | `physics/laminate.py` |
| **[NEW]** Laminate tables had non-monotonic frequency after smoothing → `np.interp` silently returned S0 attenuation ~6× too high | Fixed (re-sorted); S0 ≈ 2.5 Np/m at 150 kHz, 25 °C | `physics/laminate.py` |
| **[NEW]** Tabulated dispersion had negative group slowness at mode hybridisation points (energy arriving before the source fired) | Fixed (causal, bounded group velocity) + minimum-phase spectral shaping + causality mute | `physics/laminate.py`, `physics/tier0.py` |
| §3.5 WLF near T_g | Done (hybrid WLF/Arrhenius) | `physics/viscoelastic_eva.py` |
| §3.7 Wrap-around, pre-trigger, causal filters | Done (zero-padding, onset placement, causal Butterworth with tolerance) | `physics/tier0.py`, `daqtwin/analog_chain.py` |
| §3.8 Ambient noise, overlapping events | Done (1/f + inverter harmonics; hail-then-crack compound events; multi-hit trigger with dead time; floating threshold) | `physics/source_library.py`, `daqtwin/trigger.py` |
| §3.9 Rain vs hail, friction label | Done (10 classes; `is_crack` = new fracture) | `schemas/event.py`, ADR-006 |
| §4.2 Gated TDOA, consistency checks, robust all-sensor solve, Delta-T, vectorised AIC, causal STA/LTA | Done | `dsp/tdoa.py`, `dsp/pickers.py` |
| §4.2 Mode-aware ({S0, A0} per channel) solving | Implemented as an opt-in option; **did not improve** the benchmark (with 4 sensors each assignment leaves 1 degree of freedom), so SNR gating stays the default | `dsp/tdoa.py` |
| §5.1–5.6 Multi-class, features, permutation invariance, CNN, unknown-source detection, calibration | Done | `ml/*`, `dsp/features.py` |
| §5.9 False-alarm estimation | Done (GPD tail fit; trigger-rate assumption still `[UNSOURCED]`) | `ml/metrics.py` |
| §6 Real-data roadmap | **Open** — requires hardware | — |

### 9.1 Benchmark after the revision (simulation only; 2,500 events, train MOD-1..3 / test MOD-4..5)

| KPI | Target | Result |
|---|---|---|
| Physics V&V checks | all pass | 7/7 PASS |
| Macro-F1 (10 classes) | ≥ 0.95 | 0.53 features · 0.55 CNN · 0.60 ensemble |
| New-fracture PR-AUC | — | 0.91 |
| Recall @ ≤ 1 FA/module-day | ≥ 0.98 | 0.00 (target FPR not reachable) |
| Localisation median / P90 | ≤ 15 / ≤ 40 mm | 35 / 644 mm (glass-backsheet MOD-5: 13 mm median; glass-glass MOD-4: 198 mm) |
| Cell-ID accuracy | ≥ 0.95 | 0.51 |
| Tier-1 localisation (correct setup) | — | 5.8 mm median |

Most important open items: (1) glass-glass laminates need a symmetric-waveguide mode model;
(2) impact-type classes (hail, tap, hail-crack) remain confusable; (3) conformal coverage
drops to 0.68 under module shift; (4) everything awaits real-module data (§6).
