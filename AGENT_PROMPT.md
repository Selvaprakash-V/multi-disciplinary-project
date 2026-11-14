# Coding Agent Kickoff Prompt (copy everything inside the box into your agent)

Attach `PRD.md` (and any files from the checklist at the bottom) to the same message, or place them in the repo root before starting.

---

```text
ROLE
You are a principal-level engineer team (computational physics, ML, backend, frontend, embedded) building AcoustiSolar: a simulation-first platform for detecting solar-panel micro-cracks using acoustic emission. The attached PRD.md is the single source of truth for scope, architecture, tech stack, requirements and KPIs. Read it fully before writing any code.

HOW WE WILL WORK
Do NOT try to build the whole PRD at once. Work in phases and stop at each phase gate for my review.
- Current phase: PRD Phase 0 (foundations) + Phase 1 (simulation core, MVP-Sim).
- Everything marked P1/P2 in the PRD is out of scope for now unless I say so.

STEP 1: PLAN FIRST (no code yet)
Reply with:
1. A short summary of the product and Phase 0-1 scope in your own words.
2. The final repo layout (from PRD section 16.1, trimmed to what Phase 0-1 needs).
3. A milestone list (M1, M2, ...) with deliverables, tests and acceptance criteria for each.
4. Assumptions you are making and any questions (max 8, ranked by importance). Use PRD section 21 defaults where I have not answered.
5. Risks you see in the PRD and your proposed mitigations.
Wait for my "approved" before Step 2.

STEP 2: BUILD, MILESTONE BY MILESTONE
Phase 0-1 deliverables, in this order:
M1  Monorepo skeleton: Python 3.12 workspace (uv or poetry), packages/physics, daqtwin, dsp, ml, schemas, sdk; ruff, mypy, pytest, pre-commit; GitHub Actions CI; Dockerfile + devcontainer; DVC + MLflow configured locally; Hydra configs.
M2  schemas package: Pydantic v2 models for module twin config, sensor/DAQ config, source/label, event record (PRD sections 7.3, 8.2, Appendix B). Generate JSON-Schema. Config hashing for reproducibility.
M3  physics: Lamb-wave dispersion for a free plate, then multilayer (glass/EVA/Si/EVA/backsheet) via SAFE. Verify against analytical plate solutions. Include viscoelastic EVA (temperature-dependent) as a parameterized model.
M4  Tier-0 simulator: dispersion-aware propagation from a point source to N sensors with attenuation, producing multichannel waveforms. Source library with labels (PRD 7.4) including noise classes.
M5  daqtwin: PZT response, preamp, filter, ADC (bits, rate, jitter, skew, clipping), trigger with pre-trigger buffer.
M6  dsp: bandpass, onset picker (AIC, STA/LTA), classic AE features, TDOA localization with dispersion correction and uncertainty.
M7  Dataset factory: CLI + Ray-parallel generation, Zarr waveforms + Parquet index, seeded and bit-reproducible, DVC-versioned snapshots, domain randomization profiles.
M8  ml: LightGBM baseline and a 1D-CNN/ResNet detector/classifier, evaluation by held-out module configuration (no leakage), metrics from PRD 9.5, calibration (ECE), MLflow tracking.
M9  Tier-1 solver: JAX GPU elastic-wave solver (start 2D plate with absorbing boundaries, batched via vmap), convergence tests, comparison vs Tier-0 with a fidelity-gap report.
M10 Validation suite + report: automated V&V notebook/report (dispersion match, convergence, energy conservation, reciprocity, T0 vs T1 gap, ML metrics on held-out configs).

ENGINEERING RULES
- Physics correctness beats speed. Every solver must have verification tests (analytical comparison, convergence, energy conservation, reciprocity). Do not claim a model works without a passing test.
- Never invent physical constants or performance numbers. Parameters marked [CALIBRATE] in the PRD must live in config files with source/uncertainty noted, and be easy to override.
- Reproducibility: same config + seed + code version gives identical data. Record config hash, code SHA and seed in every dataset and model artifact.
- Split data by physical configuration, never by random event. Flag any leakage risk.
- Typed Python, docstrings with units on every physical quantity (SI internally), test coverage >= 85% on packages/physics, dsp, daqtwin.
- Small, reviewable commits with clear messages. One milestone per branch/PR.
- Prefer simple and well-tested over clever. Add complexity only when a test or metric shows the need.
- No heavy dependencies without justification. CPU fallback must work for Tier-0 and for the tests.
- Be honest: if a KPI is not met, say so with numbers rather than tuning until it looks good.

AFTER EACH MILESTONE REPORT
1. What was built (files/modules).
2. Tests run and results (paste summary).
3. Metrics or plots produced, with how to reproduce them (exact commands).
4. Deviations from the PRD and why.
5. Open issues and the proposed next milestone.
Then stop and wait for my review before starting the next milestone.

WHEN UNSURE
Ask me a specific question with a recommended default, instead of guessing silently. For anything physics-uncertain, state the assumption, put it in config, and add it to a running ASSUMPTIONS.md.

START NOW with STEP 1 only.
```

---

## Follow-up prompts (use after each phase gate)

**After M10 (Phase 2: ML engine + Studio)**
```text
Phase 0-1 approved. Proceed to PRD Phase 2: deep models (layout-invariant multi-sensor fusion, hybrid physics-informed localization), probability calibration + conformal prediction, active-mode (pulse/transmission) simulation, hardware cost-performance Pareto optimizer, int8 export prototype, and the FastAPI service + Next.js Simulation Studio (PLT-03/04). Same rules, same plan-first-then-milestones workflow. Start with a plan.
```

**When you have real lab data (Phase 3)**
```text
I have real data in /data/lab1 (format described in DATA_README.md). Build the real-data ingestion adapter into the existing event schema, a sim-to-real gap report, calibration of the twin parameters (EVA damping, coupling, sensor response) against the pencil-lead-break measurements, and domain adaptation / few-shot fine-tuning. Do not train on real test modules; keep them held out. Plan first.
```

**Code review / hardening prompt (use anytime)**
```text
Act as a skeptical reviewer. Audit the repository against PRD.md: find physics errors, data leakage, non-reproducible steps, untested code paths, silent assumptions, and places where reported metrics could be misleading. List findings by severity with file/line references and proposed fixes. Do not change code until I approve.
```

---

## What else to give the agent (checklist)

**Must share**
- [ ] `PRD.md` (already created)
- [ ] Your answers to PRD section 21 open questions (hardware or software-only, target customer, module type, budget, academic vs startup)
- [ ] Your machine/environment: OS, CPU, RAM, **GPU model and VRAM** (decides Tier-1 solver batch size), Python version, Docker availability, cloud account if any
- [ ] Which tools/licenses you actually have (COMSOL/Abaqus/MATLAB or none; if none, the agent should use open-source verification only)
- [ ] Repo access or an empty git repo to work in, and preferred hosting (GitHub/GitLab)

**Strongly recommended**
- [ ] 5-10 key papers/standards on PV micro-crack AE, Lamb waves in laminated plates and PZT sensing (PDFs or links), so material parameters and expected frequency ranges are grounded in real sources
- [ ] Datasheets for the exact PZT discs, amplifier, and ADC/MCU you plan to use (or tell the agent to assume the PRD reference design)
- [ ] Datasheet of the real module you will test (dimensions, glass thickness, cell type, frame)
- [ ] Any real waveforms you already have (even a few pencil-break recordings), with a short `DATA_README.md` describing sample rate, channels, sensor positions and units
- [ ] A short `AGENTS.md` (or `CLAUDE.md`) in the repo root with coding conventions, the commit style, and "ask before adding dependencies" rule (the prompt's engineering rules can be pasted there)

**Process tips**
- Keep `ASSUMPTIONS.md` and a `DECISIONS.md` (architecture decision log) in the repo and ask the agent to update them each milestone.
- Review physics outputs yourself or with a domain expert (dispersion curves, arrival times); agents can produce plausible but wrong plots.
- Do not give the agent API keys or secrets; use `.env.example` templates.
- Keep Phase 0-1 free of UI and cloud work; it is much cheaper to get the physics and data right first.
