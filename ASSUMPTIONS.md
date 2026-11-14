# AcoustiSolar Engineering Assumptions Log

This document records physical parameters, defaults, and architectural assumptions used in Phase 0–1.

## 1. Physical Parameters and Material Defaults
- **Front Glass:**
  - Thickness: 3.2 mm (standard tempered solar glass).
  - Density $\rho$: $2500\text{ kg/m}^3$.
  - Young's Modulus $E$: $72.0\text{ GPa}$.
  - Poisson's Ratio $\nu$: $0.22$.
  - Bulk Longitudinal Wave Speed $c_L$: $\approx 5690\text{ m/s}$.
  - Bulk Shear Wave Speed $c_T$: $\approx 3436\text{ m/s}$.
  - Low-frequency fundamental symmetric Lamb wave speed $c_{S0}$: $\approx 5300\text{ m/s}$ ($2 c_T \sqrt{1 - c_T^2/c_L^2}$).
- **EVA Encapsulant:**
  - Front and back thickness: $0.45\text{ mm}$ each.
  - Density $\rho$: $960\text{ kg/m}^3$.
  - Glass transition temperature $T_g \approx -35^\circ\text{C}$.
  - Temperature-dependent viscoelastic shear modulus modeled via standard linear solid / Arrhenius shift:
    $G'(T) = G_\infty + \frac{G_0 - G_\infty}{1 + (f / f_0)^2 \exp(-E_a / R (1/T - 1/T_{ref}))}$.
  - Room temperature ($25^\circ\text{C}$) baseline shear modulus: $12.5\text{ MPa}$, loss tangent $\tan\delta \approx 0.15$.
- **Silicon Solar Cells:**
  - Thickness: $175\ \mu\text{m}$.
  - Density $\rho$: $2330\text{ kg/m}^3$.
  - Effective isotropic in-plane modulus: $E \approx 130\text{ GPa}$, $\nu \approx 0.28$.
- **Backsheet:**
  - Thickness: $0.30\text{ mm}$.
  - Density $\rho$: $1200\text{ kg/m}^3$, $E \approx 3.5\text{ GPa}$, $\nu \approx 0.35$.

## 2. Sensor and DAQ Defaults
- **Piezoelectric Sensors:**
  - 4 PZT circular discs bonded near the corners of the front glass (offset $60\text{ mm}$ inside active area).
  - Resonant peak at $\approx 200\text{ kHz}$ with effective $Q \approx 5.0$, directivity modeled as circular piston.
- **Electronics Chain:**
  - Preamp gain: $+40\text{ dB}$ ($100\times$).
  - Input voltage noise: $5\text{ nV}/\sqrt{\text{Hz}}$.
  - Bandpass filter: 4-pole Butterworth $50\text{ kHz} - 400\text{ kHz}$.
  - ADC: $12\text{ bits}$, $1.0\text{ MSPS}$ ($1\ \mu\text{s}$ sampling period), $3.3\text{ V}$ full-scale range ($0.806\text{ mV}/\text{count}$).
  - Hardware trigger: $20\text{ mV}$ threshold with $512$ pre-trigger samples and $2048$ post-trigger samples ($2560$ total samples, $2.56\text{ ms}$ window).

## 3. Scope Boundaries (Phase 0–1)
- Focus is on Tier-0 (analytical / Green's function frequency-domain ray tracer) and Tier-1 (2D elastodynamic finite-difference solver with absorbing boundaries).
- Monorepo packages: `schemas`, `physics`, `daqtwin`, `dsp`, `ml`, `sdk`.
- Held-out evaluation: ML models are evaluated strictly by splitting by module configuration geometry, eliminating data leakage across synthetic configurations.
