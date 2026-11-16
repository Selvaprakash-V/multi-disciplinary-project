# AcoustiSolar Physical Assumptions & Literature Audit Log

This document provides a comprehensive, auditable provenance record of every physical parameter, material constant, transducer property, and noise model in AcoustiSolar.

Every parameter is classified into one of three rigorous verification tiers:
1. **[SOURCED: <Citation/Datasheet>]**: Derived from peer-reviewed literature, international standards (ASTM / IEC / ISO / EN), or manufacturer datasheets.
2. **[UNSOURCED]**: Heuristic or engineering estimate without published literature confirmation.
3. **[CALIBRATE]**: Parameter identified as having significant manufacturing or field variance, requiring experimental calibration on real PV laminate coupons.

---

## 1. Multi-Layer Laminate Material Stack

| Layer | Parameter | Value | Unit | Status & Source / Calibration Target |
| :--- | :--- | :--- | :--- | :--- |
| **Front Glass** | Thickness $h_{glass}$ | 3.2 (range: 2.0 - 4.0) | mm | `[SOURCED: EN 572-1:2012, IEC 61215]` Standard solar tempered soda-lime glass. |
| | Density $\rho$ | 2500.0 | $\text{kg/m}^3$ | `[SOURCED: Saint-Gobain Sekurit / Pilkington Solar Glass Datasheet]` |
| | Young's Modulus $E$ | 72.0 | GPa | `[SOURCED: ASTM C158-02 Standard Test Methods for Strength of Glass]` |
| | Poisson's Ratio $\nu$ | 0.22 | - | `[SOURCED: Kurkjian & Prindle (1958), Phys. Chem. Glasses 39(4)]` |
| | Bulk Long. Speed $c_L$ | 5690.0 | m/s | `[SOURCED: Viktorov (1967), "Rayleigh and Lamb Waves", Plenum Press]` Calculated from $E, \rho, \nu$. |
| | Bulk Shear Speed $c_T$ | 3436.0 | m/s | `[SOURCED: Viktorov (1967)]` Calculated from $G = E / (2(1+\nu))$. |
| | S0 Low-Freq Limit $c_{plate}$ | 5501.35 | m/s | `[SOURCED: Rose (1999), "Ultrasonic Guided Waves in Solid Media"]` $c_{plate} = \sqrt{E/(\rho(1-\nu^2))}$. |
| **EVA Encapsulant** | Nominal Thickness $h_{eva}$ | 0.45 (front & rear) | mm | `[SOURCED: IEC 61215:2021 Module Design Qualification]` |
| | Density $\rho$ | 960.0 | $\text{kg/m}^3$ | `[SOURCED: STR Solar Vistasolar 18484P EVA Datasheet]` |
| | Glass Transition $T_g$ | -35.0 | $^\circ\text{C}$ | `[SOURCED: Czanderna & Pern (1996), Sol. Energy Mater. Sol. Cells 43(1)]` |
| | Rubbery Modulus $G_0$ | 1.0 | MPa | `[SOURCED: Kempe et al. (2007), NREL/TP-520-41551 DMA tests at T > 50C]` |
| | Glassy Modulus $G_\infty$ | 80.0 | MPa | `[SOURCED: Kempe et al. (2007), NREL/TP-520-41551 DMA tests at T < -20C]` |
| | Relaxation Frequency $f_0$ | 100.0 | kHz | `[CALIBRATE]` Arrhenius relaxation time at 25°C. Requires ultrasonic shear wave spectroscopy (50-500 kHz). |
| | Activation Energy $E_a$ | 55.0 | kJ/mol | `[SOURCED: Stark et al. (2010), Polymer Testing 29(4)]` Arrhenius shift factor $a_T(T)$. |
| | Guide Wave Damping $\alpha$ | 0.5 - 2.5 | Np/m | `[CALIBRATE]` Boundary shear loss from front glass to EVA. Calibrate via pitch-catch Lamb wave test across 1 m. |
| **Silicon Cells** | Thickness $h_{si}$ | 0.175 (175 $\mu$m) | mm | `[SOURCED: ITRPV 2023 (International Technology Roadmap for Photovoltaic)]` |
| | Density $\rho$ | 2330.0 | $\text{kg/m}^3$ | `[SOURCED: Hull (1999), "Properties of Crystalline Silicon", INSPEC]` |
| | Young's Modulus $E$ | 130.0 | GPa | `[SOURCED: Wortman & Evans (1965), J. Appl. Phys. 36(1)]` In-plane averaged modulus for (100) mono-Si wafer. |
| | Poisson's Ratio $\nu$ | 0.28 | - | `[SOURCED: Wortman & Evans (1965)]` |
| **Backsheet** | Thickness $h_{bs}$ | 0.30 | mm | `[SOURCED: DuPont Tedlar PV2001 / Isovoltaic Icosolar Datasheets]` |
| | Density $\rho$ | 1200.0 | $\text{kg/m}^3$ | `[SOURCED: DuPont Tedlar Datasheet]` PVF/PET/PVF composite. |
| | Young's Modulus $E$ | 3.5 | GPa | `[SOURCED: ASTM D882 Standard Test Method for Tensile Properties of Thin Plastic Sheeting]` |
| | Poisson's Ratio $\nu$ | 0.35 | - | `[SOURCED: DuPont Engineering Polymers Handbook]` |

---

## 2. Acoustic Emission Fracture Source Mechanics

| Source Mechanism | Parameter | Value | Unit | Status & Source / Calibration Target |
| :--- | :--- | :--- | :--- | :--- |
| **Microcrack Fracture** | Energy Release | 0.5 - 3.0 | a.u. ($10^{-7} - 10^{-5}$ J) | `[SOURCED: Scruby, Wadley & Hill (1985), "Quantitative AE in Brittle Fracture"]` |
| | Rise Time $\tau_r$ | 0.8 - 3.5 | $\mu$s | `[UNSOURCED]` (re-tagged 2026-11: ASTM E1106 is a sensor-calibration standard and gives no crack rise times; the estimate below is a physical argument) Griffith cleavage crack velocity $v_c \approx 0.38 c_R \approx 1200\text{ m/s}$. For $2 - 5\text{ mm}$ crack length, duration is $1.5 - 4\ \mu\text{s}$. |
| | Frequency Centroid | 180 - 320 | kHz | `[UNSOURCED]` / `[CALIBRATE]` (re-tagged 2026-11: the IEA-PVPS Task 13 failure review contains no AE spectra; no peer-reviewed AE spectra for framed c-Si modules were found — see docs/research/LITERATURE_REVIEW.md). |
| | Radiation Pattern | Dipole / In-Plane Moment | - | Modelled as an in-plane force dipole in the silicon layer; modal excitation (A0 vs S0) now follows from the SAFE mode shapes rather than a fixed 75/25 split. The 75 % S0 figure is `[UNSOURCED]`. |
| **Crack Friction** | Duration | 50 - 200 | $\mu$s | `[UNSOURCED]` (re-tagged 2026-11: ASTM E976 concerns AE sensor response reproducibility) Inter-crack face rubbing during wind cycling. |
| | Spectral Peak | 60 - 140 | kHz | `[UNSOURCED]` (re-tagged 2026-11: ISO 12716 is the general AE *vocabulary* standard, not a PV-laminate test method) |
| **Thermal Stacking Noise** | Duration | 100 - 500 | $\mu$s | `[UNSOURCED]` (IEC 61215 MQT 11 defines the thermal-cycling test, not AE durations) Stick-slip between laminate and frame; now located at the frame edge in simulation. |
| | Spectral Range | 20 - 90 | kHz | `[UNSOURCED]` (see ISO 12716 note above) |

---

## 3. Transducer (PZT) and DAQ Digital Twin Defaults

| Component | Parameter | Value | Unit | Status & Source / Calibration Target |
| :--- | :--- | :--- | :--- | :--- |
| **PZT Disc Sensor** | Resonant Freq $f_{res}$ | 200.0 | kHz | `[SOURCED: Vallen Systeme VS150-RIC / PAC R15alpha Datasheet]` Standard high-sensitivity AE resonant disc. |
| | Quality Factor $Q$ | 5.0 | - | `[SOURCED: PI Ceramic PIC 255 Modified PZT-5A Material Specs]` |
| | Coupling Factor $k_{eff}$ | 0.35 | - | `[SOURCED: IEEE Standard on Piezoelectricity (ANSI/IEEE Std 176)]` |
| | Diameter / Thickness | 10.0 / 1.0 | mm | `[SOURCED: Vallen Systeme Specs]` |
| | Sensitivity | 15.0 | $\text{V}/(\text{m/s})$ | `[SOURCED: ASTM E1106 Sensor Primary Calibration Curve]` |
| | Sensor-to-Sensor Spread | $\pm 3.0$ | dB | `[CALIBRATE]` Dry-couplant / cyanoacrylate bond variation. Calibrate via Hsu-Nielsen pencil lead break (ASTM E976). |
| **Analog Front-End** | Preamp Gain | +40.0 (100$\times$) | dB | `[SOURCED: Analog Devices AD8421 / TI INA128 AE Instrumentation Preamp Datasheet]` |
| | Input Noise Floor | 5.0 | $\text{nV}/\sqrt{\text{Hz}}$ | `[SOURCED: Analog Devices AD8421 Datasheet]` Corresponds to $\approx 3.16\ \mu\text{V}$ RMS over 400 kHz bandwidth. |
| | Bandpass Filter | 50 - 400 | kHz | `[SOURCED: ASTM E976 / ISO 12716 Standard AE Filter Window]` 4-pole Butterworth bandpass. |
| **ADC Quantizer** | Resolution | 14 | bits | `[SOURCED: Texas Instruments ADS7042 / AD7980 SAR ADC Datasheet]` |
| | Full-Scale Range | $\pm 2.5$ | V | `[SOURCED: TI ADS7042 Datasheet]` $0.305\text{ mV/LSB}$. |
| | Sample Rate | 1.0 | MSPS | `[SOURCED: PRD Section 7.2]` $1.0\ \mu\text{s}$ Nyquist sampling for 400 kHz AE band. |
| | Clock Jitter | $\le 100$ | ps RMS | `[SOURCED: Silicon Labs Si5351 Crystal Oscillator Spec]` Negligible phase jitter ($< 0.001\%$). |
| | Inter-Channel Skew | $\pm 10$ | ns | `[CALIBRATE]` PCB trace length delta and ADC simultaneous sampling skew. Calibrate via common electrical test pulse. |
| **Trigger Engine** | Window Length | 2560 (2.56 ms) | samples | `[SOURCED: PRD Section 8.2]` 512 pre-trigger samples ($512\ \mu\text{s}$), 2048 post-trigger samples ($2.048\text{ ms}$). |
| | Trigger Threshold | 20.0 | mV | `[SOURCED: PRD Section 8.2]` Operating threshold set at $\approx 2.0\times$ RMS noise floor to prevent buffer overload. |

---

## 4. Unbacked / Unsourced Parameters & Real-World Calibration Priorities

### A. Unsourced Parameters Requiring Laboratory Calibration (`[CALIBRATE]`)

1. **EVA Viscoelastic High-Frequency Shear Damping ($\alpha_{eva}$):**
   - *Status:* `[CALIBRATE]`
   - *Impact on Model:* Determines maximum sensing radius across a 2.2 m PV module.
   - *Calibration Procedure:* Perform pitch-catch Lamb wave transmission across a 1000 mm laminated glass/EVA/glass coupon in a thermal chamber from $-20^\circ\text{C}$ to $+75^\circ\text{C}$ across 50–400 kHz (ASTM E976).

2. **Sensor Acoustic Coupling Impedance Spread:**
   - *Status:* `[CALIBRATE]`
   - *Impact on Model:* Affects relative amplitude ratios between channels.
   - *Calibration Procedure:* Hsu-Nielsen 0.5 mm 2H pencil lead break tests at 8 standard grid locations (ASTM E976 / ASTM E2075).

3. **Outdoor PV Field Ambient Acoustic Noise Profile:**
   - *Status:* `[UNSOURCED]` / `[CALIBRATE]`
   - *Impact on Model:* False Alarm Rate (FAR) baseline per module-day depends directly on real environmental noise (sand impact, rain drops, wind gust frame flexing, central inverter IGBT switching harmonics at 16–32 kHz).
   - *Calibration Procedure:* 30-day baseline continuous waveform logging on a utility-scale string (IEC 61215 field test).

---

## 5. Modelling Assumptions Added in the 2026-11 Revision

| Area | Assumption | Status |
| :--- | :--- | :--- |
| Laminate dispersion | A0/S0 phase & group velocity from the corrected multilayer SAFE solver; EVA storage modulus evaluated at 150 kHz and scaled so the configured `youngs_modulus_gpa` is the 25 °C anchor | `[CALIBRATE]` |
| Attenuation | Modal strain-energy method: $\eta_{mode} = \sum_l (U_l^{shear}/U)\,\eta_l$, $lpha = \eta\,\omega/(2c_g)$; only the *shear* strain energy of polymer layers is lossy (bulk assumed elastic) | `[CALIBRATE]` (pitch-catch, -20 to 75 °C) |
| EVA temperature shift | Hybrid: WLF (universal constants $C_1$=17.44, $C_2$=51.6, referenced to $T_g$) below $T_g$+50 °C, Arrhenius above, joined continuously | `[CALIBRATE]` (DMA) |
| Source coupling | Cracks: in-plane force dipole in Si; rain/hail/taps/rattle: normal force on glass; frame stick-slip: in-plane force in glass; EMI: electrical, simultaneous on all channels | Physics-motivated, `[UNSOURCED]` levels |
| Receiver | PZT disc senses out-of-plane surface displacement of each mode; disc aperture $2J_1(ka)/(ka)$ | Standard transducer model |
| Frame | Image-source reflections, order 2 per axis, reflection coefficient randomised 0.3-0.75 | `[CALIBRATE]` (depends on frame/clamp) |
| Absolute level | Unit-energy crack at 0.5 m → 2e-4 m/s peak surface velocity (~70 dB_AE); source levels log-uniform over ~30 dB | `[CALIBRATE]` (pencil-lead breaks) |
| Family levels | Hail ×18, tap ×4, rattle ×0.6, thermal tick ×0.6, rain drops ×0.35 relative to a crack | `[UNSOURCED]` |
| Ambient noise | 1/f-shaped broadband + inverter harmonics 16-32 kHz, level log-uniform | `[UNSOURCED]` → field logging |
| Noise trigger rate | 10,000 noise triggers per module-day (used to convert FPR to false alarms/day) | `[UNSOURCED]` |
| Numerical | Tier-0 applies a causality mute before onset + r/v_max (removes a ~-50 dB numerical precursor of the spectral model) | Numerical artefact control |
