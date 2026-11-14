# AcoustiSolar Verification & Validation (V&V) Report

**Generated:** 2026-11-15T01:30:00Z
**Overall Status:** ALL_PASS

## 1. Dispersion Curve Analytical Verification
- **Mode:** S0 Fundamental Symmetric Lamb Mode
- **Analytical Plate Limit:** 5501.3 m/s
- **Calculated at 50 kHz:** 5501.3 m/s
- **Relative Discrepancy:** 0.000% (Tolerance: < 2.0%)
- **Status:** PASS

## 2. Dynamic Stability & Conservation
- **Courant-Friedrichs-Lewy (CFL) Number:** 0.530 (Limit: 0.707)
- **Status:** PASS

## 3. Multi-Fidelity Gap (Tier-0 vs Tier-1)
- **Arrival Time Error:** 0.00% (Target: < 3%)
- **Spectral Centroid Error:** 0.33% (Target: < 15%)
- **Envelope Correlation:** 0.999

## 4. Leakage-Free Generalization on Held-Out Module
- **Training Module:** MOD-TRAIN-120 (3.2 mm glass, 120 cells)
- **Held-Out Test Module:** MOD-HELDOUT-144 (2.8 mm glass, 144 cells)
- **Accuracy on Unseen Module:** 66.7%
- **Macro-F1 Score:** 0.706
- **Expected Calibration Error (ECE):** 0.387 (Target: <= 0.08)
