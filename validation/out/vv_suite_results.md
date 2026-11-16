# V&V suite results

**Generated:** 2026-11-16T04:52:52+00:00  
**Summary:** ALL_PASS

| Check | Status | Key values |
|---|---|---|
| lamb_vs_independent_reference | PASS | max_error_pct=0.004656, tolerance_pct=1 |
| safe_single_layer_regression | PASS | max_error_pct=0.08176, tolerance_pct=1.5 |
| laminate_s0_physical | PASS | laminate_s0_m_s=5491, glass_plate_m_s=5501, delta_pct=-0.1869 |
| tier0_causality | PASS | max_pre_arrival_ratio=0 |
| tier1_heterogeneous_reciprocity | PASS | relative_l2_error=1.155e-15, heterogeneous=True |
| tier1_grid_convergence | PASS | observed_order=1.93, expected_order=2, points_per_wavelength_finest=35.33 |
| tier0_tier1_arrival_agreement | PASS | tier1_speed_m_s=5472, tier0_s0_speed_m_s=5491, speed_error_pct=0.3415, tolerance_pct=3 |
