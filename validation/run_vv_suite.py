"""
Automated Verification & Validation (V&V) suite for AcoustiSolar.

Every check computes a real quantity and compares it with a tolerance.  Status values:
* PASS / FAIL      — verification checks (the code does what the physics says)
* KNOWN_GAP        — a product KPI target that the current simulation-only system does not
                     yet meet; reported honestly, does not fail CI
Results go to ``validation/out/`` (the curated ``VV_REPORT.md`` is never overwritten).

History: an earlier version compared random noise with itself as the "Tier-0 vs Tier-1
gap", hard-coded PASS for reciprocity and grid convergence, and overwrote the report.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
for pkg in ("schemas", "physics", "daqtwin", "dsp", "ml", "sdk"):
    sys.path.insert(0, str(root / "packages" / pkg / "src"))
sys.path.insert(0, str(root / "validation"))

from acoustisolar_physics.lamb import LambDispersionSolver  # noqa: E402
from acoustisolar_physics.materials import GLASS_MATERIAL  # noqa: E402
from acoustisolar_physics.safe import MultilayerSAFESolver  # noqa: E402
from acoustisolar_physics.tier0 import Tier0Simulator, laminate_for_module  # noqa: E402
from acoustisolar_physics.tier1 import Tier1WaveSolver  # noqa: E402
from acoustisolar_schemas.module import ModuleTwinConfig  # noqa: E402
from acoustisolar_schemas.source import SourceEventConfig  # noqa: E402

OUT_DIR = root / "validation" / "out"


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_lamb_vs_independent_reference() -> dict:
    from verify_physics import independent_lamb_reference

    solver = LambDispersionSolver(thickness_m=3.2e-3)
    errs = []
    for f in (50e3, 150e3, 300e3):
        for mode, fn in (("S0", solver.find_phase_velocity_s0), ("A0", solver.find_phase_velocity_a0)):
            ref = independent_lamb_reference(f, 3.2e-3, GLASS_MATERIAL.c_longitudinal, GLASS_MATERIAL.c_transverse, mode)
            errs.append(abs(fn(f) - ref) / ref * 100.0)
    return {"status": _status(max(errs) < 1.0), "max_error_pct": float(max(errs)), "tolerance_pct": 1.0}


def check_safe_single_layer_regression() -> dict:
    safe = MultilayerSAFESolver(layers=[("glass", 3.2e-3, 2500.0, 72.0e9, 0.22)], n_elem_per_layer=12)
    lamb = LambDispersionSolver(thickness_m=3.2e-3)
    errs = []
    for k in (150.0, 300.0, 600.0):
        fs, cs, fa, ca = safe.compute_dispersion(np.array([k]))
        errs.append(abs(cs[0] - lamb.find_phase_velocity_s0(fs[0])) / cs[0] * 100.0)
        errs.append(abs(ca[0] - lamb.find_phase_velocity_a0(fa[0])) / ca[0] * 100.0)
    return {"status": _status(max(errs) < 1.5), "max_error_pct": float(max(errs)), "tolerance_pct": 1.5}


def check_laminate_s0_physical() -> dict:
    v = laminate_for_module(ModuleTwinConfig()).s0_velocity(25.0)
    c_plate = GLASS_MATERIAL.c_plate_s0_limit
    delta = (v - c_plate) / c_plate * 100.0
    return {"status": _status(-10.0 < delta < 1.0), "laminate_s0_m_s": float(v), "glass_plate_m_s": float(c_plate),
            "delta_pct": float(delta), "note": "thin soft layers mass-load the glass by a few % only"}


def check_tier0_causality() -> dict:
    sim = Tier0Simulator()
    worst = 0.0
    v_max = sim.laminate.group_velocity_limit("S0")
    for i, (x, y) in enumerate([(300, 500), (520, 880), (900, 1500)]):
        w = sim.simulate(SourceEventConfig(x_mm=x, y_mm=y), 25.0, np.random.default_rng(i), onset_s=400e-6)
        for ch, (sx, sy) in enumerate(sim.sensor_config.positions_mm):
            t_min = 400e-6 + np.hypot(x - sx, y - sy) * 1e-3 / v_max
            i0 = int(t_min * 1e6) - 10
            worst = max(worst, float(np.max(np.abs(w[ch, :i0])) / np.max(np.abs(w[ch]))))
    return {"status": _status(worst < 1e-6), "max_pre_arrival_ratio": worst}


def check_tier1_reciprocity() -> dict:
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.01, dt_s=1e-6, pml_thickness_cells=6)
    c = np.full((solver.nx, solver.ny), 5300.0)
    c[22:30, 10:20] = 3800.0
    solver.set_c_field(c)
    t = np.arange(220) * 1e-6
    f0 = 120e3
    tau = t - 2.0 / f0
    src = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-((np.pi * f0 * tau) ** 2))
    a, b = (0.12, 0.15), (0.31, 0.30)
    ab = solver.run_simulation(*a, src, [b])[0][0]
    ba = solver.run_simulation(*b, src, [a])[0][0]
    err = float(np.linalg.norm(ab - ba) / np.linalg.norm(ab))
    return {"status": _status(err < 1e-6), "relative_l2_error": err, "heterogeneous": True}


def check_tier1_convergence() -> dict:
    """Observed order of the FD scheme with a smooth (Gaussian) source, three grids."""
    c0, f0, sigma = 5300.0, 40e3, 0.012
    sigs = []
    t_end = 90e-6
    t_common = np.linspace(0, t_end, 600)
    for dx in (0.006, 0.003, 0.0015):
        dt = 0.4 * dx / c0
        n = int(t_end / dt) + 1
        t = np.arange(n) * dt
        tau = t - 1.5 / f0
        src = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-((np.pi * f0 * tau) ** 2))
        s = Tier1WaveSolver(width_m=0.36, length_m=0.36, dx_m=dx, dt_s=dt, c_speed_m_s=c0, boundary="reflecting")
        # footprint sums to 1 per cell-area: divide by dx^2 to get a grid-independent field
        w = s.run_simulation(0.18, 0.18, src, [(0.252, 0.18)], source_sigma_m=sigma)[0][0] / dx**2
        sigs.append(np.interp(t_common, t, w))
    e1 = np.linalg.norm(sigs[0] - sigs[1])
    e2 = np.linalg.norm(sigs[1] - sigs[2])
    order = float(np.log2(e1 / max(e2, 1e-30)))
    return {"status": _status(1.5 < order < 2.8), "observed_order": order, "expected_order": 2.0,
            "points_per_wavelength_finest": float(c0 / (2.5 * f0) / 0.0015)}


def check_tier0_tier1_arrival_agreement() -> dict:
    """Kinematic agreement: wave speed recovered from Tier-1 arrival times vs distance must
    match the laminate S0 speed that Tier-0 uses (slope fit removes the wavelet offset)."""
    sim = Tier0Simulator()
    v = sim.s0_velocity(25.0)
    dx, dt = 0.004, 0.35e-6
    t1 = Tier1WaveSolver(width_m=0.7, length_m=0.7, dx_m=dx, dt_s=dt, c_speed_m_s=v, pml_thickness_cells=15)
    n = 650
    t = np.arange(n) * dt
    f0 = 100e3
    tau = t - 1.5 / f0
    src = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-((np.pi * f0 * tau) ** 2))
    src_xy = (0.12, 0.12)
    sensors = [(0.24, 0.12), (0.36, 0.12), (0.48, 0.12), (0.12, 0.56)]
    w1 = t1.run_simulation(*src_xy, src, sensors)[0]
    d = np.array([np.hypot(sx - src_xy[0], sy - src_xy[1]) for sx, sy in sensors])
    first = np.array([np.argmax(np.abs(w) > 0.1 * np.max(np.abs(w))) * dt for w in w1])
    slope = np.polyfit(d, first, 1)[0]
    v_t1 = 1.0 / slope
    err = abs(v_t1 - v) / v * 100.0
    return {"status": _status(err < 3.0), "tier1_speed_m_s": float(v_t1), "tier0_s0_speed_m_s": float(v),
            "speed_error_pct": float(err), "tolerance_pct": 3.0}


def check_ml_held_out() -> dict:
    from acoustisolar_ml.evaluation import HeldOutEvaluator

    m = HeldOutEvaluator(seed=42).run_held_out_benchmark(n_train=200, n_test=100)
    targets = {"f1_score": 0.95, "ece_max": 0.05}
    gaps = []
    if m["f1_score"] < targets["f1_score"]:
        gaps.append("new-fracture F1 below PRD target")
    if m["ece"] > targets["ece_max"]:
        gaps.append("ECE above PRD target")
    return {
        "status": "KNOWN_GAP" if gaps else "PASS",
        "gaps": gaps,
        "f1_new_fracture": m["f1_score"],
        "pr_auc": m["pr_auc"],
        "ece": m["ece"],
        "macro_f1_multiclass": m.get("macro_f1_multiclass"),
        "localization": m.get("localization"),
        "train_module": m["train_module"],
        "held_out_module": m["held_out_module"],
        "note": "small run for CI; see validation/kpi_benchmark_summary.json for the scale benchmark",
    }


CHECKS = {
    "lamb_vs_independent_reference": check_lamb_vs_independent_reference,
    "safe_single_layer_regression": check_safe_single_layer_regression,
    "laminate_s0_physical": check_laminate_s0_physical,
    "tier0_causality": check_tier0_causality,
    "tier1_heterogeneous_reciprocity": check_tier1_reciprocity,
    "tier1_grid_convergence": check_tier1_convergence,
    "tier0_tier1_arrival_agreement": check_tier0_tier1_arrival_agreement,
    "ml_held_out_generalization": check_ml_held_out,
}


def run_vv_suite(include_ml: bool = True) -> dict:
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tests": {}}
    for name, fn in CHECKS.items():
        if name.startswith("ml_") and not include_ml:
            continue
        print(f"=== {name} ===")
        try:
            res = fn()
        except Exception as exc:  # a crash is a failure, never a silent pass
            res = {"status": "FAIL", "error": repr(exc)}
        report["tests"][name] = res
        print(f"  -> {res['status']}  " + ", ".join(f"{k}={v}" for k, v in res.items() if k != "status" and not isinstance(v, (dict, list))))
    statuses = [t["status"] for t in report["tests"].values()]
    if "FAIL" in statuses:
        report["summary"] = "FAILURES_DETECTED"
    elif "KNOWN_GAP" in statuses:
        report["summary"] = "PASS_WITH_KNOWN_GAPS"
    else:
        report["summary"] = "ALL_PASS"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vv_suite_results.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    lines = [f"# V&V suite results\n\n**Generated:** {report['timestamp_utc']}  \n**Summary:** {report['summary']}\n",
             "| Check | Status | Key values |", "|---|---|---|"]
    for name, res in report["tests"].items():
        vals = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in res.items() if k not in ("status", "localization", "gaps", "note"))
        lines.append(f"| {name} | {res['status']} | {vals} |")
    (OUT_DIR / "vv_suite_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {report['summary']}  (written to {OUT_DIR})")
    return report


if __name__ == "__main__":
    rep = run_vv_suite()
    sys.exit(1 if rep["summary"] == "FAILURES_DETECTED" else 0)
