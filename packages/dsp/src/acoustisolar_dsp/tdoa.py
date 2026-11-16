"""
TDOA source localisation for plate-like PV modules.

Key point (verified in docs/research/CODEBASE_EVALUATION.md §4.1): cross-correlating whole
event windows locks onto the large, slow, dispersive A0 packet while the solver assumes the
S0 speed, producing gross errors of tens to hundreds of microseconds.  The default method
here is therefore **mode-aware / first-arrival gated**:

1. pick each channel's first (S0) arrival: robust-noise threshold, refined with AIC;
2. refine the relative alignment by cross-correlating only a short gate after each onset;
3. reject time differences that violate |dt_ij| <= d_ij / v (physically impossible);
4. solve for (x, y, t0) with all sensors (equivalent to all pairs) using a grid start,
   Huber-weighted Gauss-Newton and single-outlier rejection.

The legacy full-window cross-correlation and ``solve_hyperbolic`` API are kept.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .pickers import OnsetPicker


class TDOALocalizer:
    def __init__(
        self,
        sensor_positions_mm: Sequence[Tuple[float, float]],
        nominal_velocity_m_s: float = 5300.0,
        sample_rate_hz: float = 1_000_000.0,
    ):
        self.sensors_mm = np.array(sensor_positions_mm, dtype=np.float64)
        self.v_s0 = nominal_velocity_m_s
        self.sample_rate = sample_rate_hz
        self.dt_s = 1.0 / sample_rate_hz

    # ------------------------------------------------------------------ helpers
    def _max_lag_samples(self, v_m_s: Optional[float] = None) -> int:
        v = v_m_s or self.v_s0
        d = np.max(np.linalg.norm(self.sensors_mm[:, None, :] - self.sensors_mm[None, :, :], axis=-1))
        return int(np.ceil(d * 1e-3 / v * self.sample_rate)) + 5

    @staticmethod
    def _parabolic(y: np.ndarray, i: int) -> float:
        if 0 < i < len(y) - 1:
            y1, y2, y3 = y[i - 1], y[i], y[i + 1]
            den = 2.0 * (2.0 * y2 - y1 - y3)
            if abs(den) > 1e-30:
                return float(i + (y1 - y3) / den)
        return float(i)

    # ------------------------------------------------------------------ arrival picking
    def pick_arrivals(
        self,
        waveforms: np.ndarray,
        k_sigma: float = 6.0,
        aic_back: int = 60,
        aic_fwd: int = 8,
        gate: int = 40,
        refine: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """First-arrival times (samples, float) per channel and a validity mask."""
        w = np.asarray(waveforms, dtype=np.float64)
        n_ch, n = w.shape
        onsets = np.full(n_ch, np.nan)
        for ch in range(n_ch):
            c = OnsetPicker.threshold_onset(w[ch], k_sigma=k_sigma)
            if c < 0:
                continue
            onsets[ch] = OnsetPicker.pick_aic(w[ch], (max(0, c - aic_back), min(n, c + aic_fwd)))
        valid = ~np.isnan(onsets)
        if not refine or valid.sum() < 2:
            return onsets, valid

        # Gated cross-correlation refinement against the earliest channel
        ref = int(np.nanargmin(onsets))
        r0 = int(onsets[ref])
        ref_snip = w[ref, max(0, r0 - 5) : r0 + gate]
        refined = onsets.copy()
        for ch in np.where(valid)[0]:
            if ch == ref:
                continue
            o = int(onsets[ch])
            snip = w[ch, max(0, o - 5) : o + gate]
            m = min(len(snip), len(ref_snip))
            if m < 10:
                continue
            a = snip[:m] / (np.linalg.norm(snip[:m]) + 1e-30)
            b = ref_snip[:m] / (np.linalg.norm(ref_snip[:m]) + 1e-30)
            xc = np.correlate(a, b, mode="full")
            lags = np.arange(-(m - 1), m)
            win = np.abs(lags) <= 6
            sub = xc[win]
            j = int(np.argmax(sub))
            if sub[j] < 0.5:  # waveforms too dissimilar: keep the AIC pick
                continue
            # np.correlate peak at +L means this channel's snippet is L samples *later*
            # than the reference snippet, i.e. its onset pick was L samples early.
            lag_val = lags[win][0] + self._parabolic(sub, j)
            refined[ch] = onsets[ch] + lag_val
        return refined, valid

    def estimate_time_differences(
        self, waveforms: np.ndarray, ref_channel: int = 0, method: str = "gated"
    ) -> np.ndarray:
        """delta_t [s] of each channel relative to ``ref_channel``.

        ``method="gated"`` (default): first-arrival picking + gated refinement.
        ``method="full_xcorr"``: legacy whole-window cross-correlation (prone to A0 lock-on).
        """
        w = np.asarray(waveforms, dtype=np.float64)
        if method == "gated":
            arr, valid = self.pick_arrivals(w)
            if valid[ref_channel] and valid.sum() >= 3:
                dt = (arr - arr[ref_channel]) * self.dt_s
                dt[~valid] = 0.0
                return dt
            method = "full_xcorr"  # fall back if onsets could not be picked

        n_channels, n_samples = w.shape
        dt_array = np.zeros(n_channels)
        ref = w[ref_channel]
        window = self._max_lag_samples()
        for ch in range(n_channels):
            if ch == ref_channel:
                continue
            xcorr = np.correlate(w[ch], ref, mode="full")
            lags = np.arange(-(n_samples - 1), n_samples)
            sel = np.where(np.abs(lags) <= window)[0]
            sub = xcorr[sel]
            i = int(np.argmax(sub))
            dt_array[ch] = (lags[sel][0] + self._parabolic(sub, i)) * self.dt_s
        return dt_array

    # ------------------------------------------------------------------ consistency
    def consistency_violations(self, arrivals_s: np.ndarray, v_m_s: float, tol_s: float = 3e-6) -> np.ndarray:
        """Boolean matrix of sensor pairs whose |dt| exceeds the travel time between them."""
        d = np.linalg.norm(self.sensors_mm[:, None, :] - self.sensors_mm[None, :, :], axis=-1) * 1e-3
        dt = np.abs(arrivals_s[:, None] - arrivals_s[None, :])
        return dt > d / v_m_s + tol_s

    # ------------------------------------------------------------------ solvers
    def solve_hyperbolic(
        self,
        dt_seconds: np.ndarray,
        initial_guess_mm: Tuple[float, float] = (500.0, 800.0),
        v_eff_m_s: Optional[float] = None,
        max_iter: int = 40,
        tol: float = 1e-4,
    ) -> Dict[str, Any]:
        """Legacy Gauss-Newton hyperbolic solver on differences against sensor 0."""
        v_mm_s = (v_eff_m_s or self.v_s0) * 1000.0
        sensors = self.sensors_mm
        n_sensors = len(sensors)
        x_ref, y_ref = sensors[0]
        dd_measured = v_mm_s * np.asarray(dt_seconds)
        pos = np.array(initial_guess_mm, dtype=np.float64)
        cov = np.eye(2) * 100.0
        success = False
        for _ in range(max_iter):
            x, y = pos
            d0 = np.hypot(x - x_ref, y - y_ref) + 1e-6
            res, J = [], []
            for i in range(1, n_sensors):
                xi, yi = sensors[i]
                di = np.hypot(x - xi, y - yi) + 1e-6
                res.append(di - d0 - dd_measured[i])
                J.append([(x - xi) / di - (x - x_ref) / d0, (y - yi) / di - (y - y_ref) / d0])
            res, J = np.array(res), np.array(J)
            JtJ = J.T @ J
            try:
                delta = np.linalg.solve(JtJ + 1e-3 * np.trace(JtJ) * np.eye(2), -J.T @ res)
            except np.linalg.LinAlgError:
                break
            pos += delta
            if np.linalg.norm(delta) < tol:
                success = True
                try:
                    cov = np.linalg.inv(JtJ + 1e-6 * np.eye(2)) * (np.sum(res**2) / max(1, n_sensors - 3))
                except np.linalg.LinAlgError:
                    pass
                break
        return self._pack(pos, cov, success)

    @staticmethod
    def _pack(pos, cov, success, **extra) -> Dict[str, Any]:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-4)
        out = {
            "x_mm": float(pos[0]),
            "y_mm": float(pos[1]),
            "uncertainty_semi_major_mm": float(2.0 * np.sqrt(eigvals[1])),
            "uncertainty_semi_minor_mm": float(2.0 * np.sqrt(eigvals[0])),
            "uncertainty_angle_deg": float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))),
            "covariance": cov.tolist(),
            "converged": bool(success),
        }
        out.update(extra)
        return out

    def _solve_once(self, t, Sm, vel_mm_s, bounds_mm, sigma_t_s, huber_s):
        """Grid start + Huber-weighted Gauss-Newton for (x, y, t0) with per-sensor speeds."""
        xs = np.linspace(bounds_mm[0], bounds_mm[1], 60)
        ys = np.linspace(bounds_mm[2], bounds_mm[3], 90)
        X, Y = np.meshgrid(xs, ys)
        D = np.sqrt((X[..., None] - Sm[:, 0]) ** 2 + (Y[..., None] - Sm[:, 1]) ** 2) / vel_mm_s
        t0 = np.median(t - D, axis=-1)
        R = t - D - t0[..., None]
        cost = np.sum(np.where(np.abs(R) <= huber_s, 0.5 * R**2, huber_s * (np.abs(R) - 0.5 * huber_s)), axis=-1)
        i = np.unravel_index(np.argmin(cost), cost.shape)
        p = np.array([X[i], Y[i], t0[i]])
        success = False
        for _ in range(30):
            d = np.hypot(p[0] - Sm[:, 0], p[1] - Sm[:, 1]) + 1e-6
            r = t - p[2] - d / vel_mm_s
            w = np.where(np.abs(r) <= huber_s, 1.0, huber_s / np.maximum(np.abs(r), 1e-30))
            J = np.column_stack([-(p[0] - Sm[:, 0]) / (d * vel_mm_s), -(p[1] - Sm[:, 1]) / (d * vel_mm_s),
                                 -np.ones_like(d)])
            JW = J * w[:, None]
            try:
                step = np.linalg.solve(JW.T @ J + 1e-12 * np.eye(3), -JW.T @ r)
            except np.linalg.LinAlgError:
                break
            p += step
            if np.hypot(step[0], step[1]) < 1e-3:
                success = True
                break
        p[0] = float(np.clip(p[0], bounds_mm[0], bounds_mm[1]))
        p[1] = float(np.clip(p[1], bounds_mm[2], bounds_mm[3]))
        d = np.hypot(p[0] - Sm[:, 0], p[1] - Sm[:, 1]) + 1e-6
        r = t - p[2] - d / vel_mm_s
        J = np.column_stack([-(p[0] - Sm[:, 0]) / (d * vel_mm_s), -(p[1] - Sm[:, 1]) / (d * vel_mm_s),
                             -np.ones_like(d)])
        dof = max(1, len(t) - 3)
        sig = max(sigma_t_s, float(np.sqrt(np.sum(r**2) / dof))) if len(t) > 3 else sigma_t_s
        try:
            cov = (np.linalg.inv(J.T @ J + 1e-18 * np.eye(3)) * sig**2)[:2, :2]
        except np.linalg.LinAlgError:
            cov = np.eye(2) * 1e6
        return p, r, cov, success

    def solve_arrivals(
        self,
        arrivals_s: np.ndarray,
        v_eff_m_s: Optional[float] = None,
        bounds_mm: Optional[Tuple[float, float, float, float]] = None,
        sigma_t_s: float = 1.0e-6,
        huber_s: float = 3.0e-6,
        sensor_mask: Optional[np.ndarray] = None,
        velocities_m_s: Optional[np.ndarray] = None,
        outlier_s: float = 8.0e-6,
    ) -> Dict[str, Any]:
        """Robust solve of t_i = t0 + |x - x_i| / v_i for (x, y, t0) from absolute arrivals.

        ``velocities_m_s`` gives a speed per sensor (mode-aware solving); otherwise all use
        ``v_eff_m_s``.  If the worst residual exceeds ``outlier_s`` and more than three
        sensors are available, that sensor is dropped and the problem re-solved.
        """
        S = self.sensors_mm
        n = len(S)
        arrivals_s = np.asarray(arrivals_s, dtype=float)
        vel = np.full(n, (v_eff_m_s or self.v_s0)) if velocities_m_s is None else np.asarray(velocities_m_s, float)
        mask = np.ones(n, bool) if sensor_mask is None else np.asarray(sensor_mask, bool).copy()
        mask &= ~np.isnan(arrivals_s)
        if mask.sum() < 3:
            return self._pack(np.array([np.nan, np.nan]), np.eye(2) * 1e6, False, sensors_used=int(mask.sum()),
                              residual_rms_s=float("inf"))
        if bounds_mm is None:
            lo, hi = S.min(axis=0) - 100.0, S.max(axis=0) + 100.0
            bounds_mm = (lo[0], hi[0], lo[1], hi[1])

        rejected = None
        p, r, cov, success = self._solve_once(arrivals_s[mask], S[mask], vel[mask] * 1000.0, bounds_mm,
                                              sigma_t_s, huber_s)
        if mask.sum() >= 4 and np.max(np.abs(r)) > outlier_s:
            # Leave-one-out: a single bad pick drags the whole solution, so the largest
            # residual is not necessarily on the bad channel.  Try dropping each sensor.
            best = None
            for j in np.where(mask)[0]:
                m2 = mask.copy()
                m2[j] = False
                p2, r2, cov2, ok2 = self._solve_once(arrivals_s[m2], S[m2], vel[m2] * 1000.0, bounds_mm,
                                                     sigma_t_s, huber_s)
                rms2 = float(np.sqrt(np.mean(r2**2)))
                if best is None or rms2 < best[0]:
                    best = (rms2, p2, r2, cov2, ok2, m2, int(j))
            if best is not None and best[0] < float(np.sqrt(np.mean(r**2))):
                _, p, r, cov, success, mask, rejected = best
        full_res = np.full(n, np.nan)
        full_res[mask] = r
        out = self._pack(
            p[:2], cov, success,
            t0_s=float(p[2]),
            residuals_s=full_res.tolist(),
            residual_rms_s=float(np.sqrt(np.mean(r**2))),
            sensors_used=int(mask.sum()),
        )
        if rejected is not None:
            out["rejected_sensor"] = rejected
        return out

    def locate(
        self,
        waveforms: np.ndarray,
        v_eff_m_s: Optional[float] = None,
        bounds_mm: Optional[Tuple[float, float, float, float]] = None,
        outlier_s: float = 8.0e-6,
        min_snr_db: float = 25.0,
        v_a0_m_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: gated arrival picking -> SNR gating -> mode-aware robust solve.

        At low SNR, or far from the source, the weak S0 head wave can be buried in noise and
        the picker lands on the A0 front instead.  With ``v_a0_m_s`` given, every assignment
        of {S0, A0} to the picked channels is tried (a small prior penalty discourages
        calling high-SNR channels A0) and the best-fitting one is kept.  Without it, low-SNR
        channels are excluded when at least three stronger ones remain.
        """
        v = v_eff_m_s or self.v_s0
        w = np.asarray(waveforms, dtype=np.float64)
        arr, valid = self.pick_arrivals(w)
        snr = np.array([
            20.0 * np.log10(np.max(np.abs(ch)) / max(OnsetPicker.robust_noise_sigma(ch), 1e-30)) for ch in w
        ])
        arr_all_s = np.where(valid, arr * self.dt_s, np.nan)

        res = None
        if v_a0_m_s is not None and valid.sum() >= 4:
            import itertools

            idx = np.where(valid)[0]
            best, best_score = None, np.inf
            for combo in itertools.product((0, 1), repeat=len(idx)):
                if sum(combo) > len(idx) - 2:
                    continue  # at least two S0 picks anchor the solution
                vel = np.full(len(self.sensors_mm), v)
                vel[idx[np.array(combo, bool)]] = v_a0_m_s
                res_c = self.solve_arrivals(arr_all_s, bounds_mm=bounds_mm, sensor_mask=valid,
                                            velocities_m_s=vel, outlier_s=np.inf)
                if not np.isfinite(res_c["x_mm"]):
                    continue
                penalty = sum((1.0e-6 if snr[i] >= 40.0 else 0.2e-6) for i, c in zip(idx, combo) if c)
                score = res_c["residual_rms_s"] + penalty
                if score < best_score:
                    best, best_score = (res_c, combo), score
            if best is not None:
                res = best[0]
                res["mode_assignment"] = {int(i): ("A0" if c else "S0") for i, c in zip(idx, best[1])}
        if res is None:
            strong = valid & (snr >= min_snr_db)
            if strong.sum() >= 3:
                valid = strong
            elif valid.sum() > 3:
                keep = np.argsort(-np.where(valid, snr, -np.inf))[:3]
                valid = np.zeros_like(valid)
                valid[keep] = True
            res = self.solve_arrivals(np.where(valid, arr_all_s, np.nan), v, bounds_mm, outlier_s=outlier_s)

        viol = self.consistency_violations(np.nan_to_num(arr_all_s, nan=0.0), v)
        viol[~valid, :] = False
        viol[:, ~valid] = False
        res["arrival_samples"] = [None if np.isnan(a) else float(a) for a in arr]
        res["channel_snr_db"] = [float(s) for s in snr]
        res["n_consistency_violations"] = int(np.sum(np.triu(viol, 1)))
        return res

    @staticmethod
    def as_cell(x_mm: float, y_mm: float, width_mm: float, length_mm: float, rows: int, cols: int) -> str:
        """Map a position to the layout's cell id ``C_<row>_<col>``."""
        col = int(min(max(0, (x_mm / width_mm) * cols), cols - 1))
        row = int(min(max(0, (y_mm / length_mm) * rows), rows - 1))
        return f"C_{row}_{col}"


class DeltaTMapLocalizer:
    """
    Calibration-based Delta-T mapping (Baxter et al. 2007; AIC-Delta-T, Pearson et al. 2016).

    A grid of artificial sources (pencil-lead breaks on the real module, or simulated ones)
    provides measured arrival-time-difference vectors; a new event is located by finding the
    grid position whose calibrated differences best match.  No wave-speed model and no
    straight-ray assumption are needed, so laminate dispersion, frame effects and sensor
    placement errors are absorbed by the calibration.
    """

    def __init__(self, grid_x_mm: Sequence[float], grid_y_mm: Sequence[float], n_sensors: int = 4):
        self.gx = np.asarray(grid_x_mm, float)
        self.gy = np.asarray(grid_y_mm, float)
        self.n_sensors = n_sensors
        self.pairs = [(i, j) for i in range(n_sensors) for j in range(i + 1, n_sensors)]
        self.maps: Optional[np.ndarray] = None  # (ny, nx, n_pairs) seconds

    def fit(self, arrivals_s_grid: np.ndarray) -> "DeltaTMapLocalizer":
        """``arrivals_s_grid``: (ny, nx, n_sensors) calibrated arrival times (any offset)."""
        a = np.asarray(arrivals_s_grid, float)
        self.maps = np.stack([a[..., i] - a[..., j] for i, j in self.pairs], axis=-1)
        return self

    def calibrate(self, arrival_fn) -> "DeltaTMapLocalizer":
        """Build maps by calling ``arrival_fn(x_mm, y_mm) -> arrivals_s`` at every grid node."""
        grid = np.zeros((len(self.gy), len(self.gx), self.n_sensors))
        for iy, y in enumerate(self.gy):
            for ix, x in enumerate(self.gx):
                grid[iy, ix] = arrival_fn(x, y)
        return self.fit(grid)

    def locate(self, arrivals_s: np.ndarray, upsample: int = 4) -> Dict[str, Any]:
        if self.maps is None:
            raise RuntimeError("DeltaTMapLocalizer.fit/calibrate must be called first")
        from scipy.interpolate import RegularGridInterpolator

        a = np.asarray(arrivals_s, float)
        meas = np.array([a[i] - a[j] for i, j in self.pairs])
        ok = ~np.isnan(meas)
        fx = np.linspace(self.gx[0], self.gx[-1], (len(self.gx) - 1) * upsample + 1)
        fy = np.linspace(self.gy[0], self.gy[-1], (len(self.gy) - 1) * upsample + 1)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        pts = np.column_stack([FY.ravel(), FX.ravel()])
        cost = np.zeros(len(pts))
        for p in np.where(ok)[0]:
            interp = RegularGridInterpolator((self.gy, self.gx), self.maps[..., p])
            cost += (interp(pts) - meas[p]) ** 2
        best = int(np.argmin(cost))
        return {"x_mm": float(pts[best, 1]), "y_mm": float(pts[best, 0]),
                "cost_s2": float(cost[best]), "pairs_used": int(ok.sum())}
