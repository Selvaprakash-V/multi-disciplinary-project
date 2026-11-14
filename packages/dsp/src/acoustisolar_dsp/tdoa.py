import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class TDOALocalizer:
    """
    Time Difference Of Arrival (TDOA) source localization solver.
    Solves for source (x, y) coordinates using hyperbolic multilateration
    with dispersion-corrected Lamb mode propagation velocity and uncertainty estimation.
    """

    def __init__(
        self,
        sensor_positions_mm: List[Tuple[float, float]],
        nominal_velocity_m_s: float = 5300.0,
        sample_rate_hz: float = 1_000_000.0
    ):
        self.sensors_mm = np.array(sensor_positions_mm, dtype=np.float64)
        self.v_s0 = nominal_velocity_m_s
        self.sample_rate = sample_rate_hz
        self.dt_s = 1.0 / sample_rate_hz

    def estimate_time_differences(
        self,
        waveforms: np.ndarray,
        ref_channel: int = 0
    ) -> np.ndarray:
        """
        Estimate delta_t [seconds] relative to ref_channel using cross-correlation
        with parabolic peak interpolation for sub-sample resolution.
        """
        n_channels, n_samples = waveforms.shape
        dt_array = np.zeros(n_channels, dtype=np.float64)
        ref_signal = waveforms[ref_channel]
        
        for ch in range(n_channels):
            if ch == ref_channel:
                continue
            sig = waveforms[ch]
            # Cross correlation
            xcorr = np.correlate(sig, ref_signal, mode='full')
            mid = n_samples - 1
            lags = np.arange(-mid, mid + 1)
            
            # Find maximum in reasonable lag window (max travel time across 2m panel is ~0.4 ms = 400 samples)
            window = 500
            valid_idx = np.where(np.abs(lags) <= window)[0]
            sub_xcorr = xcorr[valid_idx]
            sub_lags = lags[valid_idx]
            
            max_idx = np.argmax(sub_xcorr)
            best_lag = sub_lags[max_idx]
            
            # Sub-sample parabolic interpolation
            if 0 < max_idx < len(sub_xcorr) - 1:
                y1 = sub_xcorr[max_idx - 1]
                y2 = sub_xcorr[max_idx]
                y3 = sub_xcorr[max_idx + 1]
                denom = 2.0 * (2.0 * y2 - y1 - y3)
                if abs(denom) > 1e-12:
                    sub_shift = (y1 - y3) / denom
                    best_lag = best_lag + sub_shift
                    
            dt_array[ch] = best_lag * self.dt_s
            
        return dt_array

    def solve_hyperbolic(
        self,
        dt_seconds: np.ndarray,
        initial_guess_mm: Tuple[float, float] = (500.0, 800.0),
        v_eff_m_s: Optional[float] = None,
        max_iter: int = 40,
        tol: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Nonlinear least-squares Gauss-Newton solver for source position (x, y) [mm].
        Minimizes residual r_i = ||x - x_i|| - ||x - x_0|| - v * dt_i.
        """
        v = v_eff_m_s or self.v_s0 # m/s
        v_mm_s = v * 1000.0        # mm/s
        
        sensors = self.sensors_mm
        n_sensors = len(sensors)
        x_ref, y_ref = sensors[0]
        
        # Target distance differences: dd_i = v_mm_s * dt_i
        dd_measured = v_mm_s * dt_seconds
        
        pos = np.array(initial_guess_mm, dtype=np.float64)
        
        cov = np.eye(2) * 100.0
        success = False
        
        for iteration in range(max_iter):
            x, y = pos[0], pos[1]
            
            d0 = np.sqrt((x - x_ref)**2 + (y - y_ref)**2) + 1e-6
            
            residuals = []
            J = []
            
            for i in range(1, n_sensors):
                xi, yi = sensors[i]
                di = np.sqrt((x - xi)**2 + (y - yi)**2) + 1e-6
                
                # Model predicted distance difference
                dd_pred = di - d0
                res = dd_pred - dd_measured[i]
                residuals.append(res)
                
                # Jacobian dr_i / dx and dr_i / dy
                dr_dx = (x - xi) / di - (x - x_ref) / d0
                dr_dy = (y - yi) / di - (y - y_ref) / d0
                J.append([dr_dx, dr_dy])
                
            residuals = np.array(residuals)
            J = np.array(J)
            
            # Normal equations (J^T * J) * delta = -J^T * r
            JtJ = J.T @ J
            # Levenberg-Marquardt damping
            damping = 1e-3 * np.trace(JtJ)
            H = JtJ + damping * np.eye(2)
            
            try:
                delta = np.linalg.solve(H, -J.T @ residuals)
            except np.linalg.LinAlgError:
                break
                
            pos += delta
            
            if np.linalg.norm(delta) < tol:
                success = True
                try:
                    cov = np.linalg.inv(JtJ + 1e-6 * np.eye(2)) * (np.sum(residuals**2) / max(1, n_sensors - 3))
                except np.linalg.LinAlgError:
                    pass
                break
                
        # Calculate uncertainty ellipse parameters
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-4)
        semi_major = np.sqrt(eigvals[1]) * 2.0 # 2-sigma (~95% confidence)
        semi_minor = np.sqrt(eigvals[0]) * 2.0
        angle_deg = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        
        return {
            "x_mm": float(pos[0]),
            "y_mm": float(pos[1]),
            "uncertainty_semi_major_mm": float(semi_major),
            "uncertainty_semi_minor_mm": float(semi_minor),
            "uncertainty_angle_deg": float(angle_deg),
            "covariance": cov.tolist(),
            "converged": success,
        }
