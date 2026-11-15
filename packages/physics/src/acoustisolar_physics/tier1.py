import numpy as np
from typing import Dict, Any, Tuple, Optional
from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.source import SourceEventConfig
from .materials import GLASS_MATERIAL

class Tier1WaveSolver:
    """
    Tier-1 2D Elastodynamic Plate Wave Solver.
    Solves guided plate wave propagation on a numerical grid with sponge absorbing boundary layers (PML).
    
    Equations:
        rho * d^2 u / dt^2 = c^2 * nabla^2 u - gamma(x, y) * du / dt + s(x, y, t)
    where:
        c = c_S0 (plate mode speed)
        gamma(x, y) = absorbing boundary damping profile
    
    Verifications:
        - CFL stability condition: c * dt / dx <= 1 / sqrt(2)
        - Energy conservation without absorbing boundaries
        - Convergence under grid refinement
    """

    def __init__(
        self,
        width_m: float = 1.0,
        length_m: float = 1.7,
        dx_m: float = 0.010,           # 10 mm grid resolution
        dt_s: float = 1.0e-6,           # 1 us time step (1 MSPS)
        c_speed_m_s: float = 5300.0,    # S0 wave speed
        pml_thickness_cells: int = 10,
    ):
        self.width_m = width_m
        self.length_m = length_m
        self.dx = dx_m
        self.dt = dt_s
        self.c = c_speed_m_s
        self.pml_cells = pml_thickness_cells
        
        # Grid dimensions
        self.nx = int(np.round(width_m / dx_m))
        self.ny = int(np.round(length_m / dx_m))
        
        # CFL Courant stability number
        self.courant_number = self.c * self.dt / self.dx
        assert self.courant_number <= 0.707, f"CFL condition violated: Courant {self.courant_number} > 0.707"
        
        # Build absorbing sponge boundary damping matrix gamma(x, y)
        self.gamma = np.zeros((self.nx, self.ny), dtype=np.float64)
        gamma_max = 5.0e4
        for i in range(self.nx):
            dist_x = min(i, self.nx - 1 - i)
            if dist_x < self.pml_cells:
                self.gamma[i, :] += gamma_max * ((self.pml_cells - dist_x) / self.pml_cells)**2
        for j in range(self.ny):
            dist_y = min(j, self.ny - 1 - j)
            if dist_y < self.pml_cells:
                self.gamma[:, j] += gamma_max * ((self.pml_cells - dist_y) / self.pml_cells)**2

    def run_simulation(
        self,
        source_x_m: float,
        source_y_m: float,
        source_signal: np.ndarray,
        sensor_locations_m: list,
    ) -> Tuple[np.ndarray, float]:
        """
        Run time-stepping wave simulation.
        Returns:
            (sensor_waveforms: shape (n_sensors, n_steps), max_energy)
        """
        n_steps = len(source_signal)
        n_sensors = len(sensor_locations_m)
        sensor_waves = np.zeros((n_sensors, n_steps), dtype=np.float64)
        
        # Map physical coordinates to nearest grid cell indices
        src_i = int(np.clip(np.round(source_x_m / self.dx), 0, self.nx - 1))
        src_j = int(np.clip(np.round(source_y_m / self.dx), 0, self.ny - 1))
        
        sensor_indices = [
            (int(np.clip(np.round(sx / self.dx), 0, self.nx - 1)),
             int(np.clip(np.round(sy / self.dx), 0, self.ny - 1)))
            for sx, sy in sensor_locations_m
        ]
        
        # State fields: u_past, u_present, u_future
        u_prev = np.zeros((self.nx, self.ny), dtype=np.float64)
        u_curr = np.zeros((self.nx, self.ny), dtype=np.float64)
        u_next = np.zeros((self.nx, self.ny), dtype=np.float64)
        
        courant_sq = self.courant_number**2
        damp_factor = 1.0 / (1.0 + 0.5 * self.gamma * self.dt)
        
        total_energy = 0.0
        
        # Finite difference time stepping loop (leap-frog)
        for t in range(n_steps):
            # 5-point discrete Laplacian stencil with Dirichlet/sponge boundary
            laplacian = np.zeros_like(u_curr)
            laplacian[1:-1, 1:-1] = (
                u_curr[2:, 1:-1] + u_curr[:-2, 1:-1] +
                u_curr[1:-1, 2:] + u_curr[1:-1, :-2] -
                4.0 * u_curr[1:-1, 1:-1]
            )
            
            # Injection of point source
            force = source_signal[t]
            u_curr[src_i, src_j] += force * (self.dt**2)
            
            # Explicit update equation with absorbing damping
            u_next = damp_factor * (
                2.0 * u_curr - (1.0 - 0.5 * self.gamma * self.dt) * u_prev +
                courant_sq * laplacian
            )
            
            # Record sensor signals
            for ch, (si, sj) in enumerate(sensor_indices):
                sensor_waves[ch, t] = u_next[si, sj]
                
            # Track peak energy
            energy = float(np.sum(u_next**2))
            if energy > total_energy:
                total_energy = energy
                
            # Rotate buffers
            u_prev[:] = u_curr
            u_curr[:] = u_next
            
        return sensor_waves, total_energy

    def compare_fidelity_gap(
        self,
        tier0_waves: np.ndarray,
        tier1_waves: np.ndarray,
    ) -> Dict[str, float]:
        """
        Quantify fidelity gap between Tier-0 analytical and Tier-1 numerical wavefield (PRD 7.6).
        Metrics:
            - Arrival-time discrepancy (%)
            - Spectral centroid difference (%)
            - Cross-correlation envelope agreement
        """
        n_channels = min(tier0_waves.shape[0], tier1_waves.shape[0])
        arrival_errs = []
        spec_errs = []
        corrs = []
        
        for ch in range(n_channels):
            t0 = tier0_waves[ch]
            t1 = tier1_waves[ch]
            
            # Normalize signals
            norm_t0 = t0 / max(1e-12, np.max(np.abs(t0)))
            norm_t1 = t1 / max(1e-12, np.max(np.abs(t1)))
            
            # Arrival time via peak envelope
            arr_t0 = np.argmax(np.abs(norm_t0))
            arr_t1 = np.argmax(np.abs(norm_t1))
            rel_arr_err = abs(arr_t0 - arr_t1) / max(1.0, float(arr_t0))
            arrival_errs.append(rel_arr_err)
            
            # Spectral centroid difference
            f_axis = np.fft.rfftfreq(len(t0))
            s0 = np.abs(np.fft.rfft(norm_t0))
            s1 = np.abs(np.fft.rfft(norm_t1))
            cent0 = np.sum(f_axis * s0) / max(1e-12, np.sum(s0))
            cent1 = np.sum(f_axis * s1) / max(1e-12, np.sum(s1))
            rel_spec_err = abs(cent0 - cent1) / max(1e-6, cent0)
            spec_errs.append(rel_spec_err)
            
            # Cross correlation
            cc = np.corrcoef(norm_t0, norm_t1)[0, 1]
            corrs.append(cc if not np.isnan(cc) else 0.0)
            
        return {
            "arrival_time_error_pct": float(np.mean(arrival_errs) * 100.0),
            "spectral_centroid_error_pct": float(np.mean(spec_errs) * 100.0),
            "envelope_correlation": float(np.mean(corrs)),
            "meets_target_arrival": bool(np.mean(arrival_errs) < 0.05), # < 5% target
            "meets_target_spectral": bool(np.mean(spec_errs) < 0.20),   # < 20% target
        }

