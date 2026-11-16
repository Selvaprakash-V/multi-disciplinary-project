"""
Tier-1 2D scalar wave solver (finite differences, leap-frog).

    u_tt = div( c(x, y)^2 grad u ) - gamma(x, y) u_t + s(x, y, t)

Scope and limits (be explicit — earlier documentation over-claimed):
* This is a **scalar** (membrane / acoustic-analogue) model with one wave speed per cell.
  It has no A0/S0 mode pair, no Lamb-wave dispersion and no mode conversion.  It is used
  to study geometry effects (edge reflections, heterogeneity, sensor layout) on timing,
  not as an elastodynamic plate model.  PRD Tier-2 (SPECFEM / FEniCSx / COMSOL) remains
  the route to true elastodynamics.
* The operator is written in conservative (flux) form with face-averaged c^2, so it is
  self-adjoint and discrete reciprocity holds for heterogeneous media too.
* ``boundary="absorbing"``: polynomial sponge of ``pml_thickness_cells`` cells (not a
  true PML).  Sensors placed inside the sponge are rejected — their signals would be
  damped and distorted.
* ``boundary="reflecting"``: zero-normal-gradient (free) edges, i.e. a framed module
  with strong edge reflections.
* Accuracy needs >= ~10 grid points per wavelength (``points_per_wavelength``).
"""

from typing import Dict, Optional, Tuple

import numpy as np


class Tier1WaveSolver:
    """2D scalar wave solver with absorbing (sponge) or reflecting boundaries."""

    def __init__(
        self,
        width_m: float = 1.0,
        length_m: float = 1.7,
        dx_m: float = 0.010,
        dt_s: float = 1.0e-6,
        c_speed_m_s: float = 5300.0,
        pml_thickness_cells: int = 10,
        boundary: str = "absorbing",
        c_field: Optional[np.ndarray] = None,
    ):
        self.width_m = width_m
        self.length_m = length_m
        self.dx = dx_m
        self.dt = dt_s
        self.c = c_speed_m_s
        self.boundary = boundary
        self.pml_cells = pml_thickness_cells if boundary == "absorbing" else 0

        self.nx = int(np.round(width_m / dx_m))
        self.ny = int(np.round(length_m / dx_m))

        self.c_field = (
            np.full((self.nx, self.ny), float(c_speed_m_s)) if c_field is None else np.asarray(c_field, float)
        )
        if self.c_field.shape != (self.nx, self.ny):
            raise ValueError(f"c_field shape {self.c_field.shape} != grid {(self.nx, self.ny)}")

        c_max = float(np.max(self.c_field))
        self.courant_number = c_max * self.dt / self.dx
        assert self.courant_number <= 0.707, f"CFL condition violated: Courant {self.courant_number} > 0.707"

        self.gamma = np.zeros((self.nx, self.ny), dtype=np.float64)
        if self.boundary == "absorbing" and self.pml_cells > 0:
            # Damping strength sized so a wave crossing the sponge and back is attenuated by
            # ~1e-3 (amplitude decays as exp(-gamma t / 2); quadratic profile mean = max / 3).
            # The previous fixed 5e4 1/s attenuated only ~7 % and the layer reflected ~74 %.
            layer = self.pml_cells * self.dx
            gamma_max = 3.0 * 2.0 * np.log(1.0e3) * c_max / (2.0 * layer)
            for i in range(self.nx):
                d = min(i, self.nx - 1 - i)
                if d < self.pml_cells:
                    self.gamma[i, :] += gamma_max * ((self.pml_cells - d) / self.pml_cells) ** 2
            for j in range(self.ny):
                d = min(j, self.ny - 1 - j)
                if d < self.pml_cells:
                    self.gamma[:, j] += gamma_max * ((self.pml_cells - d) / self.pml_cells) ** 2

    # ------------------------------------------------------------------ helpers
    def set_c_field(self, c_field: np.ndarray) -> None:
        c_field = np.asarray(c_field, float)
        if c_field.shape != (self.nx, self.ny):
            raise ValueError("c_field shape mismatch")
        self.c_field = c_field
        self.courant_number = float(np.max(c_field)) * self.dt / self.dx
        assert self.courant_number <= 0.707, f"CFL condition violated: Courant {self.courant_number} > 0.707"

    def points_per_wavelength(self, freq_hz: float) -> float:
        return float(np.min(self.c_field)) / freq_hz / self.dx

    def absorber_margin_m(self) -> float:
        return self.pml_cells * self.dx

    def _check_sensor(self, x: float, y: float) -> None:
        m = self.absorber_margin_m()
        if self.boundary == "absorbing" and (
            x < m or y < m or x > self.width_m - m or y > self.length_m - m
        ):
            raise ValueError(
                f"Sensor/source at ({x:.3f}, {y:.3f}) m lies inside the {m * 1e3:.0f} mm absorbing "
                "sponge; move it inward or use boundary='reflecting'."
            )

    def _index(self, x: float, y: float) -> Tuple[int, int]:
        """Cell containing (x, y).  Cells are centred at (i + 0.5) dx so that the zero-flux
        walls lie exactly at 0 and nx*dx (= width)."""
        return (
            int(np.clip(np.floor(x / self.dx), 0, self.nx - 1)),
            int(np.clip(np.floor(y / self.dx), 0, self.ny - 1)),
        )

    def _bilinear(self, x: float, y: float):
        """Indices and weights for bilinear read-out at (x, y) from cell-centred values."""
        fx = np.clip(x / self.dx - 0.5, 0.0, self.nx - 1.0)
        fy = np.clip(y / self.dx - 0.5, 0.0, self.ny - 1.0)
        i0, j0 = int(np.floor(fx)), int(np.floor(fy))
        i1, j1 = min(i0 + 1, self.nx - 1), min(j0 + 1, self.ny - 1)
        ax, ay = fx - i0, fy - j0
        return [(i0, j0, (1 - ax) * (1 - ay)), (i1, j0, ax * (1 - ay)), (i0, j1, (1 - ax) * ay), (i1, j1, ax * ay)]

    def _operator(self, u: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
        """Conservative div(c^2 grad u) / c_ref^2-free form (multiplied by dt^2/dx^2 outside)."""
        out = np.zeros_like(u)
        fx = kx * (u[1:, :] - u[:-1, :])  # flux on x-faces
        fy = ky * (u[:, 1:] - u[:, :-1])  # flux on y-faces
        out[:-1, :] += fx
        out[1:, :] -= fx
        out[:, :-1] += fy
        out[:, 1:] -= fy
        return out  # zero-flux (free) outer boundary built in

    # ------------------------------------------------------------------ run
    def run_simulation(
        self,
        source_x_m: float,
        source_y_m: float,
        source_signal: np.ndarray,
        sensor_locations_m: list,
        source_sigma_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Time-step the wavefield; returns (sensor_waveforms (n_sensors, n_steps), peak energy).

        ``source_sigma_m`` spreads the source over a normalised Gaussian footprint instead of
        a single cell (a point source is a spatial delta, which limits measured convergence
        to roughly first order; a smooth source exposes the scheme's 2nd-order accuracy).
        """
        self._check_sensor(source_x_m, source_y_m)
        for sx, sy in sensor_locations_m:
            self._check_sensor(sx, sy)

        n_steps = len(source_signal)
        sensor_waves = np.zeros((len(sensor_locations_m), n_steps), dtype=np.float64)
        # Point sources are injected with the same bilinear weights used for read-out, so the
        # source and receiver operators are adjoint and reciprocity holds exactly.
        source_taps = self._bilinear(source_x_m, source_y_m)
        readout = [self._bilinear(x, y) for x, y in sensor_locations_m]
        footprint = None
        if source_sigma_m is not None:
            gx = ((np.arange(self.nx) + 0.5) * self.dx - source_x_m)[:, None]
            gy = ((np.arange(self.ny) + 0.5) * self.dx - source_y_m)[None, :]
            footprint = np.exp(-(gx**2 + gy**2) / (2.0 * source_sigma_m**2))
            footprint /= footprint.sum()

        c2 = self.c_field**2
        kx = 0.5 * (c2[1:, :] + c2[:-1, :]) * (self.dt / self.dx) ** 2
        ky = 0.5 * (c2[:, 1:] + c2[:, :-1]) * (self.dt / self.dx) ** 2

        u_prev = np.zeros((self.nx, self.ny))
        u_curr = np.zeros((self.nx, self.ny))
        g = 0.5 * self.gamma * self.dt
        damp = 1.0 / (1.0 + g)
        peak_energy = 0.0

        for t in range(n_steps):
            # sample t is the field at time t*dt (u_curr), consistent with source_signal[t]
            for ch, taps in enumerate(readout):
                sensor_waves[ch, t] = sum(w * u_curr[a, b] for a, b, w in taps)
            lap = self._operator(u_curr, kx, ky)
            rhs = 2.0 * u_curr - (1.0 - g) * u_prev + lap
            if footprint is None:
                for a, b, w in source_taps:
                    rhs[a, b] += w * source_signal[t] * self.dt**2
            else:
                rhs += footprint * (source_signal[t] * self.dt**2)
            u_next = damp * rhs
            e = float(np.sum(u_next**2))
            peak_energy = max(peak_energy, e)
            u_prev, u_curr = u_curr, u_next

        return sensor_waves, peak_energy

    # ------------------------------------------------------------------ comparison
    def compare_fidelity_gap(self, tier0_waves: np.ndarray, tier1_waves: np.ndarray) -> Dict[str, float]:
        """Arrival-time, spectral-centroid and correlation agreement between two wavefields."""
        n_channels = min(tier0_waves.shape[0], tier1_waves.shape[0])
        n = min(tier0_waves.shape[1], tier1_waves.shape[1])
        arrival_errs, spec_errs, corrs = [], [], []
        for ch in range(n_channels):
            t0 = tier0_waves[ch, :n]
            t1 = tier1_waves[ch, :n]
            norm_t0 = t0 / max(1e-30, np.max(np.abs(t0)))
            norm_t1 = t1 / max(1e-30, np.max(np.abs(t1)))
            a0 = int(np.argmax(np.abs(norm_t0) > 0.1))
            a1 = int(np.argmax(np.abs(norm_t1) > 0.1))
            arrival_errs.append(abs(a0 - a1) / max(1.0, float(a0)))
            f_axis = np.fft.rfftfreq(n)
            s0 = np.abs(np.fft.rfft(norm_t0))
            s1 = np.abs(np.fft.rfft(norm_t1))
            c0 = np.sum(f_axis * s0) / max(1e-12, np.sum(s0))
            c1 = np.sum(f_axis * s1) / max(1e-12, np.sum(s1))
            spec_errs.append(abs(c0 - c1) / max(1e-6, c0))
            cc = np.corrcoef(norm_t0, norm_t1)[0, 1]
            corrs.append(cc if not np.isnan(cc) else 0.0)
        return {
            "arrival_time_error_pct": float(np.mean(arrival_errs) * 100.0),
            "spectral_centroid_error_pct": float(np.mean(spec_errs) * 100.0),
            "envelope_correlation": float(np.mean(corrs)),
            "meets_target_arrival": bool(np.mean(arrival_errs) < 0.05),
            "meets_target_spectral": bool(np.mean(spec_errs) < 0.20),
        }
