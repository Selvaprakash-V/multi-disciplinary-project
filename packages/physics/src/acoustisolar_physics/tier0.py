"""
Tier-0 analytical simulator for guided-wave acoustic emission in a framed PV module.

Frequency-domain modal Green's-function model:

    v_r(f) = S(f) * sum_{m in {A0,S0}} A_m(f) * sum_{images} R^n * exp(-(alpha_m + i k_m) r) / sqrt(r)

* A0/S0 phase velocity, group velocity and attenuation come from the corrected multilayer
  SAFE solver for the configured stack and temperature (``LaminateDispersionModel``).
* Modal excitation depends on the source physics: in-plane dipole in the silicon (cracks),
  normal force on the glass (rain, hail, taps, rattle), in-plane force at the frame
  (thermal stick-slip).  Hail therefore excites mostly A0 and cracks relatively more S0.
* The receiver couples to each mode through its out-of-plane surface displacement (PZT
  discs are mainly out-of-plane sensitive) and the disc aperture 2 J1(ka)/(ka).
* Frame / edge reflections are included with the image-source method up to
  ``max_reflection_order`` per axis with reflection coefficient ``edge_reflection_coeff``.
* FFTs are zero-padded (no circular wrap-around) and the source can be placed at
  ``onset_s`` so that the record contains a noise-only pre-trigger segment.
* EMI is not acoustic: it is injected simultaneously (zero TDOA) on all channels.

Absolute amplitude is anchored so that a unit-energy crack 0.5 m from a sensor gives a
peak surface velocity of ``reference_velocity_m_s`` (default 2e-4 m/s, i.e. ~3 mV from a
15 V/(m/s) sensor ≈ 70 dB_AE, mid-range for brittle-fracture AE).  [CALIBRATE] against
pencil-lead breaks.

A causality mute zeroes each channel before onset + r_direct / v_max: the composite
spectral model leaves a residual numerical precursor of about -50 dB, which would otherwise
bias onset pickers on high-SNR or clipped events.  Pruned image sources: contributions that
cannot arrive inside the record or would wrap past the FFT length are skipped.
"""

from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from acoustisolar_schemas.source import SourceClass, SourceEventConfig

from .laminate import LaminateDispersionModel, LayerSpec, layers_from_module_config
from .source_library import COUPLING_ELECTRICAL, COUPLING_INPLANE_DIPOLE, SourceLibrary, source_coupling


@lru_cache(maxsize=16)
def _cached_laminate(layer_key: Tuple[Tuple, ...]) -> LaminateDispersionModel:
    return LaminateDispersionModel([LayerSpec(*lk) for lk in layer_key])


def laminate_for_module(module_config: ModuleTwinConfig) -> LaminateDispersionModel:
    """Laminate model for a module stack (cached: building it costs a few seconds)."""
    key = tuple(
        (l.name, l.thickness_m, l.density, l.youngs_modulus_pa, l.poissons_ratio, l.loss_factor, l.viscoelastic)
        for l in layers_from_module_config(module_config)
    )
    return _cached_laminate(key)


def minimum_phase(log_mag: np.ndarray, n_fft: int) -> np.ndarray:
    """Minimum-phase phase spectrum (rfft grid) for a given log-magnitude, via the real
    cepstrum.  Applying |A| exp(i phi_min) instead of the real gain |A| keeps the magnitude
    but makes the response causal.  For attenuation exp(-alpha(f) r) this is the
    Kramers-Kronig-consistent dispersion term."""
    c = np.fft.irfft(log_mag, n=n_fft)
    fold = np.zeros(n_fft)
    fold[0] = 1.0
    fold[1 : n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    return np.imag(np.fft.rfft(c * fold, n=n_fft))


def _aperture(k: np.ndarray, radius_m: float) -> np.ndarray:
    """Circular-disc aperture response 2 J1(ka)/(ka)."""
    from scipy.special import j1

    ka = np.maximum(np.abs(k) * radius_m, 1e-9)
    return 2.0 * j1(ka) / ka


class Tier0Simulator:
    """Tier-0 analytical multimodal simulator (see module docstring)."""

    def __init__(
        self,
        module_config: Optional[ModuleTwinConfig] = None,
        sensor_config: Optional[SensorArrayConfig] = None,
        sample_rate_hz: float = 1_000_000.0,
        n_samples: int = 2560,
        max_reflection_order: int = 2,
        edge_reflection_coeff: float = 0.5,
        reference_velocity_m_s: float = 2.0e-4,
        laminate_model: Optional[LaminateDispersionModel] = None,
    ):
        self.module = module_config or ModuleTwinConfig()
        self.sensor_config = sensor_config or SensorArrayConfig.for_module(
            self.module.width_mm, self.module.length_mm
        )
        self.sample_rate = sample_rate_hz
        self.n_samples = n_samples
        self.dt = 1.0 / sample_rate_hz
        self.max_reflection_order = max_reflection_order
        self.edge_reflection_coeff = edge_reflection_coeff
        self.laminate = laminate_model or laminate_for_module(self.module)
        self.source_lib = SourceLibrary(sample_rate_hz=sample_rate_hz, n_samples=n_samples)
        self.aperture_radius_m = 0.5 * self.sensor_config.disc.diameter_mm * 1.0e-3

        # Zero-padded FFT length (avoids circular wrap-around of slow, low-frequency A0)
        self.n_fft = int(2 ** np.ceil(np.log2(2 * n_samples)))
        self.freqs = np.fft.rfftfreq(self.n_fft, d=self.dt)
        self._props_cache: Dict[float, Dict] = {}
        self._coupling_norm: Dict[str, float] = {}

        # Absolute amplitude anchor
        self._amp_scale = 1.0
        self._amp_scale = self._calibrate_amplitude(reference_velocity_m_s)

    # ------------------------------------------------------------------ helpers
    def default_temperature(self) -> float:
        for layer in self.module.stack:
            if layer.material_model == "viscoelastic_prony":
                return float(layer.temperature_c)
        return float(self.module.stack[0].temperature_c)

    def _mode_props(self, temp_c: float) -> Dict:
        key = round(float(temp_c), 1)
        if key not in self._props_cache:
            if len(self._props_cache) > 64:
                self._props_cache.clear()
            self._props_cache[key] = self.laminate.evaluate(self.freqs, key)
        return self._props_cache[key]

    def _image_sources(self, x_m: float, y_m: float):
        """Image-source positions and reflection counts for the rectangular module."""
        W = self.module.width_mm * 1.0e-3
        L = self.module.length_mm * 1.0e-3
        N = self.max_reflection_order
        xs, nx = [], []
        for m in range(-N, N + 1):
            xs.append(2 * m * W + x_m)
            nx.append(abs(2 * m))
            xs.append(2 * m * W - x_m)
            nx.append(abs(2 * m - 1))
        ys, ny = [], []
        for m in range(-N, N + 1):
            ys.append(2 * m * L + y_m)
            ny.append(abs(2 * m))
            ys.append(2 * m * L - y_m)
            ny.append(abs(2 * m - 1))
        pts, orders = [], []
        for xi, ni in zip(xs, nx):
            for yi, nj in zip(ys, ny):
                if ni <= N and nj <= N:
                    pts.append((xi, yi))
                    orders.append(ni + nj)
        return np.asarray(pts), np.asarray(orders)

    def _raw_modal_amplitudes(self, props: Dict, coupling: str) -> Dict[str, np.ndarray]:
        omega = 2.0 * np.pi * self.freqs
        amps = {}
        for mode in ("A0", "S0"):
            p = props[mode]
            k = omega / np.maximum(p["cp"], 1.0)
            src = p[coupling] if coupling in p else p["normal_force_top"]
            # far-field 2D modal Green's function ~ phi_s phi_r / (c_g sqrt(k));  x omega -> velocity
            amp = src * p["receiver"] * _aperture(k, self.aperture_radius_m)
            amp = amp * omega / (np.maximum(p["cg"], 1.0) * np.sqrt(np.maximum(k, 1e-6)))
            amps[mode] = amp
        return amps

    def _modal_amplitudes(self, props: Dict, coupling: str) -> Dict[str, np.ndarray]:
        """Modal amplitudes with each coupling type normalised at 150 kHz / 25 C.

        Force-type and moment-type sources have different units; normalising keeps each
        source's A0/S0 *ratio* (the physics) and leaves the relative level to the source
        library's family multipliers.
        """
        if coupling not in self._coupling_norm:
            ref = self._mode_props(25.0)
            i = int(np.argmin(np.abs(self.freqs - 150.0e3)))
            a_ref = self._raw_modal_amplitudes(ref, COUPLING_INPLANE_DIPOLE)
            a_c = self._raw_modal_amplitudes(ref, coupling)
            num = abs(a_ref["A0"][i]) + abs(a_ref["S0"][i])
            den = abs(a_c["A0"][i]) + abs(a_c["S0"][i])
            self._coupling_norm[coupling] = num / max(den, 1e-30)
        norm = self._coupling_norm[coupling]
        raw = self._raw_modal_amplitudes(props, coupling)
        out = {}
        for m, v in raw.items():
            mag = np.abs(v) * norm
            log_mag = np.log(np.maximum(mag, 1e-3 * max(float(np.max(mag)), 1e-30)))
            out[m] = mag * np.exp(1j * minimum_phase(log_mag, self.n_fft))
        return out

    def _hp_taper(self) -> np.ndarray:
        """Minimum-phase 15 kHz high-pass taper (causal; a real-valued taper would leak
        energy ahead of the arrival)."""
        if not hasattr(self, "_hp_cache"):
            f_safe = np.maximum(self.freqs, 1.0)
            mag = 1.0 / (1.0 + (15.0e3 / f_safe) ** 4)
            log_mag = np.log(np.maximum(mag, 1e-12))
            self._hp_cache = mag * np.exp(1j * minimum_phase(log_mag, self.n_fft))
        return self._hp_cache

    def _attenuation_phase(self, props: Dict) -> Dict[str, np.ndarray]:
        """Kramers-Kronig phase per metre for each mode's attenuation alpha(f)."""
        return {m: minimum_phase(-props[m]["alpha"], self.n_fft) for m in ("A0", "S0")}

    def _calibrate_amplitude(self, target_velocity: float) -> float:
        cfg = SourceEventConfig(
            source_class=SourceClass.CRACK_INITIATION,
            x_mm=self.module.width_mm / 2.0,
            y_mm=self.module.length_mm / 2.0,
            energy_arbitrary=1.0,
            rise_time_us=1.5,
        )
        # single sensor at 0.5 m, no reflections, deterministic carrier
        saved = self.max_reflection_order
        self.max_reflection_order = 0
        sensor = [(cfg.x_mm + 500.0, cfg.y_mm)]
        w = self._propagate(cfg, 25.0, np.random.default_rng(0), 0.0, sensor, sens_spread=0.0)
        self.max_reflection_order = saved
        peak = float(np.max(np.abs(w)))
        return target_velocity / max(peak, 1e-30)

    # ------------------------------------------------------------------ core
    def _propagate(self, source, temp_c, rng, onset_s, sensors_mm, sens_spread=0.03) -> np.ndarray:
        n, n_fft = self.n_samples, self.n_fft
        s_t = self.source_lib.generate_source_signal(source, rng=rng)
        # Smooth 15 kHz high-pass taper: content below the AE band (A0 at a few kHz travels
        # at a few hundred m/s) would otherwise exceed the FFT length and wrap around.
        # The sensor/preamp chain (50-400 kHz) removes this band anyway.
        S_f = np.fft.rfft(s_t, n=n_fft) * np.exp(-2j * np.pi * self.freqs * onset_s) * self._hp_taper()
        coupling = source_coupling(source)
        out = np.zeros((len(sensors_mm), n))

        if coupling == COUPLING_ELECTRICAL:
            # Electrical interference: identical, simultaneous on every channel
            common = np.fft.irfft(S_f, n=n_fft)[:n] * self._amp_scale * 0.5
            for ch in range(len(sensors_mm)):
                out[ch] = common
            return out

        props = self._mode_props(temp_c)
        amps = self._modal_amplitudes(props, coupling)
        att_phase = self._attenuation_phase(props)
        omega = 2.0 * np.pi * self.freqs
        images, orders = self._image_sources(source.x_mm * 1.0e-3, source.y_mm * 1.0e-3)
        refl = self.edge_reflection_coeff ** orders
        r0 = 0.02  # near-field regularisation (m)
        t_record = self.n_samples * self.dt
        t_fft = self.n_fft * self.dt
        band = self.freqs >= 20.0e3
        # Per-mode range limits: an image contribution is kept only if (a) it can arrive
        # inside the record and (b) its slowest in-band energy cannot wrap past the FFT
        # length (slow A0 from distant images otherwise leaks into the pre-trigger).
        r_limits = {}
        for mode in ("A0", "S0"):
            v_fast = float(np.max(props[mode]["cp"][band]))
            v_slow = float(np.min(props[mode]["cg"][band]))
            r_record = v_fast * max(t_record - onset_s + 1.0e-4, 1.0e-4)
            r_wrap = v_slow * max(t_fft - onset_s - 2.0e-4, 1.0e-4)
            r_limits[mode] = min(r_record, r_wrap) + 0.05

        for ch, (sx, sy) in enumerate(sensors_mm):
            r_all = np.hypot(images[:, 0] - sx * 1.0e-3, images[:, 1] - sy * 1.0e-3)
            weight_all = refl / np.sqrt(np.maximum(r_all, r0))
            H = np.zeros(len(self.freqs), dtype=np.complex128)
            for mode in ("A0", "S0"):
                p = props[mode]
                alpha_min = float(np.min(p["alpha"][band]))
                keep = (r_all <= r_limits[mode]) & (
                    weight_all * np.exp(-alpha_min * r_all) >= 1.0e-3 * np.max(weight_all)
                )
                r, weight = r_all[keep], weight_all[keep]
                k = omega / np.maximum(p["cp"], 1.0)
                # attenuation with its causal (Kramers-Kronig) phase, then propagation phase
                prop = np.exp(-np.outer(r, p["alpha"] - 1j * att_phase[mode] + 1j * k))  # (n_img, n_f)
                H += amps[mode] * (weight @ prop)
            gain = 1.0 + (rng.normal(0, sens_spread) if sens_spread > 0 else 0.0)
            out[ch] = np.fft.irfft(S_f * H, n=n_fft)[:n] * gain * self._amp_scale
            # Causality mute: nothing can arrive before onset + r_direct / v_max.  The
            # spectral model leaves a residual numerical precursor (about -50 dB), which
            # is visible on high-SNR or clipped events and would bias onset pickers.
            v_max = self.laminate.group_velocity_limit("S0")
            r_direct = float(np.hypot(source.x_mm - sx, source.y_mm - sy)) * 1.0e-3
            i0 = int(np.floor((onset_s + r_direct / v_max - 2.0e-6) / self.dt))
            if i0 > 0:
                taper = 4
                out[ch, : max(0, i0 - taper)] = 0.0
                lo = max(0, i0 - taper)
                if i0 > lo:
                    out[ch, lo:i0] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, i0 - lo)))
        return out

    def simulate(
        self,
        source: SourceEventConfig,
        temperature_c: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        onset_s: float = 0.0,
        sensor_positions_mm=None,
    ) -> np.ndarray:
        """Surface velocity (m/s) at every sensor: array of shape (n_channels, n_samples).

        ``sensor_positions_mm`` overrides the configured (nominal) positions, e.g. to model
        installation tolerance while the localiser keeps using the nominal positions.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        temp = self.default_temperature() if temperature_c is None else float(temperature_c)
        positions = self.sensor_config.positions_mm if sensor_positions_mm is None else sensor_positions_mm
        return self._propagate(source, temp, rng, onset_s, positions)

    def s0_velocity(self, temperature_c: Optional[float] = None) -> float:
        """S0 phase velocity of the laminate at 150 kHz (for TDOA localisation)."""
        temp = self.default_temperature() if temperature_c is None else float(temperature_c)
        return self.laminate.s0_velocity(temp)

    def a0_front_velocity(self, temperature_c: Optional[float] = None) -> float:
        """Fastest A0 group velocity in the 50-400 kHz sensor band: the speed of the A0
        *front* that a threshold picker lands on when the S0 head wave is buried."""
        temp = self.default_temperature() if temperature_c is None else float(temperature_c)
        props = self._mode_props(temp)
        band = (self.freqs >= 50.0e3) & (self.freqs <= 400.0e3)
        return float(np.max(props["A0"]["cg"][band]))

    def mode_coupling_ratio(self, source: SourceEventConfig, temperature_c: float = 25.0) -> float:
        """Received S0/A0 amplitude ratio at 150 kHz for a source type (diagnostic)."""
        props = self._mode_props(temperature_c)
        amps = self._modal_amplitudes(props, source_coupling(source))
        i = int(np.argmin(np.abs(self.freqs - 150.0e3)))
        return float(abs(amps["S0"][i]) / max(abs(amps["A0"][i]), 1e-30))

    @property
    def is_crack_coupling(self) -> str:
        return COUPLING_INPLANE_DIPOLE
