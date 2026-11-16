"""
Source library: source time functions s(t) for crack events and environmental sources,
plus the physical metadata Tier-0 needs (force direction / modal coupling, whether the
source propagates acoustically, where on the module it can occur).

Waveform shapes and relative levels are [UNSOURCED] engineering priors: no peer-reviewed
AE spectra exist for framed glass/c-Si modules (see docs/research/LITERATURE_REVIEW.md).
They must be replaced or re-weighted with LAB-1 / NOISE-1 measurements.
"""

from typing import Optional

import numpy as np

from acoustisolar_schemas.source import CrackType, NoiseClass, SourceClass, SourceEventConfig

# Modal coupling of each source family (see laminate.SOURCE_COUPLINGS)
COUPLING_INPLANE_DIPOLE = "inplane_dipole_si"   # fracture inside the silicon layer
COUPLING_NORMAL_FORCE = "normal_force_top"      # impacts on the glass surface
COUPLING_INPLANE_FORCE = "inplane_force_glass"  # stick-slip at the frame
COUPLING_ELECTRICAL = "electrical"              # no acoustic propagation

# Relative source level multipliers (amplitude) per family — [UNSOURCED] priors.
_LEVEL = {
    NoiseClass.HAIL: 18.0,
    NoiseClass.RAIN_HAIL: 18.0,
    NoiseClass.RAIN: 0.35,
    NoiseClass.WIND_RATTLE: 0.6,
    NoiseClass.THERMAL_TICK: 0.6,
    NoiseClass.EMI_BURST: 1.0,
    NoiseClass.HANDLING_TAP: 4.0,
    NoiseClass.WHITE_NOISE: 0.2,
}


def source_coupling(config: SourceEventConfig) -> str:
    """Which modal-excitation model applies to this source."""
    if config.source_class in (
        SourceClass.CRACK_INITIATION,
        SourceClass.CRACK_GROWTH,
        SourceClass.CRACK_FRICTION,
    ):
        return COUPLING_INPLANE_DIPOLE
    if config.source_class == SourceClass.ACTIVE_PULSE:
        return COUPLING_NORMAL_FORCE
    nc = config.noise_class
    if nc == NoiseClass.EMI_BURST:
        return COUPLING_ELECTRICAL
    if nc == NoiseClass.THERMAL_TICK:
        return COUPLING_INPLANE_FORCE
    return COUPLING_NORMAL_FORCE


def occurs_at_frame(noise_class: NoiseClass) -> bool:
    """Sources physically located at the frame / mounting clamps."""
    return noise_class in (NoiseClass.THERMAL_TICK, NoiseClass.WIND_RATTLE)


class SourceLibrary:
    """Source time functions s(t) for crack and environmental sources (PRD §7.4)."""

    def __init__(self, sample_rate_hz: float = 1_000_000.0, n_samples: int = 2560):
        self.sample_rate = sample_rate_hz
        self.n_samples = n_samples
        self.dt = 1.0 / sample_rate_hz
        self.time = np.arange(n_samples) * self.dt

    # ------------------------------------------------------------------ primitives
    def _burst(self, t0: float, tau_r: float, f_c: float, phase: float = 0.0) -> np.ndarray:
        t_rel = np.maximum(self.time - t0, 0.0)
        env = (t_rel / tau_r) * np.exp(-t_rel / tau_r)
        return np.where(self.time >= t0, env * np.sin(2.0 * np.pi * f_c * t_rel + phase), 0.0)

    def _contact_force(self, t0: float, contact_s: float) -> np.ndarray:
        """Half-sine contact force (Hertz-like impact) of duration ``contact_s``."""
        t_rel = self.time - t0
        inside = (t_rel >= 0) & (t_rel <= contact_s)
        return np.where(inside, np.sin(np.pi * np.clip(t_rel, 0, None) / contact_s), 0.0)

    # ------------------------------------------------------------------ public
    def generate_source_signal(
        self, config: SourceEventConfig, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Source time function, peak-normalised to ``config.energy_arbitrary`` × family level."""
        if rng is None:
            rng = np.random.default_rng(42)
        t = self.time
        n = self.n_samples
        signal = np.zeros(n, dtype=np.float64)
        level = 1.0

        if config.source_class == SourceClass.CRACK_INITIATION:
            # Single cleavage burst: fast rise, short ring-down [UNSOURCED spectral prior]
            tau_r = max(config.rise_time_us * 1.0e-6, 0.5e-6)
            signal = self._burst(0.0, tau_r, rng.uniform(150.0e3, 280.0e3), rng.uniform(0, 2 * np.pi))

        elif config.source_class == SourceClass.CRACK_GROWTH:
            # Clustered micro-bursts with escalating energy
            for b in range(int(rng.integers(2, 5))):
                t_off = rng.uniform(10.0e-6, 150.0e-6) * b
                signal += (1.0 + 0.5 * b) * self._burst(
                    t_off, rng.uniform(0.8e-6, 2.5e-6), rng.uniform(160.0e3, 300.0e3)
                )

        elif config.source_class == SourceClass.CRACK_FRICTION:
            # Rubbing of existing crack faces: low-level repetitive stick-slip packets
            f_c = rng.uniform(80.0e3, 140.0e3)
            for _ in range(int(rng.integers(4, 8))):
                t_p = rng.uniform(0, t[-1] * 0.5)
                tau = rng.uniform(5.0e-6, 15.0e-6)
                t_rel = np.maximum(t - t_p, 0.0)
                pkt = np.exp(-((t_rel / tau) ** 2)) * np.sin(2.0 * np.pi * f_c * t_rel)
                signal += np.where(t >= t_p, pkt, 0.0)
            level = 0.3

        elif config.source_class == SourceClass.NOISE:
            signal = self._generate_noise_class(config.noise_class, rng)
            level = _LEVEL.get(config.noise_class, 1.0)

        elif config.source_class == SourceClass.ACTIVE_PULSE:
            f_pulse = 200.0e3
            duration = 5 / f_pulse
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / duration))
            signal = np.where(t <= duration, window * np.sin(2.0 * np.pi * f_pulse * t), 0.0)

        if config.crack_type == CrackType.PARALLEL_TO_BUSBAR:
            theta = np.radians(config.orientation_deg)
            signal *= 0.7 + 0.3 * np.abs(np.cos(theta))

        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * config.energy_arbitrary * level
        return signal

    def _generate_noise_class(self, noise_class: NoiseClass, rng: np.random.Generator) -> np.ndarray:
        t = self.time
        n = self.n_samples

        if noise_class in (NoiseClass.HAIL, NoiseClass.RAIN_HAIL):
            # One large ice impact: long contact (tens of us) -> energy mostly below the AE
            # band, plus a short high-frequency contact transient from local glass crushing.
            contact = rng.uniform(40.0e-6, 150.0e-6)
            force = self._contact_force(0.0, contact)
            transient = 0.15 * self._burst(0.0, rng.uniform(1.0e-6, 3.0e-6), rng.uniform(80.0e3, 250.0e3))
            return force + transient

        if noise_class == NoiseClass.RAIN:
            # Poisson train of small drop impacts (short contacts, random amplitudes)
            sig = np.zeros(n)
            rate = rng.uniform(4.0e3, 20.0e3)  # drops per second seen in the window
            n_drops = max(1, rng.poisson(rate * t[-1]))
            for _ in range(n_drops):
                t0 = rng.uniform(0, t[-1])
                sig += rng.lognormal(0.0, 0.6) * self._contact_force(t0, rng.uniform(5.0e-6, 20.0e-6))
            return sig

        if noise_class == NoiseClass.WIND_RATTLE:
            # Repeated light knocks of the laminate against the frame / clamp
            sig = np.zeros(n)
            for _ in range(int(rng.integers(2, 6))):
                t0 = rng.uniform(0, t[-1] * 0.7)
                sig += rng.uniform(0.3, 1.0) * self._contact_force(t0, rng.uniform(20.0e-6, 60.0e-6))
            return sig

        if noise_class == NoiseClass.THERMAL_TICK:
            # Stick-slip of laminate in the frame: abrupt in-plane slip, broadband
            t_click = rng.uniform(0.0, 50.0e-6)
            return self._burst(t_click, rng.uniform(1.5e-6, 5.0e-6), rng.uniform(80.0e3, 160.0e3))

        if noise_class == NoiseClass.EMI_BURST:
            # Electrical glitch: injected simultaneously on all channels by Tier-0
            t_emi = rng.uniform(0.0, 50.0e-6)
            tau = rng.uniform(0.5e-6, 1.5e-6)
            t_rel = np.maximum(t - t_emi, 0.0)
            burst = np.exp(-(((t_rel - 2 * tau) / tau) ** 2)) * np.sin(
                2.0 * np.pi * rng.uniform(300.0e3, 450.0e3) * t_rel
            )
            return np.where(t >= t_emi, burst, 0.0)

        if noise_class == NoiseClass.HANDLING_TAP:
            return self._contact_force(0.0, rng.uniform(15.0e-6, 40.0e-6))

        # WHITE_NOISE
        return rng.normal(0, 0.2, n)

    # ------------------------------------------------------------------ ambient
    def ambient_background(
        self, n_channels: int, rng: np.random.Generator, level: float = 1.0
    ) -> np.ndarray:
        """Continuous ambient mechanical background (in source units, pre-propagation-free).

        Components [UNSOURCED, CALIBRATE with field logging]:
        * broadband structural noise (wind turbulence, distant traffic), 1/f-shaped
        * inverter switching harmonics in the 16-32 kHz band (largely removed by the
          50 kHz high-pass but leaks through finite filter slopes)
        """
        n = self.n_samples
        out = np.zeros((n_channels, n))
        freqs = np.fft.rfftfreq(n, d=self.dt)
        shaping = 1.0 / np.sqrt(np.maximum(freqs, 1.0e3) / 1.0e3)
        f_sw = rng.uniform(16.0e3, 32.0e3)
        for ch in range(n_channels):
            spec = (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))) * shaping
            x = np.fft.irfft(spec, n=n)
            x /= max(np.std(x), 1e-12)
            harmonics = sum(
                (0.5**h) * np.sin(2.0 * np.pi * h * f_sw * self.time + rng.uniform(0, 2 * np.pi))
                for h in (1, 2, 3)
            )
            out[ch] = level * (x + 0.5 * harmonics)
        return out
