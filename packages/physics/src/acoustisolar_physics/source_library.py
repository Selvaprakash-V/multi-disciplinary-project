import numpy as np
from typing import Tuple
from acoustisolar_schemas.source import SourceClass, CrackType, NoiseClass, SourceEventConfig

class SourceLibrary:
    """
    Source library producing source time functions s(t) and spectra S(omega)
    for crack events and environmental noise classes per PRD Section 7.4.
    """

    def __init__(self, sample_rate_hz: float = 1_000_000.0, n_samples: int = 2560):
        self.sample_rate = sample_rate_hz
        self.n_samples = n_samples
        self.dt = 1.0 / sample_rate_hz
        self.time = np.arange(n_samples) * self.dt

    def generate_source_signal(self, config: SourceEventConfig, rng: np.random.Generator = None) -> np.ndarray:
        """Generate normalized source excitation waveform s(t)."""
        if rng is None:
            rng = np.random.default_rng(42)

        t = self.time
        n = self.n_samples
        signal = np.zeros(n, dtype=np.float64)

        if config.source_class == SourceClass.CRACK_INITIATION:
            # Moment-tensor burst with rapid rise time tau_r and exponential ringdown
            tau_r = max(config.rise_time_us * 1.0e-6, 0.5e-6)
            f_center = rng.uniform(150.0e3, 280.0e3)
            # Source time function: t * exp(-t / tau_r) * sin(2*pi*f*t)
            envelope = (t / tau_r) * np.exp(-t / tau_r)
            carrier = np.sin(2.0 * np.pi * f_center * t + rng.uniform(0, 2*np.pi))
            signal = envelope * carrier * config.energy_arbitrary

        elif config.source_class == SourceClass.CRACK_GROWTH:
            # Clustered series of micro-bursts with escalating energy
            n_bursts = rng.integers(2, 5)
            for b in range(n_bursts):
                t_offset = rng.uniform(10.0e-6, 150.0e-6) * b
                tau_r = rng.uniform(0.8e-6, 2.5e-6)
                f_c = rng.uniform(160.0e3, 300.0e3)
                t_rel = np.maximum(t - t_offset, 0.0)
                mask = t >= t_offset
                amp = (1.0 + 0.5 * b) * config.energy_arbitrary
                burst = (t_rel / tau_r) * np.exp(-t_rel / tau_r) * np.sin(2.0 * np.pi * f_c * t_rel)
                signal += np.where(mask, burst * amp, 0.0)

        elif config.source_class == SourceClass.CRACK_FRICTION:
            # Low-amplitude repetitive stick-slip packets
            n_packets = rng.integers(4, 8)
            f_c = rng.uniform(80.0e3, 140.0e3)
            for p in range(n_packets):
                t_p = rng.uniform(0, t[-1] * 0.8)
                tau = rng.uniform(5.0e-6, 15.0e-6)
                t_rel = np.maximum(t - t_p, 0.0)
                mask = t >= t_p
                pkt = np.exp(-((t_rel) / tau)**2) * np.sin(2.0 * np.pi * f_c * t_rel)
                signal += np.where(mask, pkt * 0.3 * config.energy_arbitrary, 0.0)

        elif config.source_class == SourceClass.NOISE:
            signal = self._generate_noise_class(config.noise_class, config.energy_arbitrary, rng)

        elif config.source_class == SourceClass.ACTIVE_PULSE:
            # Active pulser: 5-cycle Hanning-windowed tone burst at 200 kHz
            f_pulse = 200.0e3
            n_cycles = 5
            duration = n_cycles / f_pulse
            mask = t <= duration
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / duration))
            carrier = np.sin(2.0 * np.pi * f_pulse * t)
            signal = np.where(mask, window * carrier * config.energy_arbitrary, 0.0)

        # Scale by orientation effect if crack type has preferred radiation dipole
        if config.crack_type == CrackType.PARALLEL_TO_BUSBAR:
            # Dipole radiation oriented perpendicular to busbars
            theta = np.radians(config.orientation_deg)
            signal *= (0.7 + 0.3 * np.abs(np.cos(theta)))

        # Normalize or clamp
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * config.energy_arbitrary

        return signal

    def _generate_noise_class(self, noise_class: NoiseClass, energy: float, rng: np.random.Generator) -> np.ndarray:
        t = self.time
        n = self.n_samples

        if noise_class == NoiseClass.RAIN_HAIL:
            # Heavy broadband impact with rapid decay and low-frequency ring
            tau_impact = rng.uniform(15.0e-6, 40.0e-6)
            f_ring = rng.uniform(40.0e3, 90.0e3)
            env = np.exp(-t / tau_impact)
            noise = rng.normal(0, 0.4, n)
            return (env * (np.sin(2.0 * np.pi * f_ring * t) + noise)) * energy

        elif noise_class == NoiseClass.WIND_RATTLE:
            # Low-frequency structural vibration modulation (5 kHz - 25 kHz)
            f_mod = rng.uniform(8.0e3, 22.0e3)
            carrier = np.sin(2.0 * np.pi * f_mod * t + rng.uniform(0, 2*np.pi))
            mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * rng.uniform(500, 2000) * t))
            return carrier * mod * energy * 0.6

        elif noise_class == NoiseClass.THERMAL_TICK:
            # Sharp impulse click from mounting clip thermal expansion
            t_click = rng.uniform(50.0e-6, 200.0e-6)
            t_rel = np.maximum(t - t_click, 0.0)
            tau_click = 3.0e-6
            click = (t_rel / tau_click) * np.exp(-t_rel / tau_click) * np.sin(2.0 * np.pi * 120.0e3 * t_rel)
            return np.where(t >= t_click, click, 0.0) * energy

        elif noise_class == NoiseClass.EMI_BURST:
            # Single-cycle impulsive electrical glitch (high frequency > 350 kHz)
            t_emi = rng.uniform(20.0e-6, 100.0e-6)
            tau_emi = 1.0e-6
            t_rel = np.maximum(t - t_emi, 0.0)
            burst = np.exp(-((t_rel - 2*tau_emi) / tau_emi)**2) * np.sin(2.0 * np.pi * 380.0e3 * t_rel)
            return np.where(t >= t_emi, burst, 0.0) * energy * 1.5

        elif noise_class == NoiseClass.HANDLING_TAP:
            # Broad mechanic tap
            tau_tap = 25.0e-6
            return np.exp(-t / tau_tap) * np.sin(2.0 * np.pi * 60.0e3 * t) * energy

        else: # WHITE_NOISE
            return rng.normal(0, 0.2, n) * energy

