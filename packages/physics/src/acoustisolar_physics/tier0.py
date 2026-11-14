import numpy as np
from typing import List, Tuple, Optional
from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from acoustisolar_schemas.source import SourceEventConfig
from .materials import GLASS_MATERIAL
from .lamb import LambDispersionSolver
from .viscoelastic_eva import ViscoelasticEVA
from .source_library import SourceLibrary

class Tier0Simulator:
    """
    Tier-0 Analytical Simulator for guided Lamb wave propagation in solar modules.
    
    Features:
    - Frequency-domain Green's function propagation
    - Multimodal dispersion (S0 and A0 mode phase delays)
    - Geometric spreading (1 / sqrt(r))
    - Temperature-dependent viscoelastic EVA attenuation alpha(f, T)
    - Multichannel acoustic emission waveform generation
    """

    def __init__(
        self,
        module_config: Optional[ModuleTwinConfig] = None,
        sensor_config: Optional[SensorArrayConfig] = None,
        sample_rate_hz: float = 1_000_000.0,
        n_samples: int = 2560,
    ):
        self.module = module_config or ModuleTwinConfig()
        self.sensor_config = sensor_config or SensorArrayConfig()
        self.sample_rate = sample_rate_hz
        self.n_samples = n_samples
        self.dt = 1.0 / sample_rate_hz
        
        glass_thick_m = self.module.stack[0].thickness_mm * 1.0e-3
        self.lamb_solver = LambDispersionSolver(GLASS_MATERIAL, thickness_m=glass_thick_m)
        self.eva_model = ViscoelasticEVA()
        self.source_lib = SourceLibrary(sample_rate_hz=sample_rate_hz, n_samples=n_samples)
        
        # Precompute dispersion grid for instantaneous interpolation across simulation runs
        f_max = sample_rate_hz / 2.0
        self.f_grid = np.linspace(0.0, f_max, 64)
        self.vs0_grid = np.array([self.lamb_solver.find_phase_velocity_s0(f) for f in self.f_grid])
        self.va0_grid = np.array([self.lamb_solver.find_phase_velocity_a0(f) for f in self.f_grid])

    def simulate(
        self,
        source: SourceEventConfig,
        temperature_c: float = 25.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate acoustic emission propagation to all sensors.
        Returns:
            waveforms: np.ndarray of shape (n_channels, n_samples)
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # 1. Generate excitation source signal s(t)
        s_t = self.source_lib.generate_source_signal(source, rng=rng)
        
        # 2. Compute frequency-domain spectrum S(f)
        n = self.n_samples
        freqs = np.fft.rfftfreq(n, d=self.dt)
        S_f = np.fft.rfft(s_t)
        
        # 3. Fast interpolated dispersion velocities and attenuation across frequencies
        v_s0 = np.interp(freqs, self.f_grid, self.vs0_grid)
        v_a0 = np.interp(freqs, self.f_grid, self.va0_grid)
        
        # Safe phase velocities
        v_s0 = np.maximum(v_s0, 1000.0)
        v_a0 = np.maximum(v_a0, 200.0)
        
        omega = 2.0 * np.pi * freqs
        k_s0 = omega / v_s0
        k_a0 = omega / v_a0
        
        alpha_eva = self.eva_model.attenuation_factor(freqs, temp_c=temperature_c)
        
        # Sensor positions
        sensors = self.sensor_config.positions_mm
        n_sensors = len(sensors)
        output_waveforms = np.zeros((n_sensors, n), dtype=np.float64)
        
        # Distance from source to each sensor (in meters)
        xs_m = source.x_mm * 1.0e-3
        ys_m = source.y_mm * 1.0e-3
        
        r0 = 0.02 # 2 cm near-field regularization
        
        # Modal excitation weights (crack release in silicon couples predominantly to S0 in front glass)
        w_s0 = 0.75
        w_a0 = 0.25
        
        for ch in range(n_sensors):
            xr_m = sensors[ch][0] * 1.0e-3
            yr_m = sensors[ch][1] * 1.0e-3
            
            dist_m = np.sqrt((xs_m - xr_m)**2 + (ys_m - yr_m)**2)
            dist_eff = max(dist_m, r0)
            
            # Geometric cylindrical spreading in plate: 1 / sqrt(r)
            geom_spreading = np.sqrt(r0 / dist_eff)
            
            # Frequency-dependent viscoelastic damping
            damping = np.exp(-alpha_eva * dist_m)
            
            # Phase shifts for S0 and A0 modes
            transfer_s0 = np.exp(-1j * k_s0 * dist_m)
            transfer_a0 = np.exp(-1j * k_a0 * dist_m)
            
            # Total acoustic transfer function
            H_f = geom_spreading * damping * (w_s0 * transfer_s0 + w_a0 * transfer_a0)
            
            # Modulate sensor sensitivity variation (sensor calibration spread)
            sens_factor = 1.0 + rng.normal(0, 0.03)
            Y_f = S_f * H_f * sens_factor
            
            # Reconstruct time domain waveform
            ch_wave = np.fft.irfft(Y_f, n=n)
            output_waveforms[ch, :] = ch_wave

        return output_waveforms
