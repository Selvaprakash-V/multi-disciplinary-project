import numpy as np
from typing import Dict, Any, Tuple

class Waveform1DResNet:
    """
    1D-CNN / ResNet architecture for multichannel acoustic waveforms.
    Performs multi-scale 1D temporal convolutions, residual shortcut,
    global average pooling, and binary/multiclass classification.
    """

    def __init__(self, in_channels: int = 4, n_features: int = 16, seed: int = 42):
        self.in_channels = in_channels
        self.n_features = n_features
        rng = np.random.default_rng(seed)
        
        # Conv1: temporal kernel (in_channels, n_features, kernel_size=11)
        k1 = 11
        self.W_conv1 = rng.normal(0, np.sqrt(2.0 / (in_channels * k1)), (n_features, in_channels, k1))
        self.b_conv1 = np.zeros(n_features)
        
        # Conv2: residual block conv (n_features, n_features, kernel_size=7)
        k2 = 7
        self.W_conv2 = rng.normal(0, np.sqrt(2.0 / (n_features * k2)), (n_features, n_features, k2))
        self.b_conv2 = np.zeros(n_features)
        
        # Dense classification head
        self.W_fc = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, 1))
        self.b_fc = np.zeros(1)

    def _conv1d(self, x: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 2) -> np.ndarray:
        """Vectorized 1D convolution over batch and temporal axis."""
        batch_size, in_c, length = x.shape
        out_c, _, k_size = W.shape
        pad = k_size // 2
        
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="constant")
        out_len = (length + 2 * pad - k_size) // stride + 1
        out = np.zeros((batch_size, out_c, out_len), dtype=np.float32)
        
        for i in range(out_len):
            t_start = i * stride
            window = x_pad[:, :, t_start : t_start + k_size] # (B, in_c, k_size)
            # Sum over in_channels and kernel window
            val = np.tensordot(window, W, axes=([1, 2], [1, 2])) # (B, out_c)
            out[:, :, i] = val + b
            
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass on input batch x of shape (batch_size, in_channels, n_samples).
        Returns predicted probabilities P(is_crack).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
            
        # Layer 1: Conv1D + ReLU + Strided Downsampling
        h1 = self._conv1d(x, self.W_conv1, self.b_conv1, stride=4)
        h1 = np.maximum(h1, 0.0) # ReLU
        
        # Layer 2: Residual Block (Conv1D + ReLU + Skip)
        h2 = self._conv1d(h1, self.W_conv2, self.b_conv2, stride=2)
        h2 = np.maximum(h2, 0.0)
        
        # Global Average Pooling over temporal dimension
        pooled = np.mean(h2, axis=-1) # (B, n_features)
        
        # Linear Head + Sigmoid
        logits = pooled @ self.W_fc + self.b_fc
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return probs.squeeze(-1)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.forward(x)
        return (probs >= threshold).astype(int)

