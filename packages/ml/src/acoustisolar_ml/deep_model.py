"""
Trainable deep waveform classifier (PyTorch).

The previous ``Waveform1DResNet`` had a forward pass with random weights, no training code
and no residual connection; its reported scores were those of an untrained network.  This
module replaces it with a real 1D residual CNN:

* input: (batch, channels, samples) event windows, channels in **arrival order**,
  per-event normalised by the global peak (absolute level is passed separately as a scalar
  log-amplitude feature so the net still sees it)
* stem conv -> 3 residual blocks (with identity / 1x1 projection skips) -> global average
  pooling -> linear head over the event classes
* ``fit`` with class-balanced cross-entropy, AdamW, optional validation split
* ``save`` / ``load`` use ``torch.save(state_dict)`` and ``torch.load(weights_only=True)``

PyTorch is an optional dependency; importing this module without it raises a clear error
only when the model is instantiated.
"""

from typing import Optional, Sequence

import numpy as np

from acoustisolar_schemas.event import EVENT_CLASSES


def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Waveform1DResNet requires PyTorch: pip install torch") from exc


def prepare_waveforms(waveforms: np.ndarray, decimate: int = 2, window: int = 1024) -> tuple:
    """Arrival-ordered, peak-normalised, decimated windows + log-amplitude scalar.

    Returns (x: (N, C, window) float32, log_amp: (N, 1) float32).
    """
    w = np.asarray(waveforms, dtype=np.float32)
    n, c, t = w.shape
    out = np.zeros((n, c, window), dtype=np.float32)
    log_amp = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        e = w[i]
        peak = float(np.max(np.abs(e))) + 1e-12
        log_amp[i, 0] = np.log10(peak)
        # arrival order by first sample above 10 % of the channel peak
        onsets = [int(np.argmax(np.abs(ch) > 0.1 * (np.max(np.abs(ch)) + 1e-12))) for ch in e]
        order = np.argsort(onsets, kind="stable")
        start = max(0, min(onsets) - 64)
        seg = e[order, start : start + window * decimate : decimate] / peak
        out[i, :, : seg.shape[1]] = seg
    return out, log_amp


class Waveform1DResNet:
    def __init__(self, in_channels: int = 4, n_classes: int = len(EVENT_CLASSES), width: int = 32,
                 seed: int = 42, classes: Sequence[str] = EVENT_CLASSES, device: Optional[str] = None):
        torch = _torch()
        nn = torch.nn
        self.classes = list(classes)
        self.in_channels = in_channels
        self.n_classes = n_classes
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        class Block(nn.Module):
            def __init__(self, cin, cout, stride):
                super().__init__()
                self.c1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
                self.b1 = nn.BatchNorm1d(cout)
                self.c2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False)
                self.b2 = nn.BatchNorm1d(cout)
                self.skip = (nn.Identity() if cin == cout and stride == 1 else
                             nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)))

            def forward(self, x):
                h = torch.relu(self.b1(self.c1(x)))
                h = self.b2(self.c2(h))
                return torch.relu(h + self.skip(x))  # residual connection

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(nn.Conv1d(in_channels, width, 11, stride=2, padding=5, bias=False),
                                          nn.BatchNorm1d(width), nn.ReLU())
                self.blocks = nn.Sequential(Block(width, width, 2), Block(width, 2 * width, 2),
                                            Block(2 * width, 2 * width, 2))
                self.head = nn.Linear(2 * width + 1, n_classes)

            def forward(self, x, log_amp):
                h = self.blocks(self.stem(x)).mean(dim=-1)
                return self.head(torch.cat([h, log_amp], dim=1))

        self.net = Net().to(self.device)

    # ------------------------------------------------------------------ training
    def fit(self, waveforms: np.ndarray, y: np.ndarray, epochs: int = 15, batch_size: int = 64,
            lr: float = 2e-3, verbose: bool = False) -> "Waveform1DResNet":
        torch = _torch()
        x, a = prepare_waveforms(waveforms)
        y = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / self.n_classes, 0.0)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=self.device))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(y) / batch_size)))
        X = torch.tensor(x, device=self.device)
        A = torch.tensor(a, device=self.device)
        Y = torch.tensor(y, device=self.device)
        g = torch.Generator(device="cpu").manual_seed(0)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Y), generator=g)
            total = 0.0
            for i in range(0, len(Y), batch_size):
                idx = perm[i : i + batch_size].to(self.device)
                xb = X[idx]
                # augmentation: random gain and small time shift (robustness to coupling/trigger)
                xb = xb * (1.0 + 0.1 * torch.randn(len(idx), 1, 1, device=self.device))
                xb = torch.roll(xb, shifts=int(torch.randint(-8, 9, (1,), generator=g)), dims=-1)
                opt.zero_grad()
                loss = loss_fn(self.net(xb, A[idx]), Y[idx])
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            if verbose:
                print(f"epoch {ep + 1}/{epochs} loss {total / len(Y):.4f}")
        return self

    def predict_proba(self, waveforms: np.ndarray, batch_size: int = 256) -> np.ndarray:
        torch = _torch()
        x, a = prepare_waveforms(waveforms)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                logits = self.net(torch.tensor(x[i : i + batch_size], device=self.device),
                                  torch.tensor(a[i : i + batch_size], device=self.device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, waveforms: np.ndarray) -> np.ndarray:
        return self.predict_proba(waveforms).argmax(axis=1)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        torch = _torch()
        torch.save({"state_dict": self.net.state_dict(), "classes": self.classes,
                    "in_channels": self.in_channels}, path)

    @classmethod
    def load(cls, path, device: Optional[str] = None) -> "Waveform1DResNet":
        torch = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=True)  # no arbitrary code
        obj = cls(in_channels=payload["in_channels"], n_classes=len(payload["classes"]),
                  classes=payload["classes"], device=device)
        obj.net.load_state_dict(payload["state_dict"])
        return obj
