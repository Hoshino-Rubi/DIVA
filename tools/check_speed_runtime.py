"""Production-runtime CPU fixtures only; no real-data neural requests."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
from pathlib import Path
from speed_invariance_common import write_json_new
from any2xsampa.training.speed_invariance import make_head, adapter_state, load_adapter, speed_view, objective


def check(destination: Path) -> None:
    import torch
    import numpy as np
    torch.set_num_threads(4)
    versions = {name: metadata.version(name) for name in ('torch', 'torchaudio', 'scipy', 'numpy', 'safetensors')}
    assert versions['torch'] == versions['torchaudio'] == '2.8.0+cu126'
    assert versions['scipy'] == '1.15.3' and versions['numpy'] == '2.2.6'
    torch.manual_seed(17)
    head = {'weight': torch.randn(428, 1024) / 100, 'bias': torch.zeros(428)}
    a = make_head(torch, head, device='cpu'); b = make_head(torch, head, device='cpu')
    load_adapter(torch, b, adapter_state(a))
    x = torch.randn(1, 8, 1024)
    teacher = torch.nn.functional.linear(x, head['weight'], head['bias'])
    assert torch.equal(a(x), teacher) and torch.equal(b(x), teacher)
    loss, diagnostics = objective(torch, a(x), teacher, [1, 1, 427], 'original')
    loss.backward(); assert torch.isfinite(a.residual[2].weight.grad).all()
    wave = np.sin(np.arange(160000, dtype=np.float32) / 80)
    assert [len(speed_view(wave, v)[0]) for v in ('raw', 'slow_0p9', 'fast_1p1')] == [160000, 177778, 145455]
    assert not torch.cuda.is_initialized()
    write_json_new(destination, {'passed': True, 'versions': versions, 'fixture_device': 'cpu',
        'cuda_initialized': False, 'actual_data_neural_requests': 0, 'optimizer_updates': 0, 'loss_fixture': diagnostics})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); check(p.parse_args().output)
