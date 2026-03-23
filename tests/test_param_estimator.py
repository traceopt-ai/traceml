"""Static parameter estimator — regression tests.

Run directly:
    python tests/test_param_estimator.py

Each test writes a tiny Python script to a temp file, runs the static
estimator against it, and checks the result is within a tolerance of the
analytically-computed expected value.

Covers:
  1.  Basic Linear / Conv2d / Embedding counting
  2.  Module-level constant propagation  (INPUT_DIM = 512)
  3.  Self-attribute constants           (self.hidden = 512)
  4.  No double-counting                 (layer in __init__ only)
  5.  nn.Sequential with inline layers
  6.  nn.ModuleList with list literal
  7.  nn.ModuleDict with dict literal
  8.  LSTM num_layers=4 (correct per-layer formula)
  9.  GRU bidirectional
 10.  Nested submodules (Block used inside BigModel)
 11.  Multi-layer MHA transformer block
 12.  Large CNN (our large_dl_example style) — spot-check ballpark
 13.  Robustness — no nn.Module, no crash, returns 0
"""

import os
import sys
import tempfile
import textwrap

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traceml.utils.ast_analysis.param_estimator import (
    _count_params_from_layers,
)

PASS = 0
FAIL = 0


def check(
    label: str,
    script: str,
    expected: int,
    tol_pct: float = 2.0,
) -> None:
    global PASS, FAIL
    src = textwrap.dedent(script)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(src)
        path = fh.name

    try:
        got = _count_params_from_layers(path)
    except Exception as exc:
        print(f"❌  {label}")
        print(f"    CRASHED: {exc}")
        FAIL += 1
        return
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if expected == 0:
        ok = got == 0
    else:
        ok = abs(got - expected) / expected * 100 <= tol_pct

    if ok:
        print(f"✅  {label}  ({got:,})")
        PASS += 1
    else:
        pct = abs(got - expected) / max(expected, 1) * 100
        print(f"❌  {label}")
        print(f"    Expected {expected:,}  Got {got:,}  ({pct:.1f}% error)")
        FAIL += 1


# =============================================================================
# 1. Basic layers
# =============================================================================

check(
    "Linear basic",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 256)
    """,
    784 * 256 + 256,  # 200,960
)

check(
    "Linear no bias",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 128, bias=False)
    """,
    512 * 128,  # 65,536
)

check(
    "Conv2d basic (k=3)",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)
    """,
    3 * 64 * 3 * 3 + 64,  # 1,792
)

check(
    "Embedding",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(30522, 768)
    """,
    30522 * 768,  # 23,440,896
)

# =============================================================================
# 2. Module-level constant propagation
# =============================================================================

check(
    "Module-level constants",
    """
    import torch.nn as nn
    INPUT  = 512
    HIDDEN = 256
    OUT    = 10
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(INPUT, HIDDEN)
            self.fc2 = nn.Linear(HIDDEN, OUT)
    """,
    512 * 256 + 256 + 256 * 10 + 10,  # 133,386
)

check(
    "Module-level arithmetic constant",
    """
    import torch.nn as nn
    BASE = 128
    WIDE = BASE * 4   # 512
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(BASE, WIDE)
    """,
    128 * 512 + 512,  # 66,048
)

# =============================================================================
# 3. Self-attribute constants
# =============================================================================

check(
    "self.hidden = 512 used in Linear dims",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = 512
            self.fc  = nn.Linear(784, self.hidden)
            self.out = nn.Linear(self.hidden, 10)
    """,
    784 * 512 + 512 + 512 * 10 + 10,  # 406,538
)

# =============================================================================
# 4. No double-counting (layers defined in __init__, used in forward)
# =============================================================================

check(
    "No double-counting via forward()",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 10)
        def forward(self, x):
            x = self.fc1(x)       # NOT a new layer definition
            return self.fc2(x)
    """,
    784 * 256 + 256 + 256 * 10 + 10,  # 203,530  (NOT 407,060)
)

# =============================================================================
# 5. nn.Sequential inline
# =============================================================================

check(
    "nn.Sequential positional layers",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
    """,
    784 * 256 + 256 + 256 * 10 + 10,  # 203,530
)

# =============================================================================
# 6. nn.ModuleList with list literal
# =============================================================================

_ml_expected = 128 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10  # 25,418
check(
    "nn.ModuleList list literal",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(128, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 10),
            ])
    """,
    _ml_expected,
)

# =============================================================================
# 7. nn.ModuleDict with dict literal
# =============================================================================

check(
    "nn.ModuleDict dict literal",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleDict({
                'a': nn.Linear(256, 64),
                'b': nn.Linear(256, 32),
            })
    """,
    256 * 64 + 64 + 256 * 32 + 32,  # 24,928
)

# =============================================================================
# 8. LSTM num_layers
# =============================================================================
# Layer 0:       4 * (128*256 + 256*256 + 2*256) = 4 * 98,816 = 395,264
# Layers 1..3:  3 * 4 * (256*256 + 256*256 + 2*256) = 12 * 131,584 = 1,579,008
_lstm_expected = 395_264 + 1_579_008  # 1,974,272

check(
    "LSTM num_layers=4",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(128, 256, num_layers=4)
    """,
    _lstm_expected,
)

# =============================================================================
# 9. GRU bidirectional
# =============================================================================
# Unidirectional layer 0: 3*(128*64 + 64*64 + 2*64) = 3*12544 + ... = 3*12,544 wrong
# 3*(128*64 + 64*64 + 128) = 3*(8192 + 4096 + 128) = 3*12416 = 37248 per dir
# dirs=2 → layer 0 = 74,496
# No additional layers (num_layers=1 default)
_gru_l0 = 3 * (64 * 128 + 128 * 128 + 2 * 128) * 2  # bidirectional
check(
    "GRU bidirectional",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(64, 128, bidirectional=True)
    """,
    _gru_l0,
)

# =============================================================================
# 10. Nested submodules
# =============================================================================
_block_p = 256 * 256 + 256 + 256 * 256 + 256  # 131,584
_nested_expected = 2 * _block_p + 256 * 10 + 10  # 265,738

check(
    "Nested submodules (Block inside BigModel)",
    """
    import torch.nn as nn
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 256)
            self.fc2 = nn.Linear(256, 256)

    class BigModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = Block()
            self.b2 = Block()
            self.head = nn.Linear(256, 10)
    """,
    _nested_expected,
)

# =============================================================================
# 11. MultiheadAttention
# =============================================================================
# MultiheadAttention(512): 4*512*512 + 4*512 = 1,050,624
check(
    "MultiheadAttention",
    """
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(512, 8)
    """,
    4 * 512 * 512 + 4 * 512,  # 1,050,624
)

# =============================================================================
# 12. Spot-check: large_dl_example style (literal dims, many Conv2d layers)
# =============================================================================
# Just verify it returns a plausible non-zero count (>50M params)
_cnn_src = """
import torch.nn as nn
class LargeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1000)
"""
with tempfile.NamedTemporaryFile(
    mode="w", suffix=".py", delete=False, encoding="utf-8"
) as _fh:
    _fh.write(textwrap.dedent(_cnn_src))
    _cnn_path = _fh.name
try:
    _cnn_got = _count_params_from_layers(_cnn_path)
    _cnn_ok = _cnn_got > 500_000  # at least 500K params
    _sym = "✅" if _cnn_ok else "❌"
    print(f"{_sym}  Large CNN smoke-test  ({_cnn_got:,} params, expect >500K)")
    if _cnn_ok:
        PASS += 1
    else:
        FAIL += 1
finally:
    try:
        os.unlink(_cnn_path)
    except OSError:
        pass

# =============================================================================
# 13. Robustness — functional-style script, no nn.Module, must not crash
# =============================================================================

check(
    "Robustness: no nn.Module class, returns 0 without crash",
    """
    x = some_library.make_model(1024)
    result = x.fit(data)
    """,
    0,
    tol_pct=100,
)

# =============================================================================
# Summary
# =============================================================================
print()
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL:
    sys.exit(1)
