"""Shared fixtures.

* ``mace`` is not needed to run the test-suite: if it is not installed, a stub is registered so that
  ``libraries.mlip`` can be imported, and every test that evaluates the ML-IAP uses the ``emt_backend`` fixture, which
  replaces ``mace_mp`` by ASE's analytical EMT calculator (a real, deterministic interatomic potential for Cu, Al, ...).
* Matplotlib uses the non-interactive Agg backend and every figure is closed after each test.
"""
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mace.calculators  # noqa: F401  (the real package, if installed: libraries.mlip imports it anyway)
except ImportError:
    _mace = types.ModuleType('mace')
    _calculators = types.ModuleType('mace.calculators')

    def _mace_mp_not_installed(*args, **kwargs):
        raise RuntimeError('MACE is not installed: use the emt_backend fixture in tests')

    _calculators.mace_mp = _mace_mp_not_installed
    _mace.calculators = _calculators
    sys.modules['mace'] = _mace
    sys.modules['mace.calculators'] = _calculators


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def emt_backend(monkeypatch):
    """Replace the MACE calculator by ASE's EMT. Returns the list of keyword arguments mace_mp was called with."""
    from ase.calculators.emt import EMT

    import libraries.mlip as mlip

    calls = []

    def fake_mace_mp(**kwargs):
        calls.append(kwargs)
        return EMT()

    monkeypatch.setattr(mlip, 'mace_mp', fake_mace_mp)
    return calls


@pytest.fixture
def broken_backend(monkeypatch):
    """A potential that cannot be loaded (e.g. missing model file)."""
    import libraries.mlip as mlip

    def failing_mace_mp(**kwargs):
        raise RuntimeError('cannot load the model')

    monkeypatch.setattr(mlip, 'mace_mp', failing_mace_mp)
