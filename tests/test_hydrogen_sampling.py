import os
import sys

from pathlib import Path

from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libraries.utilities import generate_inequivalent_hydrogen_sites


def test_generate_inequivalent_hydrogen_sites(tmp_path):
    structure = Structure(
        lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
        species=['Ti', 'O', 'Ti'],
        coords=[[0.0, 0.0, 0.8], [0.5, 0.5, 0.2], [0.5, 0.0, 0.8]],
    )

    surface_dir = tmp_path / 'surface'
    surface_dir.mkdir()
    structure.to(filename=str(surface_dir / 'POSCAR'))
    (surface_dir / 'KPOINTS').write_text('Automatic mesh\n0\nGamma\n 10 10 1\n')

    output_dir = tmp_path / 'H-absorption' / 'test_surface'
    configs = generate_inequivalent_hydrogen_sites(
        str(surface_dir),
        output_dir=str(output_dir),
        adsorption_height=1.5,
        z_tolerance=0.30,
    )

    assert len(configs) >= 2
    assert all(os.path.isdir(path) for path in configs)
    assert all(os.path.exists(os.path.join(path, 'POSCAR')) for path in configs)
    assert all(os.path.exists(os.path.join(path, 'KPOINTS')) for path in configs)
    assert any('conf_00_' in os.path.basename(path) for path in configs)

    first_poscar = os.path.join(configs[0], 'POSCAR')
    with open(first_poscar) as f:
        lines = f.read().splitlines()
    assert 'H' in lines[-1]
