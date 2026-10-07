import os
import sys

import numpy as np
from pathlib import Path

from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libraries.utilities import (
    _voronoi_site_groups,
    generate_inequivalent_hydrogen_sites,
    read_lattice_vectors,
    read_volume,
)


def test_read_structure_from_poscar_without_vasprun(tmp_path):
    structure = Structure(
        lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
        species=['Ti', 'O', 'Ti'],
        coords=[[0.0, 0.0, 0.8], [0.5, 0.5, 0.2], [0.5, 0.0, 0.8]],
    )

    folder = tmp_path / 'surface_no_vasprun'
    folder.mkdir()
    structure.to(filename=str(folder / 'POSCAR'))

    volume = read_volume(str(folder))
    lattice = read_lattice_vectors(str(folder))

    assert np.isclose(volume, structure.volume)
    np.testing.assert_allclose(lattice, structure.lattice.matrix)


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


def test_voronoi_site_groups_handles_unequal_neighbor_counts():
    structure = Structure(
        lattice=[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 12.0]],
        species=['Ti'] * 9,
        coords=[
            [0.15, 0.15, 0.8],
            [0.45, 0.15, 0.8],
            [0.75, 0.15, 0.8],
            [0.15, 0.45, 0.8],
            [0.45, 0.45, 0.8],
            [0.75, 0.45, 0.8],
            [0.15, 0.75, 0.8],
            [0.45, 0.75, 0.8],
            [0.75, 0.75, 0.8],
        ],
    )

    groups = _voronoi_site_groups(list(structure))

    assert sum(len(group) for group in groups) == len(structure)
    assert all(len(group) > 0 for group in groups)


def test_generate_inequivalent_hydrogen_sites_uses_surface_normal(tmp_path):
    lattice = [[4.0, 0.0, 1.0], [0.0, 4.0, 1.0], [2.0, 2.0, 4.0]]
    structure = Structure(
        lattice=lattice,
        species=['Ti'] * 3,
        coords=[
            [0.1, 0.1, 0.8],
            [0.5, 0.2, 0.8],
            [0.2, 0.7, 0.8],
        ],
    )

    surface_dir = tmp_path / 'tilted_surface'
    surface_dir.mkdir()
    structure.to(filename=str(surface_dir / 'POSCAR'))
    (surface_dir / 'KPOINTS').write_text('Automatic mesh\n0\nGamma\n 10 10 1\n')

    output_dir = tmp_path / 'H-absorption' / 'tilted_surface'
    configs = generate_inequivalent_hydrogen_sites(
        str(surface_dir),
        output_dir=str(output_dir),
        adsorption_height=1.5,
        z_tolerance=0.50,
    )

    assert len(configs) >= 1
    poscar = Structure.from_file(configs[0] + '/POSCAR')
    adsorbed_h = poscar.cart_coords[-1]
    max_top_z = max(site.z for site in structure)
    assert abs(adsorbed_h[2] - (max_top_z + 1.5)) > 0.1
