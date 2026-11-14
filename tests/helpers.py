"""Small structure builders shared by the tests."""
import os

import numpy as np

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

SQRT3 = np.sqrt(3.0)


def hex_lattice(a, c):
    return Lattice([[a, 0, 0], [-a / 2, a * SQRT3 / 2, 0], [0, 0, c]])


def fcc111_slab(a=2.7, layers=3, vacuum=12.0, species='Cu', spacing=2.08):
    """ABC-stacked (111)-like slab on a one-atom hexagonal cell; the vacuum is along c."""
    c = layers * spacing + vacuum
    stacking = [(0.0, 0.0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]
    coords = [[*stacking[k % 3], 0.05 + k * spacing / c] for k in range(layers)]
    return Structure(hex_lattice(a, c), [species] * layers, coords)


def square_slab(a=3.0, layers=2, vacuum=12.0, species='Cu', spacing=1.8):
    """Square-lattice slab with one atom per layer (layers alternate between the two hollow positions)."""
    c = layers * spacing + vacuum
    coords = [[0.5 * (k % 2), 0.5 * (k % 2), 0.05 + k * spacing / c] for k in range(layers)]
    return Structure(Lattice.tetragonal(a, c), [species] * layers, coords)


def two_layer_hex_slab(a=3.0, species=('Ti', 'O'), vacuum=15.0, spacing=2.0):
    """Two-layer hexagonal slab: film/substrate stand-in for the lattice-matching tests."""
    c = 2 * spacing + vacuum
    return Structure(hex_lattice(a, c), list(species), [[0, 0, 0.0], [1 / 3, 1 / 3, spacing / c]])


def oblique_slab(a=3.0, b=3.7, gamma=80.0, vacuum=15.0, species='Cu', spacing=2.0, scale=1.0):
    """Two-layer slab on an oblique (low-symmetry) in-plane lattice: no accidental symmetry in the ZSL matches."""
    g = np.radians(gamma)
    c = 2 * spacing + vacuum
    lattice = Lattice([[a * scale, 0, 0], [b * scale * np.cos(g), b * scale * np.sin(g), 0], [0, 0, c]])
    return Structure(lattice, [species, species], [[0, 0, 0], [0.5, 0.5, spacing / c]])


def rotate_about_z(structure, angle_degrees):
    """Same crystal, Cartesian frame rotated about z."""
    t = np.radians(angle_degrees)
    rotation = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return Structure(Lattice(structure.lattice.matrix @ rotation), structure.species, structure.frac_coords)


def write_poscar(structure, folder, name='POSCAR'):
    os.makedirs(folder, exist_ok=True)
    Poscar(structure).write_file(os.path.join(folder, name))
    return os.path.join(folder, name)


def top_height(structure, normal):
    return float((structure.cart_coords @ normal).max())
