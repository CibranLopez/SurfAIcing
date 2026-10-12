"""Hydrogen adsorption sites (periodic Voronoi decomposition of the top layer) and configurations."""
import json
import os
import shutil

import numpy as np

import libraries.vasp_files as vasp_files

from scipy.spatial import Voronoi

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar


def get_hydrogen_pair_distance(
        folder
):
    """Return the H-H distance in a relaxed H2 adsorption configuration.

    Used to check whether an 'H2' configuration is still molecular after relaxation (about 0.75 A)
    or has dissociated into two chemisorbed H atoms (well above 1 A). Returns None if the folder
    does not contain exactly two H atoms (e.g. the slab itself contains hydrogen).

    Args:
        folder (str): Path to the folder containing the relaxed configuration.

    Returns:
        distance (float): H-H distance in Angstrom, taking periodic images into account.
    """
    structure = vasp_files.read_structure(folder)
    if structure is None:
        return None
    h_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'H']
    if len(h_indices) != 2:
        return None
    return structure.get_distance(h_indices[0], h_indices[1])


def _surface_frame(
        structure,
        surface='top'
):
    """Return the slab with its atoms unwrapped along c, the surface normal and an in-plane basis (u, v).

    The normal is the cross product of the first two lattice vectors, oriented along c, so it is exact
    for any Miller index (a PCA of the atomic positions is NOT: for a thick slab with a small in-plane
    cell the direction of smallest variance lies in the surface plane). The slab is unwrapped
    across the periodic boundary by cutting the cell in the middle of the largest gap along c, i.e. the
    vacuum. ``surface='bottom'`` flips the normal to sample the opposite termination.
    """
    if surface not in ('top', 'bottom'):
        raise ValueError(f"surface must be 'top' or 'bottom', got {surface!r}")

    matrix = structure.lattice.matrix
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, matrix[2]) < 0:
        normal = -normal

    frac = structure.frac_coords.copy()
    sorted_c = np.sort(frac[:, 2] % 1.0)
    gaps = np.diff(np.r_[sorted_c, sorted_c[0] + 1.0])
    slab_start = sorted_c[(int(np.argmax(gaps)) + 1) % len(sorted_c)]
    frac[:, 2] = (frac[:, 2] - slab_start) % 1.0
    unwrapped = Structure(structure.lattice, structure.species, frac)

    if surface == 'bottom':
        normal = -normal
    u = matrix[0] / np.linalg.norm(matrix[0])
    v = np.cross(normal, u)
    return unwrapped, normal, u, v


def _environment_signature(structure, position, cutoff):
    """Sorted (atomic number, distance) list of the slab atoms around a point, used as a fingerprint."""
    neighbors = structure.get_sites_in_sphere(position, cutoff)
    return sorted((int(n.specie.Z), float(n.nn_distance)) for n in neighbors)


def _same_environment(signature_a, signature_b, tolerance):
    """Two fingerprints match if they list the same species and distances within a tolerance."""
    if len(signature_a) != len(signature_b):
        return False
    for (z_a, d_a), (z_b, d_b) in zip(signature_a, signature_b):
        if z_a != z_b or abs(d_a - d_b) > tolerance:
            return False
    return True


def voronoi_adsorption_sites(
        structure,
        normal,
        u,
        v,
        z_tolerance=0.25,
        adsorption_height=1.5,
        env_cutoff=4.5,
        env_tolerance=0.05
):
    """Inequivalent adsorption sites of the top layer from a periodic 2D Voronoi decomposition.

    The top-layer atoms (within ``z_tolerance`` of the highest one along the normal) are projected on
    the surface plane and tiled periodically. Their Voronoi diagram defines three kinds of sites:

    * atop   : on each top-layer atom;
    * bridge : midpoint of each pair of atoms that share a Voronoi ridge (nearest neighbours);
    * hollow : each Voronoi vertex (equidistant from three or more atoms).

    Tiling the layer makes the diagram periodic, so atoms at the cell edge have the same neighbours as
    equivalent atoms in the middle of the cell. Sites are then reduced to the symmetry-inequivalent ones
    by comparing the species and distances of the slab atoms around the actual adsorption position
    (within ``env_cutoff``, which must reach the second layer to tell apart e.g. fcc and hcp hollows). Species are included, so Bi and S atoms of the same layer are never merged.

    Returns:
        list[dict]: one entry per inequivalent site with its 'kind' and Cartesian 'position'.
    """
    heights = structure.cart_coords @ normal
    top_idx = np.where(heights >= heights.max() - z_tolerance)[0]

    a, b = structure.lattice.matrix[0], structure.lattice.matrix[1]

    # In-plane (2D) cell. Everything is done in the orthonormal basis (u, v, normal), so the in-plane
    # component of c (c is not perpendicular to the surface for general Miller indices) never matters.
    cell_2d     = np.array([[a @ u, a @ v], [b @ u, b @ v]])
    cell_2d_inv = np.linalg.inv(cell_2d)

    # Wrap the top-layer atoms into the central 2D cell, then tile that cell periodically
    top_2d = np.c_[structure.cart_coords[top_idx] @ u, structure.cart_coords[top_idx] @ v]
    top_fractional = top_2d @ cell_2d_inv
    top_2d = (top_fractional - np.floor(top_fractional)) @ cell_2d

    shifts = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]
    plane, point_heights, is_central = [], [], []
    for (i, j) in shifts:
        for n, k in enumerate(top_idx):
            plane.append(top_2d[n] + i * cell_2d[0] + j * cell_2d[1])
            point_heights.append(heights[k])
            is_central.append((i, j) == (0, 0))
    plane         = np.array(plane)
    point_heights = np.array(point_heights)
    is_central    = np.array(is_central)

    def inside_cell(point_2d):
        fractional = point_2d @ cell_2d_inv
        return np.all(fractional >= -1e-8) and np.all(fractional < 1 - 1e-8)

    candidates = []  # (kind, 2D position, indices of the atoms that define the site)
    for n in np.where(is_central)[0]:
        candidates.append(('atop', plane[n], [n]))

    voronoi = Voronoi(plane)
    for i, j in voronoi.ridge_points:
        midpoint = 0.5 * (plane[i] + plane[j])
        if inside_cell(midpoint):
            candidates.append(('bridge', midpoint, [i, j]))

    for vertex in voronoi.vertices:
        if inside_cell(vertex):
            distances = np.linalg.norm(plane - vertex, axis=1)
            defining = np.where(distances < distances.min() + 1e-3)[0]
            candidates.append(('hollow', vertex, list(defining)))

    kind_order = {'atop': 0, 'bridge': 1, 'hollow': 2}
    candidates.sort(key=lambda candidate: kind_order[candidate[0]])

    sites, signatures = [], []
    for kind, position_2d, defining in candidates:
        height   = point_heights[defining].max() + adsorption_height
        position = position_2d[0] * u + position_2d[1] * v + height * normal
        signature = _environment_signature(structure, position, env_cutoff)
        if any(_same_environment(signature, other, env_tolerance) for other in signatures):
            continue
        signatures.append(signature)
        sites.append({'kind': kind, 'position': position})

    return sites


def generate_inequivalent_hydrogen_sites(
        surface_dir,
        output_dir=None,
        adsorption_height=1.5,
        z_tolerance=0.25,
        sample_label='conf',
        adsorbate='H',
        h2_bond_length=0.74,
        surface='top',
        env_cutoff=4.5,
        min_distance=1.0,
        overwrite=False
):
    """Generate hydrogen adsorption configurations on the inequivalent sites of a surface.

    The slab is read from CONTCAR (relaxed) if it exists, otherwise from POSCAR. The surface normal is
    computed from the lattice, the top layer is decomposed with a periodic Voronoi tessellation into
    atop, bridge and hollow sites, and symmetry-equivalent sites are removed (see
    ``voronoi_adsorption_sites``). One folder is written per site (and per orientation for H2):
    ``<sample_label>_<index>_<kind>[_<orientation>]``, with a POSCAR and a copy of KPOINTS, POTCAR,
    INCAR and run.sh if they exist in the surface folder. A ``sites.json`` summary is also written.

    Args:
        surface_dir (str): Directory containing the slab CONTCAR/POSCAR (and optional KPOINTS/POTCAR).
        output_dir (str): Directory where the generated adsorption calculations will be written.
            If not provided, a sibling directory called H-absorption is created.
        adsorption_height (float): Height of the (lowest) H atom above the top atoms defining the site, in Angstrom.
        z_tolerance (float): Atoms within this distance of the highest atom belong to the top layer.
        sample_label (str): Prefix used for the generated folders.
        adsorbate (str): 'H' for a hydrogen atom or 'H2' for a hydrogen molecule. For 'H2', each site is
            generated with two orientations (perpendicular and parallel to the surface).
        h2_bond_length (float): H-H bond length in Angstrom, used when adsorbate is 'H2'.
        surface (str): 'top' or 'bottom' termination of the slab (they differ for asymmetric slabs).
        env_cutoff (float): Radius in Angstrom used to compare local environments when merging equivalent sites.
        min_distance (float): Configurations with an H atom closer than this to a slab atom are skipped.
        overwrite (bool): If configurations already exist in output_dir they are reused (their paths are
            returned) unless overwrite=True, which deletes them and generates them again.

    Returns:
        list[str]: Paths to the generated adsorption directories.
    """
    if adsorbate not in ('H', 'H2'):
        raise ValueError(f"adsorbate must be 'H' or 'H2', got {adsorbate!r}")

    surface_dir = os.path.abspath(surface_dir)
    structure_path = vasp_files.get_structure_file(surface_dir)
    if structure_path is None:
        raise FileNotFoundError(f'Neither CONTCAR nor POSCAR found in surface directory: {surface_dir}')

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(surface_dir), 'H-absorption')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    existing = sorted(os.path.join(output_dir, name) for name in os.listdir(output_dir)
                      if name.startswith(sample_label) and os.path.isdir(os.path.join(output_dir, name)))
    if existing and not overwrite:
        print(f'{len(existing)} configurations already exist in {output_dir}; reusing them (overwrite=True regenerates them).')
        return existing
    for path in existing:
        shutil.rmtree(path)

    structure, normal, u, v = _surface_frame(Structure.from_file(structure_path), surface=surface)
    sites = voronoi_adsorption_sites(structure, normal, u, v,
                                     z_tolerance=z_tolerance,
                                     adsorption_height=adsorption_height,
                                     env_cutoff=env_cutoff)
    if not sites:
        raise ValueError(f'Could not identify any adsorption site in {surface_dir}.')

    if adsorbate == 'H':
        orientations = [('', [np.zeros(3)])]
    else:
        orientations = [
            ('perp', [np.zeros(3), h2_bond_length * normal]),
            ('para', [-0.5 * h2_bond_length * u, 0.5 * h2_bond_length * u]),
        ]

    created_dirs, summary = [], {}
    for site_index, site in enumerate(sites):
        for orientation_name, displacements in orientations:
            positions = [site['position'] + d for d in displacements]

            # Skip configurations in which an H atom overlaps with the slab
            if any(len(structure.get_sites_in_sphere(p, min_distance)) > 0 for p in positions):
                continue

            config_name = f'{sample_label}_{site_index:02d}_{site["kind"]}'
            if orientation_name:
                config_name += f'_{orientation_name}'
            config_dir = os.path.join(output_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)
            created_dirs.append(config_dir)

            new_structure = structure.copy()
            for position in positions:
                new_structure.append('H', position, coords_are_cartesian=True)
            Poscar(new_structure).write_file(os.path.join(config_dir, 'POSCAR'))

            for filename in ['KPOINTS', 'POTCAR', 'INCAR', 'run.sh']:
                source = os.path.join(surface_dir, filename)
                if os.path.exists(source):
                    shutil.copy(source, os.path.join(config_dir, filename))

            summary[config_name] = {'kind': site['kind'], 'orientation': orientation_name or None,
                                    'cartesian': [float(x) for x in site['position']]}

    with open(os.path.join(output_dir, 'sites.json'), 'w') as json_file:
        json.dump({'surface': surface, 'adsorbate': adsorbate, 'adsorption_height': adsorption_height,
                   'configurations': summary}, json_file, indent=2)

    return created_dirs
