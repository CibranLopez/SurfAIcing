import json
import os

import numpy as np

from ase.io.vasp import write_vasp

from pymatgen.core.interface import Interface
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import get_2d_transform, from_2d_to_3d


# ---------------------------------------------------------------------------------------------
# Reading the slabs
# ---------------------------------------------------------------------------------------------
def _largest_gap(structure, axis):
    """Largest gap (as a fraction of the cell, taking periodicity into account) between atoms along a lattice axis."""
    frac = np.sort(structure.frac_coords[:, axis] % 1.0)
    return float(np.max(np.diff(np.r_[frac, frac[0] + 1.0])))


def check_slab_file(
        structure,
        label,
        poscar_path,
        expected_miller=None
):
    """Check that a file really is a slab cut so that the FIRST TWO lattice vectors span the surface.

    The matching uses lattice.matrix[:2], so it silently gives wrong (or no) matches if the vacuum is not along c
    or if the file is not the slab it is supposed to be. Checks: (1) the vacuum along the surface normal (a x b),
    (2) no vacuum along a or b, (3) the in-plane area against the area stored by generate-slabs.ipynb
    (slab_data.json next to the POSCAR), (4) the Miller index against the stored one.

    Args:
        structure (Structure): Slab read from the file.
        label (str): Name used in the messages ('substrate' or 'film').
        poscar_path (str): Path of the file the structure was read from.
        expected_miller (tuple): Miller index the user declared for this slab.

    Returns:
        None. Problems are reported as printed warnings.
    """
    matrix = structure.lattice.matrix
    area   = np.linalg.norm(np.cross(matrix[0], matrix[1]))
    height = structure.volume / area                      # cell height along the surface normal

    vacuum_normal = _largest_gap(structure, 2) * height
    print(f"{label}: in-plane area {area:.1f} Å², vacuum along the normal {vacuum_normal:.1f} Å")
    if vacuum_normal < 5.0:
        print(f"[warning] '{label}' has only {vacuum_normal:.1f} Å of vacuum along c: it does NOT look like a slab whose first two lattice "
              f"vectors span the surface (bulk cell, or vacuum along another axis?).")
    for axis, name in [(0, 'a'), (1, 'b')]:
        vacuum = _largest_gap(structure, axis) * (structure.volume / np.linalg.norm(np.cross(matrix[(axis + 1) % 3], matrix[(axis + 2) % 3])))
        if vacuum > 5.0:
            print(f"[warning] '{label}' has {vacuum:.1f} Å of vacuum along {name}: the in-plane lattice used for the matching "
                  f"is probably not the surface plane.")

    json_path = os.path.join(os.path.dirname(os.path.abspath(poscar_path)), 'slab_data.json')
    if os.path.exists(json_path):
        with open(json_path) as json_file:
            slab_data = json.load(json_file)
        if abs(slab_data['surface_area'] - area) > 0.01 * area:
            print(f"[warning] '{label}': in-plane area of the file ({area:.1f} Å²) differs from the area stored in slab_data.json "
                  f"({slab_data['surface_area']:.1f} Å²): the POSCAR is not the slab that was generated.")
        if expected_miller is not None and tuple(slab_data['miller_index']) != tuple(expected_miller):
            print(f"[warning] '{label}': slab_data.json says Miller index {tuple(slab_data['miller_index'])}, but the declared "
                  f"Miller index is {tuple(expected_miller)}.")


def wrap_as_slab(
        structure,
        miller_index
):
    """Wrap a plain Structure (as read back from a POSCAR) into a Slab object.

    `oriented_unit_cell` and `shift` are not recoverable from a bare slab file, so they are just placeholders: only
    `.lattice`/`.species`/`.frac_coords` and `.make_supercell()`/`.get_orthogonal_c_slab()` are used afterwards, never
    the bulk-provenance metadata. `reorient_lattice=False` so pymatgen does not silently re-orient the lattice.
    """
    return Slab(structure.lattice, structure.species, structure.frac_coords,
                miller_index=miller_index, oriented_unit_cell=structure,
                shift=0.0, scale_factor=np.eye(3), reorient_lattice=False,
                coords_are_cartesian=False)


def describe_lattice(
        slab
):
    """Lengths, angle and area of the in-plane lattice of a slab, as a string."""
    a, b = slab.lattice.matrix[:2]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    angle = np.degrees(np.arccos(np.dot(a, b) / (la * lb)))
    return f'|a| = {la:.2f} Å, |b| = {lb:.2f} Å, angle = {angle:.1f}°, area = {np.linalg.norm(np.cross(a, b)):.1f} Å²'


# ---------------------------------------------------------------------------------------------
# Lattice matching
# ---------------------------------------------------------------------------------------------
def make_zsl_generator(
        max_area,
        max_area_ratio_tol,
        max_length_tol,
        max_angle_tol
):
    """Zur-McGill (ZSL) coincidence-lattice generator that searches supercells of either lattice (bidirectional)."""
    return ZSLGenerator(
        max_area_ratio_tol=max_area_ratio_tol,
        max_area=max_area,
        max_length_tol=max_length_tol,
        max_angle_tol=max_angle_tol,
        bidirectional=True,
    )


def match_strain(
        match
):
    """Von Mises strain of a ZSL match: lower means a less-distorted, more physically reasonable coincidence lattice."""
    return Deformation(match.match_transformation).green_lagrange_strain.von_mises_strain


def find_matches(
        film_slab,
        substrate_slab,
        generator,
        n_keep=None
):
    """All ZSL matches between the in-plane lattices of the two slabs, sorted by ascending strain and capped to `n_keep`.

    Args:
        film_slab (Slab): Film slab.
        substrate_slab (Slab): Substrate slab.
        generator (ZSLGenerator): Generator from make_zsl_generator.
        n_keep (int): Number of matches to keep (all if None).

    Returns:
        list: ZSL matches.
    """
    matches = list(generator(film_slab.lattice.matrix[:2], substrate_slab.lattice.matrix[:2], lowest=False))
    matches.sort(key=match_strain)
    return matches if n_keep is None else matches[:n_keep]


def explain_no_match(
        film_slab,
        substrate_slab,
        max_area,
        max_area_ratio_tol
):
    """Report (without generating anything) how far a pair is from having a match.

    Prints the smallest length/angle tolerance, at the configured and at twice the configured interface area, for
    which ZSL finds a coincidence. The settings used for the generation are NOT changed.
    """
    print('No match with the configured settings. How far is this pair from matching?')
    for area in (max_area, 2 * max_area):
        for tol in (0.04, 0.05, 0.06, 0.08, 0.10, 0.15):
            generator = make_zsl_generator(area, max_area_ratio_tol, tol, tol)
            matches = find_matches(film_slab, substrate_slab, generator)
            if matches:
                print(f'    max_area={area:g} Å², tolerance={tol:.0%}: {len(matches)} match(es); lowest strain {match_strain(matches[0]):.3f}, '
                      f'supercell area {matches[0].match_area:.0f} Å²')
                break
        else:
            print(f'    max_area={area:g} Å²: no match up to a 15% tolerance')


def _in_plane_frame(
        slab
):
    """Orthonormal in-plane basis (u, v) of a slab: u along the first lattice vector."""
    matrix = slab.lattice.matrix
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    u = matrix[0] / np.linalg.norm(matrix[0])
    v = np.cross(normal, u)
    return np.array([u, v]).T


def effective_twist(
        match,
        film_slab,
        substrate_slab
):
    """Relative in-plane orientation (degrees) of the film and substrate crystals implied by a ZSL match.

    The match maps the film supercell vectors onto the substrate supercell vectors. Written in the crystal frame of
    each slab (first lattice vector along x), that map is a rotation plus a strain; the angle of its rotation part is
    the angle between the crystallographic axes of film and substrate, i.e. the twist. It does not depend on how the
    slabs happen to be oriented in Cartesian space. NaN is returned when the match reverses the in-plane orientation
    (the map contains a reflection), where a twist angle is not defined.
    """
    film_vectors      = np.array(match.film_sl_vectors) @ _in_plane_frame(film_slab)
    substrate_vectors = np.array(match.substrate_sl_vectors) @ _in_plane_frame(substrate_slab)
    transform = np.linalg.solve(film_vectors, substrate_vectors)
    if np.linalg.det(transform) < 0:
        return float('nan')
    u, _, vt = np.linalg.svd(transform)
    rotation = u @ vt
    x_axis   = np.array([1.0, 0.0]) @ rotation
    return float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))


# ---------------------------------------------------------------------------------------------
# Building the interfaces
# ---------------------------------------------------------------------------------------------
def build_supercell_slabs(
        film_slab,
        substrate_slab,
        match
):
    """Build the film/substrate supercells implied by a ZSL match.

    Same recipe CoherentInterfaceBuilder.get_interfaces uses internally, applied directly to our own pre-cut slabs
    instead of slabs it would otherwise cut itself from a bulk cell.
    """
    film_transform = np.round(from_2d_to_3d(
        get_2d_transform(film_slab.lattice.matrix[:2], match.film_sl_vectors)
    )).astype(int)
    film_sl_slab = film_slab.copy()
    film_sl_slab.make_supercell(film_transform)

    sub_transform = np.round(from_2d_to_3d(
        get_2d_transform(substrate_slab.lattice.matrix[:2], match.substrate_sl_vectors)
    )).astype(int)
    sub_sl_slab = substrate_slab.copy()
    sub_sl_slab.make_supercell(sub_transform)

    return film_sl_slab, sub_sl_slab


def subsample_evenly(
        items,
        n_keep
):
    """Return `n_keep` evenly spaced items of a list (the whole list if it is not longer than `n_keep`)."""
    if len(items) <= n_keep:
        return list(items)
    idx = sorted(set(np.linspace(0, len(items) - 1, n_keep).round().astype(int)))
    return [items[i] for i in idx]


def enumerate_registries(
        film_sl_slab,
        sub_sl_slab,
        n_keep,
        reference_gap,
        vacuum_over_film,
        tolerance=0.1
):
    """Symmetry-distinct in-plane (lateral) registries of a film/substrate supercell pair, as fractional shifts of the film.

    The candidate shifts come from the adsorbate-site algorithm of pymatgen (`Interface.get_shifts_based_on_adsorbate_sites`).
    They are reduced modulo the in-plane lattice and shifts closer than `tolerance` (Angstrom, periodic images included)
    are merged, because a shift that differs by a lattice translation is the same registry. The remaining ones are
    evenly subsampled to `n_keep`.

    Args:
        film_sl_slab (Slab): Film supercell.
        sub_sl_slab (Slab): Substrate supercell.
        n_keep (int): Maximum number of registries to return.
        reference_gap (float): Gap of the reference interface used to find the sites, in Angstrom.
        vacuum_over_film (float): Vacuum over the film of the reference interface, in Angstrom.
        tolerance (float): Registries closer than this (Angstrom) are the same registry.

    Returns:
        list[tuple[float, float]]: fractional in-plane shifts of the film, each in [0, 1).
    """
    reference = Interface.from_slabs(substrate_slab=sub_sl_slab.copy(), film_slab=film_sl_slab.copy(),
                                     gap=reference_gap, vacuum_over_film=vacuum_over_film)
    shifts = [tuple(shift) for shift in reference.get_shifts_based_on_adsorbate_sites(tolerance)] or [(0.0, 0.0)]

    a, b = reference.lattice.matrix[:2]
    images = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    kept = []
    for shift in shifts:
        shift = np.round(np.array(shift, dtype=float), 6) % 1.0
        duplicate = any(
            min(np.linalg.norm((shift - other + np.array(image)) @ np.array([a, b])) for image in images) < tolerance
            for other in kept
        )
        if not duplicate:
            kept.append(shift)

    registries = sorted((float(s[0]), float(s[1])) for s in kept)
    return subsample_evenly(registries, n_keep)


def build_interface(
        film_sl_slab,
        sub_sl_slab,
        registry,
        gap,
        vacuum_over_film
):
    """Build a film/substrate interface with the film laterally shifted by `registry`.

    `Interface.from_slabs(in_plane_offset=...)` only stores the offset as an attribute: it never moves the film (checked
    with pymatgen 2026.x: interfaces built with different offsets have identical atomic positions). The shift is therefore
    applied here, explicitly, to the film atoms, in fractional coordinates of the interface cell.

    Args:
        film_sl_slab (Slab): Film supercell.
        sub_sl_slab (Slab): Substrate supercell.
        registry (tuple[float, float]): Fractional in-plane shift of the film along a and b.
        gap (float): Distance between film and substrate, in Angstrom.
        vacuum_over_film (float): Vacuum over the film, in Angstrom.

    Returns:
        Interface: the interface, with the film shifted.
    """
    interface = Interface.from_slabs(substrate_slab=sub_sl_slab.copy(), film_slab=film_sl_slab.copy(),
                                     gap=gap, vacuum_over_film=vacuum_over_film)
    film_indices = np.where(np.array(interface.site_properties['interface_label']) == 'film')[0].tolist()
    interface.translate_sites(film_indices, [registry[0], registry[1], 0.0], frac_coords=True, to_unit_cell=True)
    return interface


def split_interface(
        interface
):
    """Film and substrate atoms of an interface, each as its own Structure **in the interface cell**.

    Because the interface is built on the substrate in-plane lattice, the film in it is strained. Using these parts as
    the isolated-film and isolated-substrate references puts the three energies of the adhesion energy in the same
    strained cell, so the strain energy of the film cancels instead of being counted as adhesion.
    """
    labels = np.array(interface.site_properties['interface_label'])
    parts = {}
    for label in ('film', 'substrate'):
        idx = np.where(labels == label)[0]
        parts[label] = Structure(interface.lattice, [interface[i].specie for i in idx],
                                 interface.frac_coords[idx], coords_are_cartesian=False)
    return parts['film'], parts['substrate']


def to_plain_structure(
        structure
):
    """Interface/Slab objects carry extra site_properties that some ASE/pymatgen version combinations choke on when
    converting to ase.Atoms. Stripping to a plain Structure with just species/coords avoids that."""
    return Structure(structure.lattice, structure.species, structure.frac_coords, coords_are_cartesian=False)


def write_structure(
        structure,
        path
):
    """Write a structure to a VASP file, creating the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_vasp(path, AseAtomsAdaptor.get_atoms(to_plain_structure(structure)), direct=True, sort=True)


def get_adhesion_energy_of_formation(
        interface_energy,
        film_energy,
        substrate_energy,
        interface_area
):
    r"""Adhesion energy of formation of an interface, in J/m^2:

    \begin{equation}
        E_{adhesion} = \frac{E_{interface} - E_{film} - E_{substrate}}{A}
    \end{equation}

    Analogous to get_surface_energy_of_formation, but normalized by a single interface (not the two free surfaces of
    an isolated slab). Energies are total energies in eV and the area is in Angstrom^2 (1 eV/Angstrom^2 = 16.0218 J/m^2).
    """
    return (interface_energy - film_energy - substrate_energy) * 16.0218 / interface_area
