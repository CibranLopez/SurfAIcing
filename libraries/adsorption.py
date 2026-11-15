"""Hydrogen adsorption sites (periodic Voronoi decomposition of the top layer), configurations and adsorption free energies.

The adsorption quantity is a free (Gibbs) energy difference,

    dG_ads = G_conf - ( n G_surf + n_H/2 G_H2 )            per H atom ('H') or per H2 molecule ('H2')

evaluated at the level of the free energy that the caller chooses: the static 0 K level (G = E0, ``energy_lookup.read_energy``,
the first part of the H-absorption notebook) or finite temperature, G(T) = U0 + F_vib from the NPT runs of the surface and of
every configuration (``adsorption_free_energy``). G is a sum of terms collected in a dictionary, so a further term is one more
entry. The isolated H2 reference is the free energy of its molecular vibration only (``h2_gas_free_energy``); its translation and
rotation, and the pressure of the gas, are not included.
"""
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


def _warn_if_settings_differ(
        output_dir,
        adsorbate,
        surface,
        adsorption_height
):
    """Warn when reusable configurations were generated with other settings than the ones requested now."""
    path = os.path.join(output_dir, 'sites.json')
    if not os.path.exists(path):
        return
    with open(path) as json_file:
        previous = json.load(json_file)
    requested = {'adsorbate': adsorbate, 'surface': surface, 'adsorption_height': adsorption_height}
    for key, value in requested.items():
        if key in previous and previous[key] != value:
            print(f'[warning] the configurations in {output_dir} were generated with {key}={previous[key]!r}, '
                  f'but {key}={value!r} was requested; use overwrite=True to regenerate them.')


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

    The slab is read from relaxation/CONTCAR (relaxed) if it exists, otherwise from the POSCAR of the folder. The surface normal is
    computed from the lattice, the top layer is decomposed with a periodic Voronoi tessellation into
    atop, bridge and hollow sites, and symmetry-equivalent sites are removed (see
    ``voronoi_adsorption_sites``). One folder is written per site (and per orientation for H2):
    ``<sample_label>_<index>_<kind>[_<orientation>]``, with a POSCAR and a copy of KPOINTS, POTCAR,
    INCAR and run.sh if they exist in the surface folder; the same files are copied into the ``relaxation`` folder
    of each configuration, which is where its results go. A ``sites.json`` summary is also written.

    Args:
        surface_dir (str): Directory of the slab (relaxation/CONTCAR or POSCAR, and optional KPOINTS/POTCAR/INCAR/run.sh).
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
    # Validate before touching the disk: an invalid value must not delete existing configurations (overwrite=True)
    if surface not in ('top', 'bottom'):
        raise ValueError(f"surface must be 'top' or 'bottom', got {surface!r}")

    surface_dir = os.path.abspath(surface_dir)
    structure_path = vasp_files.get_structure_file(surface_dir)
    if structure_path is None:
        raise FileNotFoundError(f'Neither relaxation/CONTCAR nor POSCAR found in surface directory: {surface_dir}')

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(surface_dir), 'H-absorption')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    existing = sorted(os.path.join(output_dir, name) for name in os.listdir(output_dir)
                      if name.startswith(sample_label) and os.path.isdir(os.path.join(output_dir, name)))
    if existing and not overwrite:
        print(f'{len(existing)} configurations already exist in {output_dir}; reusing them (overwrite=True regenerates them).')
        _warn_if_settings_differ(output_dir, adsorbate, surface, adsorption_height)
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

            vasp_files.prepare_relaxation_folder(config_dir)

            summary[config_name] = {'kind': site['kind'], 'orientation': orientation_name or None,
                                    'cartesian': [float(x) for x in site['position']]}

    with open(os.path.join(output_dir, 'sites.json'), 'w') as json_file:
        json.dump({'surface': surface, 'adsorbate': adsorbate, 'adsorption_height': adsorption_height,
                   'configurations': summary}, json_file, indent=2)

    return created_dirs


# ---------------------------------------------------------------------------------------------------------------------
# adsorption FREE energy: dG_ads = G_conf - (n G_surf + G_ref)
# ---------------------------------------------------------------------------------------------------------------------

def h2_vibration(
        gas_folder,
        model_load_path=None,
        device=None,
        omega_meV=None
):
    """Harmonic stretching energy (meV) of the isolated H2 molecule in `gas_folder` (relaxed in ``relaxation/``).

    `omega_meV` given (for instance from a DFT frequency calculation): it is returned as it is. Otherwise it is computed with
    the ML-IAP from the relaxed molecule by finite differences (``mlip.diatomic_vibration``) and cached in
    ``relaxation/h2_vibration.json`` (recomputed if the model or the relaxed structure change).

    Returns a dict with 'omega_meV' and its 'source'.
    """
    import libraries.mlip as mlip
    model_load_path = mlip.DEFAULT_MODEL if model_load_path is None else model_load_path
    device = mlip.DEFAULT_DEVICE if device is None else device
    if omega_meV is not None:
        return {'omega_meV': float(omega_meV), 'source': 'given'}

    relax_dir = vasp_files.relaxation_dir(gas_folder)
    contcar = os.path.join(relax_dir, 'CONTCAR')
    if not os.path.exists(contcar):
        raise FileNotFoundError(f'{gas_folder} has no relaxation/CONTCAR: relax the H2 molecule first (energy_lookup.ensure_relaxed)')
    cache = os.path.join(relax_dir, 'h2_vibration.json')
    stamp = {'model': model_load_path, 'contcar_mtime_ns': os.stat(contcar).st_mtime_ns}
    if os.path.exists(cache):
        with open(cache) as json_file:
            cached = json.load(json_file)
        if cached.get('stamp') == stamp:
            return {'omega_meV': cached['omega_meV'], 'source': 'ML-IAP (cached)'}
    result = mlip.diatomic_vibration(contcar, model_load_path=model_load_path, device=device)
    with open(cache, 'w') as json_file:
        json.dump({'stamp': stamp, **result}, json_file, indent=2)
    return {'omega_meV': result['omega_meV'], 'source': 'ML-IAP'}


def h2_gas_free_energy(
        gas_folder,
        operating_T,
        model_load_path=None,
        device=None,
        omega_meV=None,
        extra_terms=None
):
    """Free energy G_H2(T) (eV per molecule) of the isolated H2 reference, as a sum of terms, from its molecular vibration only.

        G_H2(T) = U0 + F_vib(T),     U0 = E0 of the relaxed molecule,     F_vib = hbar w / 2 + k T ln(1 - exp(-hbar w / k T))

    That is the free energy of the stretching mode as a harmonic oscillator (zero-point energy included) around the relaxed
    molecule. U0 is the energy of the relaxed molecule because there is no MD of the gas to take it from, and in the harmonic
    model of one mode it is the energy at the bottom of the well. NOT included: the translation and the rotation of the
    molecule (their entropy is about -0.4 eV at 300 K and 1 bar, and the reference pressure enters through the translational
    term), and the anharmonicity of the stretch. Both ends of an adsorption free energy then differ in what they contain; read
    the result as a vibrational free energy of adsorption with the gas referenced at its vibrational level.

    A further term (a translational/rotational free energy at a pressure, an ideal-gas term, ...) is one entry of `extra_terms`
    ({name: eV per molecule}) and is added to G_H2 without any other change.

    Args:
        gas_folder (str): Folder of the H2 molecule in a box, relaxed (``relaxation/``).
        operating_T (float): Temperature in K.
        omega_meV (float): Stretching energy in meV (None: from the ML-IAP, see ``h2_vibration``).
        extra_terms (dict): Additional free-energy terms per molecule, in eV.

    Returns:
        dict: 'G_eV' (sum), 'terms' (dict), 'omega_meV', 'omega_source'.
    """
    import libraries.energy_lookup as energy_lookup
    import libraries.mlip as mlip
    import libraries.thermal as thermal
    model_load_path = mlip.DEFAULT_MODEL if model_load_path is None else model_load_path
    device = mlip.DEFAULT_DEVICE if device is None else device
    vib = h2_vibration(gas_folder, model_load_path, device, omega_meV)
    e0 = energy_lookup.read_energy(gas_folder, model_load_path=model_load_path, device=device)
    if not np.isfinite(e0):
        raise thermal.MissingData(f'no energy for the H2 reference in {gas_folder}')
    terms = {'U0': float(e0), 'F_vib': float(thermal.f_oscillator(vib['omega_meV'], operating_T))}
    for name, value in (extra_terms or {}).items():
        if name in terms:
            raise ValueError(f'extra term {name!r} is already part of the H2 free energy')
        terms[name] = float(value)
    return {'G_eV': float(sum(terms.values())), 'terms': terms, 'omega_meV': vib['omega_meV'], 'omega_source': vib['source']}


def _conf_folders(adsorbate_dir):
    return [os.path.join(adsorbate_dir, name) for name in sorted(os.listdir(adsorbate_dir))
            if name.startswith('conf') and os.path.isdir(os.path.join(adsorbate_dir, name))]


def adsorption_free_energy(
        surface_dir,
        adsorbate_dirs,
        h2_gas_dir,
        operating_T,
        model_load_path=None,
        device=None,
        omega_h2_meV=None,
        extra_h2_terms=None,
        equil_ps=None,
        vdos_max_lag_ps=None,
        n_cell_dof=0
):
    """Adsorption free energy of every H / H2 configuration at `operating_T`, next to its 0 K value.

        dG_ads = G_conf - ( n G_surf + n_H/2 G_H2 )        reported per H atom ('H') or per H2 molecule ('H2')

    ``G`` of the surface and of every configuration is U0 + F_vib per atom from its own NPT runs (``thermal.thermal_terms``;
    the atoms are those of the cell of the configuration, so a configuration built on a supercell of the surface is handled
    by `n`, the number of surface cells it contains, from the volumes), and G_H2 is ``h2_gas_free_energy`` (molecular
    vibration only). The 0 K column is the static level of the same expression, G = E0 for every term. The contribution of every
    term of G to dG_ads is in a column ``d_<term>_eV`` (they add up to ``dG_ads_T_eV``), so the vibrational part
    (``d_F_vib_eV``: the change of zero-point energy and vibrational free energy on adsorption) is read directly.

    A configuration without a finished MD run gets NaN and a flag. Flags also report what the analysis of its run found (an H
    that desorbed, an H2 that dissociated, an H that hops between sites), runs at another temperature than `operating_T`, and
    a fixed or free cell that differs between the configuration and the surface.

    Args:
        surface_dir (str): Folder of the bare surface (with ``md/<T>K``).
        adsorbate_dirs (dict): {'H': folder with the 'conf*' subfolders of the H atom, 'H2': same for the molecule}.
        h2_gas_dir (str): Folder of the isolated H2 molecule.
        operating_T (float): Temperature in K.
        omega_h2_meV, extra_h2_terms: see ``h2_gas_free_energy``.

    Returns:
        pandas.DataFrame, sorted by adsorbate and dG_ads(T): columns adsorbate, conf, n_surface, n_H, dG_ads_0K_eV,
        dG_ads_T_eV, d_<term>_eV, T_run_K, flags, rank_0K, rank_T (within each adsorbate). ``df.attrs`` holds 'operating_T'
        and 'h2_reference' (the terms of G_H2, the frequency used and what is not in it).
    """
    import pandas as pd
    import libraries.energy_lookup as energy_lookup
    import libraries.mlip as mlip
    import libraries.thermal as thermal
    model_load_path = mlip.DEFAULT_MODEL if model_load_path is None else model_load_path
    device = mlip.DEFAULT_DEVICE if device is None else device
    kwargs = dict(equil_ps=equil_ps, vdos_max_lag_ps=thermal.VDOS_MAX_LAG_PS if vdos_max_lag_ps is None else vdos_max_lag_ps,
                  n_cell_dof=n_cell_dof)

    surf = thermal.thermal_terms(surface_dir, operating_T, **kwargs)            # MissingData: the surface needs a run
    surf_structure = vasp_files.read_structure(surface_dir)
    n_H_surf = int(round(surf_structure.composition['H'])) if 'H' in surf_structure.composition else 0
    e0_surf = energy_lookup.read_energy(surface_dir, model_load_path=model_load_path, device=device)

    gas = h2_gas_free_energy(h2_gas_dir, operating_T, model_load_path, device, omega_h2_meV, extra_h2_terms)
    e0_h2 = gas['terms']['U0']
    term_names = list(surf['terms'])
    for name in gas['terms']:
        if name not in term_names:
            term_names.append(name)

    rows = []
    for adsorbate, adsorbate_dir in adsorbate_dirs.items():
        for folder in _conf_folders(adsorbate_dir):
            row = {'adsorbate': adsorbate, 'conf': os.path.basename(folder), 'n_surface': np.nan, 'n_H': np.nan,
                   'dG_ads_0K_eV': np.nan, 'dG_ads_T_eV': np.nan, **{f'd_{t}_eV': np.nan for t in term_names},
                   'T_run_K': np.nan, 'flags': ''}
            rows.append(row)
            structure = vasp_files.read_structure(folder)
            if structure is None:
                row['flags'] = 'no structure'
                continue
            ratio = structure.volume / surf_structure.volume
            n_rep = int(round(ratio))
            if n_rep < 1 or abs(ratio - n_rep) > 1e-3:
                row['flags'] = f'volume ratio {ratio:.3f} to the bare surface is not a positive integer'
                continue
            n_H = (int(round(structure.composition['H'])) if 'H' in structure.composition else 0) - n_rep * n_H_surf
            if n_H < 1:
                row['flags'] = 'no adsorbed H atom'
                continue
            per = n_H if adsorbate == 'H' else n_H / 2.0
            row['n_surface'], row['n_H'] = n_rep, n_H

            e0 = energy_lookup.read_energy(folder, model_load_path=model_load_path, device=device)
            if np.isfinite(e0) and np.isfinite(e0_surf):
                row['dG_ads_0K_eV'] = (e0 - n_rep * e0_surf - n_H / 2.0 * e0_h2) / per
            try:
                conf = thermal.thermal_terms(folder, operating_T, **kwargs)
            except thermal.MissingData as exc:
                row['flags'] = f'no finite-temperature result: {exc}'
                continue
            for name in term_names:
                g_conf = conf['n_atoms_cell'] * conf['terms'].get(name, 0.0)
                g_surf = n_rep * surf['n_atoms_cell'] * surf['terms'].get(name, 0.0)
                g_gas = n_H / 2.0 * gas['terms'].get(name, 0.0)
                row[f'd_{name}_eV'] = (g_conf - g_surf - g_gas) / per
            row['dG_ads_T_eV'] = float(sum(row[f'd_{name}_eV'] for name in term_names))
            row['T_run_K'] = conf['T_run_K']
            flags = list(conf['flags'])
            if conf['T_run_K'] != surf['T_run_K']:
                flags.append(f"the configuration uses the MD run at {conf['T_run_K']:g} K, the surface the one at {surf['T_run_K']:g} K")
            if conf['fix_cell'] != surf['fix_cell']:
                flags.append(f"the cell is treated differently in the runs (configuration {conf['fix_cell']!r}, surface {surf['fix_cell']!r})")
            row['flags'] = '; '.join(flags)

    columns = ['adsorbate', 'conf', 'n_surface', 'n_H', 'dG_ads_0K_eV', 'dG_ads_T_eV'] + [f'd_{t}_eV' for t in term_names] \
              + ['T_run_K', 'flags']
    df = pd.DataFrame(rows, columns=columns)
    for adsorbate in df['adsorbate'].unique():
        group = df[(df['adsorbate'] == adsorbate)]
        df.loc[group.index, 'rank_0K'] = group['dG_ads_0K_eV'].rank(method='min')
        df.loc[group.index, 'rank_T'] = group['dG_ads_T_eV'].rank(method='min')
    df = df.sort_values(['adsorbate', 'dG_ads_T_eV']).reset_index(drop=True)
    df.attrs['operating_T'] = operating_T
    df.attrs['h2_reference'] = {
        **gas, 'not_included': 'translation and rotation of the molecule, anharmonicity of the stretch' if not extra_h2_terms
        else 'anharmonicity of the stretch (the extra terms given are included)'}
    return df


def lowest_configurations(df):
    """The configuration of each adsorbate with the lowest dG_ads at the operating temperature and the one at 0 K (rows of
    the table of ``adsorption_free_energy``, with a column ``level`` saying which): they differ when the temperature changes the
    most stable site."""
    rows = []
    for adsorbate, group in df.groupby('adsorbate'):
        for level, column in (('T', 'dG_ads_T_eV'), ('0K', 'dG_ads_0K_eV')):
            valid = group.dropna(subset=[column])
            if len(valid):
                rows.append({'level': level, **valid.loc[valid[column].idxmin()].to_dict()})
    import pandas as pd
    return pd.DataFrame(rows)
