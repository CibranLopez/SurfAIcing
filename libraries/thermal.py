"""Finite-temperature surface free energies: NPT folders, MD analysis and the surface free energy gamma(T).

Every surface quantity of this library is a difference of free (Gibbs) energies G of the slab and of the bulk,

    gamma = [ G_slab - N g_bulk ] / (2 A)

and what changes between the 0 K ranking and the finite-temperature one is only the LEVEL at which G is evaluated, that is,
which contributions G is made of:

    level                     G per atom                      evaluated by
    static, 0 K               G = E0                          energy_lookup.read_energy
    vibrational, finite T     G(T) = U0(T) + F_vib(T)         thermal_terms (this module)

G is a sum of terms and nothing downstream depends on which ones: ``thermal_terms`` collects them in a dictionary and sums
them, and ``surfaces.get_surface_energy_of_formation`` only sees the total. A further contribution (a P V term, an electronic
or configurational entropy, ...) is one more term and needs no other change; none is included today. The static level has
neither zero-point energy nor entropy, so G(T -> 0) = E0 + F_vib(0 K) differs from E0 by the zero-point energy.

Finite-T level. G comes from NPT molecular dynamics with the effective-harmonic scheme (Hellman et al., PRB 87, 104111 (2013)),
the same one used for the anharmonic level of HeatUp:

    G(T) = U0(T) + F_vib[omega_eff(T)]                                          (per atom)

    U0(T)   = <U_MD> / N - (3N - 3) k T / (2 N)        mean MD potential energy minus the classical harmonic part
    F_vib   = int g(omega) [ omega/2 + k T ln(1 - exp(-omega/kT)) ] domega       quantum oscillators, zero-point included

``g(omega)`` is the vibrational density of states of the MD run: the Fourier transform of the mass-weighted velocity
autocorrelation, normalised to 3 modes per atom. It contains the anharmonic renormalisation of the frequencies at the
temperature and cell of the run ("effective" harmonic spectrum). U0 is the part of G that is not vibrational: the energy of
the effective-harmonic model at the structure and cell the run explores (it grows with T through the thermal expansion and
the anharmonic shifts), since the MD ions are classical and the classical harmonic thermal energy has been removed from <U_MD>.

The surface free energy is then (same expression as at 0 K, with the free energies and the area at temperature T)

    gamma(T) = [ G_slab(T) - N g_bulk(T) ] / (2 A(T)),        g_bulk per atom from an NPT run of the bulk.

E0 is not part of G(T). U0 comes straight from the NPT run, so G = U0 + F_vib makes no reference to the 0 K energy. E0 only
defines the static 0 K column of the ranking, which the finite-temperature column is compared with.

Operating temperature. The ranking can be evaluated at any temperature T_op, not only at the MD temperatures. F_vib is the only
term evaluated at T_op: it is computed exactly at T_op from the vibrational spectrum of the MD run closest in temperature, and
U0 and the area are those of that run. That assumes that the phonons (and U0, A) do not change between the temperature of the
run and T_op, a small approximation for a modest difference; a flag is raised when they differ by more than ``T_RUN_TOL_K``.
Run the MD at the temperature you need to remove the assumption.

The same G(T) is what the other quantities of the library are made of: the adsorption free energy
dG_ads = G_conf - (n G_surf + G_ref) (``libraries.adsorption``) and the adhesion free energy
dG_adh = (G_interface - G_film - G_substrate) / A (``libraries.heterostructure``). Each folder involved (every adsorbate
configuration, the interface, the film and the substrate in the interface cell) is prepared and run exactly like a slab, and
``thermal_terms`` gives its G; only the way the terms are combined changes.

Folder convention (inside the folder of every slab and of the bulk, and of every adsorbate configuration, interface, film and
substrate, next to its POSCAR and its ``relaxation/``; see ``libraries.vasp_files``. ``slab_data.json`` stays in the root and
is not copied here):

    <folder>/md/<T>K/POSCAR        supercell of the structure, NOT relaxed (the NPT run takes care of it)
    <folder>/md/<T>K/md.json       what the run is (plan, with ``equil_ps``, the ps discarded as thermalisation, ``fix_cell``
                                   and, for an adsorbate configuration, ``adsorbate``) and, once finished, how it was run
    <folder>/md/<T>K/output.traj   ML-IAP run (written by ``mlip.npt_dynamics``), or
    <folder>/md/<T>K/vasprun.xml   VASP run (ISIF = 3 NPT; every step stored)
    <folder>/md/<T>K/md_analysis.json   cached analysis (U0, area, spectrum, flags)

The cell is fully flexible during the run unless ``fix_cell`` says otherwise ('all' fixes every lattice vector, 'z' or 'xy'
fix one direction; see ``mlip.cell_mask``). The analysis only reports (flags) what happened to the cell, e.g. a change of the
length of c or a vacuum thinner than the range of the ML-IAP. A fixed cell is at constant volume: its free energy is the one of
that cell, so the film, the substrate and the interface of an adhesion free energy must all be run with the same ``fix_cell``.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

import numpy as np

import libraries.vasp_files as vasp_files

from pymatgen.core.structure import Structure

# ---------------------------------------------------------------------------------------------------------------
# constants and defaults
# ---------------------------------------------------------------------------------------------------------------
KB_EV          = 8.617333262e-5      # eV/K
KB_MEV         = 8.617333262e-2      # meV/K
H_MEV_FS       = 4135.667696         # Planck constant in meV*fs (frequency in cycles/fs -> energy in meV)
EV_A3_TO_GPA   = 160.21766208

OMEGA_MIN_MEV    = 0.1               # modes below this frequency (meV) are not counted in F_vib
MIN_LENGTH       = 15.0              # Angstrom, minimum length of the lattice vectors of an MD cell
EQUIL_PS         = 2.0               # ps discarded (thermalisation) at the start of a run, unless its md.json sets another value
VDOS_MAX_LAG_PS  = 4.0               # ps of the velocity autocorrelation used for the spectrum
MIN_PRODUCTION_PS = 5.0              # shorter production runs are flagged
EPOT_DRIFT_TOL   = 5e-4              # eV/atom, drift of the potential energy over the production run that is flagged
VACUUM_MIN_A     = 6.0               # a slab whose vacuum gets thinner than this is flagged (range of the MACE models)
C_CHANGE_TOL     = 0.10              # relative change of the length of c of a slab that is flagged
ADS_MAX_BOND_A   = 2.5               # an H atom further than this from every other atom has desorbed (flagged)
HH_MOLECULE_A    = 1.1               # H-H distances below this are a molecule (the H2 bond is 0.74 A)
ADS_HOP_A        = 1.5               # an adsorbed H that moves further than this relative to the surface hops between sites
T_RUN_TOL_K      = 1.0               # K, an operating temperature further than this from the MD run it uses is flagged


class MissingData(Exception):
    """The data a free energy needs are not (yet) there."""


# ---------------------------------------------------------------------------------------------------------------
# supercell and md folders
# ---------------------------------------------------------------------------------------------------------------

def supercell_for_min_length(
        structure,
        min_length=MIN_LENGTH,
        axes=(0, 1, 2)
):
    """Smallest diagonal supercell (n_a, n_b, n_c) whose lattice vectors along `axes` are all at least `min_length` long.

    The cell keeps its shape (a diagonal supercell never changes the angles). For a slab use ``axes=(0, 1)``: the
    thickness and the vacuum along c are left as generated.
    """
    lengths = structure.lattice.abc
    n = [1, 1, 1]
    for i in axes:
        n[i] = max(1, int(np.ceil(min_length / lengths[i] - 1e-8)))
    return tuple(n)


def md_run_dir(folder, temperature):
    return os.path.join(folder, 'md', f'{float(temperature):g}K')


def md_temperatures(folder):
    """Temperatures (K) of the ``md/<T>K`` folders of a structure folder, ascending."""
    root = os.path.join(folder, 'md')
    if not os.path.isdir(root):
        return []
    temps = []
    for name in os.listdir(root):
        if name.endswith('K') and os.path.isdir(os.path.join(root, name)):
            try:
                temps.append(float(name[:-1]))
            except ValueError:
                pass
    return sorted(temps)


def read_md_json(md_dir):
    path = os.path.join(md_dir, 'md.json')
    if not os.path.exists(path):
        return None
    with open(path) as json_file:
        return json.load(json_file)


def write_md_json(md_dir, data):
    with open(os.path.join(md_dir, 'md.json'), 'w') as json_file:
        json.dump(data, json_file, indent=2)


def _fix_cell_label(fix_cell):
    """Canonical form of ``fix_cell`` as stored in md.json: None, 'all' or the fixed axes as a sorted string ('xy', 'z', ...)."""
    import libraries.mlip as mlip
    mask = mlip.cell_mask(fix_cell)                       # validates it (None: fully flexible)
    if mask is None:
        return None
    fixed = ''.join(a for a, m in zip('xyz', mask) if m == 0)
    return 'all' if fixed == 'xyz' else fixed


def md_settings_of(folder, timestep=None, loginterval=None):
    """The MD settings of the runs already planned in `folder` (the first ``md/<T>K``), as keyword arguments of
    ``prepare_md_folders``: the way to give other structures (the adsorbate configurations of a surface) the settings of the
    surface and the same thermalisation time without defining them twice.

    The settings include ``fix_cell`` and the thermalisation time ``equil_ps``. With `timestep` or `loginterval` the run keeps its simulated time: ``n_steps`` is rescaled for the new time step. Hydrogen
    needs the shorter values (about 0.5 fs, and a frame every 2 fs or less so that the O-H stretch, up to ~450 meV, is below
    the Nyquist frequency of the spectrum).

    Returns {} if `folder` has no md runs (``prepare_md_folders`` then uses its own defaults).
    """
    temperatures = md_temperatures(folder)
    if not temperatures:
        return {}
    plan = read_md_json(md_run_dir(folder, temperatures[0]))
    settings = {'n_steps': plan['n_steps'], 'timestep': plan['timestep_fs'], 'ttime': plan['ttime_fs'],
                'ptime': plan['ptime_fs'], 'barostat_modulus': plan['barostat_modulus_GPa'],
                'loginterval': plan['loginterval'], 'pressure': plan['pressure_GPa'],
                'min_length': plan.get('min_length_A', MIN_LENGTH), 'equil_ps': resolve_equil_ps(plan),
                'fix_cell': plan.get('fix_cell')}
    if timestep is not None:
        settings['n_steps'] = int(round(plan['n_steps'] * plan['timestep_fs'] / timestep))
        settings['timestep'] = float(timestep)
    if loginterval is not None:
        settings['loginterval'] = int(loginterval)
    return settings


def prepare_md_folders(
        source_folder,
        kind,
        temperatures,
        min_length=MIN_LENGTH,
        n_steps=20000,
        timestep=1.0,
        ttime=50.0,
        ptime=500.0,
        barostat_modulus=1.0,
        loginterval=10,
        pressure=0.0,
        overwrite=False,
        max_atoms=1000,
        equil_ps=EQUIL_PS,
        fix_cell=None,
        adsorbate=None,
        start_from='POSCAR'
):
    """Create ``<source_folder>/md/<T>K`` with the supercell POSCAR and the plan of one NPT run per temperature.

    The structure is the (unrelaxed) POSCAR of `source_folder` (relaxation/CONTCAR if there is no POSCAR), or its relaxed
    structure with ``start_from='relaxed'``. It is repeated until the lattice vectors are at least `min_length` long: a and b
    only for a slab (any structure with a vacuum: slab, adsorbate configuration, interface, film, substrate), all three for the
    bulk. A POTCAR present in `source_folder` is copied (the species order does not change).

    Args:
        source_folder (str): Folder of a slab, of the bulk, of an adsorbate configuration, of an interface, ...
        kind (str): 'slab' or 'bulk'.
        temperatures (sequence): MD temperatures in K.
        min_length (float): Minimum lattice vector length in Angstrom.
        n_steps, timestep, ttime, ptime, barostat_modulus, loginterval, pressure: see ``mlip.npt_dynamics``.
        equil_ps (float): thermalisation time in ps: the first `equil_ps` ps of the run are discarded when it is analysed.
            Stored in md.json; for an existing folder that is reused it is updated (it is not a parameter of the run).
        fix_cell (str): None (fully flexible cell), 'all' or the axes kept fixed ('z', 'xy', ...): see ``mlip.cell_mask``.
            Stored in md.json. It is a parameter of the run: an existing folder that was planned with another value is
            kept and a warning is printed (overwrite=True regenerates it).
        adsorbate (str): 'H' or 'H2' for an adsorbate configuration (the H atoms are checked during the analysis: a
            desorbed H, a dissociated H2); None otherwise.
        start_from (str): 'POSCAR' (default, the unrelaxed structure; the NPT run takes care of it) or 'relaxed' (the
            ``relaxation/CONTCAR``, which must exist). Adsorbate configurations and interfaces start from the relaxed
            structure so that the run stays in the configuration that was built (an H atom placed 1.5 A above the surface
            can otherwise leave its site while thermalising).
        overwrite (bool): Existing md folders are kept (reused) unless overwrite=True, which regenerates them.
        max_atoms (int): A warning is printed if the supercell has more atoms.

    Returns:
        list[str]: the md folders.
    """
    from ase.io.vasp import write_vasp
    from pymatgen.io.ase import AseAtomsAdaptor

    if kind not in ('slab', 'bulk'):
        raise ValueError(f"kind must be 'slab' or 'bulk', got {kind!r}")
    if start_from not in ('POSCAR', 'relaxed'):
        raise ValueError(f"start_from must be 'POSCAR' or 'relaxed', got {start_from!r}")
    fix_cell = _fix_cell_label(fix_cell)
    relaxed = os.path.join(vasp_files.RELAXATION_DIR, 'CONTCAR')
    source = None
    for candidate in ((relaxed,) if start_from == 'relaxed' else ('POSCAR', relaxed)):
        if os.path.exists(os.path.join(source_folder, candidate)):
            source = os.path.join(source_folder, candidate)
            break
    if source is None:
        raise FileNotFoundError(f'{source_folder} has no ' + ('relaxation/CONTCAR (relax it first)' if start_from == 'relaxed'
                                                               else 'POSCAR nor relaxation/CONTCAR'))

    structure = Structure.from_file(source)
    n = supercell_for_min_length(structure, min_length, axes=(0, 1) if kind == 'slab' else (0, 1, 2))
    supercell = structure.copy()
    supercell.make_supercell(list(n))
    if len(supercell) > max_atoms:
        print(f'[warning] {source_folder}: the MD cell has {len(supercell)} atoms (> {max_atoms}); '
              f'the runs will be slow. Lower min_length if that is acceptable.')

    folders = []
    for temperature in temperatures:
        md_dir = md_run_dir(source_folder, temperature)
        if os.path.exists(os.path.join(md_dir, 'md.json')) and not overwrite:
            plan = read_md_json(md_dir)
            if plan.get('fix_cell') != fix_cell:
                print(f"[warning] {md_dir} was planned with fix_cell={plan.get('fix_cell')!r}, not {fix_cell!r}: it is kept as "
                      f"it is (overwrite=True regenerates it).")
            if resolve_equil_ps(plan) != float(equil_ps):
                plan['equil_ps'] = float(equil_ps)
                write_md_json(md_dir, plan)
                print(f'{md_dir} already exists; reusing it, with the thermalisation time set to {equil_ps:g} ps '
                      f'(overwrite=True regenerates it).')
            else:
                print(f'{md_dir} already exists; reusing it (overwrite=True regenerates it).')
            folders.append(md_dir)
            continue
        if os.path.isdir(md_dir):
            shutil.rmtree(md_dir)
        os.makedirs(md_dir)

        write_vasp(os.path.join(md_dir, 'POSCAR'), AseAtomsAdaptor.get_atoms(supercell), direct=True, sort=True)
        potcar = os.path.join(source_folder, 'POTCAR')
        if os.path.exists(potcar):
            shutil.copy(potcar, os.path.join(md_dir, 'POTCAR'))

        write_md_json(md_dir, {
            'kind': kind, 'temperature_K': float(temperature), 'pressure_GPa': float(pressure),
            'timestep_fs': float(timestep), 'ttime_fs': float(ttime), 'ptime_fs': float(ptime),
            'barostat_modulus_GPa': float(barostat_modulus), 'n_steps': int(n_steps), 'loginterval': int(loginterval),
            'frame_dt_fs': float(timestep) * int(loginterval), 'min_length_A': float(min_length),
            'supercell': [int(i) for i in n], 'n_atoms_cell': len(structure), 'n_atoms': len(supercell),
            'equil_ps': float(equil_ps), 'fix_cell': fix_cell, 'adsorbate': adsorbate, 'start_from': start_from,
            'finished': False})
        folders.append(md_dir)
    return folders


def _is_vasp_run(md_dir):
    return any(os.path.exists(os.path.join(md_dir, f)) for f in ('vasprun.xml', 'vasprun.xml.gz'))


def _vasprun_path(md_dir):
    for f in ('vasprun.xml', 'vasprun.xml.gz'):
        if os.path.exists(os.path.join(md_dir, f)):
            return os.path.join(md_dir, f)
    return None


def _vasprun_complete(path):
    """A finished vasprun.xml ends with the closing tag; cheap check of the last bytes."""
    try:
        if path.endswith('.gz'):
            import gzip
            with gzip.open(path, 'rb') as f:
                tail = f.read()[-200:]
        else:
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 200))
                tail = f.read()
        return b'</modeling>' in tail
    except OSError:
        return False


def md_state(md_dir):
    """'missing' (no md.json), 'pending' (planned, not run), 'unfinished', or 'done'. A VASP run is recognised by its vasprun.xml."""
    plan = read_md_json(md_dir)
    if plan is None:
        return 'missing'
    vasprun = _vasprun_path(md_dir)
    if vasprun is not None:
        return 'done' if _vasprun_complete(vasprun) else 'unfinished'
    if os.path.exists(os.path.join(md_dir, 'output.traj')):
        return 'done' if plan.get('finished') else 'unfinished'
    return 'pending'


def structure_folders(slab_folder):
    """The bulk folder and the slab folders (those with slab_data.json) of a ``slab_v<i>`` folder."""
    names = sorted(name for name in os.listdir(slab_folder)
                   if name != 'bulk' and os.path.isfile(os.path.join(slab_folder, name, 'slab_data.json')))
    return os.path.join(slab_folder, 'bulk'), names


def md_status(slab_folder):
    """Table of every planned MD run of a ``slab_v<i>`` folder: structure, T, state, atoms, code."""
    bulk_folder, names = structure_folders(slab_folder)
    return md_status_of([('bulk', bulk_folder)] + [(n, os.path.join(slab_folder, n)) for n in names])


def md_status_of(named_folders):
    """Table of every planned MD run of the structure folders given as (name, folder) pairs: structure, T, state, atoms, code."""
    import pandas as pd
    rows = []
    for name, folder in named_folders:
        for temperature in md_temperatures(folder):
            md_dir = md_run_dir(folder, temperature)
            plan = read_md_json(md_dir) or {}
            rows.append({'structure': name, 'T_K': temperature, 'state': md_state(md_dir),
                         'code': 'VASP' if _is_vasp_run(md_dir) else ('MACE' if plan.get('finished') else None),
                         'n_atoms': plan.get('n_atoms'), 'supercell': plan.get('supercell')})
    return pd.DataFrame(rows, columns=['structure', 'T_K', 'state', 'code', 'n_atoms', 'supercell'])


def run_md(
        md_dir,
        model_load_path=None,
        device=None,
        dispersion=False,
        seed=None
):
    """Run the NPT simulation planned in `md_dir` with the ML-IAP (a 'pending' or 'unfinished' run; unfinished ones restart)."""
    import libraries.mlip as mlip
    model_load_path = mlip.DEFAULT_MODEL if model_load_path is None else model_load_path
    device = mlip.DEFAULT_DEVICE if device is None else device

    plan = read_md_json(md_dir)
    if plan is None:
        raise FileNotFoundError(f'{md_dir} has no md.json (create it with prepare_md_folders)')
    if _is_vasp_run(md_dir):
        raise ValueError(f'{md_dir} contains a VASP run; refusing to run the ML-IAP there')
    plan['finished'] = False
    write_md_json(md_dir, plan)
    for stale in ('output.traj', 'md_analysis.json'):
        if os.path.exists(os.path.join(md_dir, stale)):
            os.remove(os.path.join(md_dir, stale))

    _, info = mlip.npt_dynamics(os.path.join(md_dir, 'POSCAR'), model_load_path=model_load_path, device=device,
                                dispersion=dispersion, temperature=plan['temperature_K'],
                                pressure=plan['pressure_GPa'], timestep=plan['timestep_fs'], ttime=plan['ttime_fs'],
                                ptime=plan['ptime_fs'], barostat_modulus=plan['barostat_modulus_GPa'],
                                n_steps=plan['n_steps'], loginterval=plan['loginterval'], seed=seed,
                                fix_cell=plan.get('fix_cell'), output_folder=md_dir)
    plan.update(info)
    plan.update({'code': 'MACE', 'finished': True})
    write_md_json(md_dir, plan)
    return md_dir


def run_pending_md(
        slab_folder,
        model_load_path=None,
        device=None,
        temperatures=None
):
    """Run with the ML-IAP every pending/unfinished MD run of the bulk and of the slabs (the bulk first)."""
    bulk_folder, names = structure_folders(slab_folder)
    return run_pending_md_in([bulk_folder] + [os.path.join(slab_folder, n) for n in names],
                             model_load_path=model_load_path, device=device, temperatures=temperatures)


def run_pending_md_in(
        folders,
        model_load_path=None,
        device=None,
        temperatures=None
):
    """Run with the ML-IAP every pending/unfinished MD run of the structure folders `folders` (any folders with ``md/<T>K``:
    adsorbate configurations, interfaces, ...), in the order given; runs made with VASP are skipped."""
    todo = []
    for folder in folders:
        for temperature in md_temperatures(folder):
            if temperatures is not None and temperature not in temperatures:
                continue
            md_dir = md_run_dir(folder, temperature)
            if md_state(md_dir) in ('pending', 'unfinished') and not _is_vasp_run(md_dir):
                todo.append(md_dir)
    for i, md_dir in enumerate(todo, 1):
        print(f'[{i}/{len(todo)}] NPT run in {md_dir}')
        run_md(md_dir, model_load_path=model_load_path, device=device)
    if not todo:
        print('No pending MD runs.')
    return todo


# ---------------------------------------------------------------------------------------------------------------
# reading a finished run (ML-IAP: output.traj; VASP: vasprun.xml)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class MDData:
    """Production part of a run. Shapes: frac (T, N, 3) unwrapped fractional coordinates, cell (T, 3, 3) with the
    lattice vectors as rows, mom (T, N, 3) momenta (amu*A/fs), epot (T,) potential energy in eV."""
    frac: np.ndarray
    cell: np.ndarray
    mom: np.ndarray
    masses: np.ndarray
    symbols: list
    epot: np.ndarray
    dt_fs: float
    temperature_K: float
    code: str
    momenta_from: str            # 'stored' (ML-IAP) or 'finite_difference' (VASP: positions only)
    initial_cell: np.ndarray
    initial_frac: np.ndarray

    @property
    def n_frames(self):
        return self.frac.shape[0]

    @property
    def n_atoms(self):
        return self.frac.shape[1]

    @property
    def duration_ps(self):
        return self.n_frames * self.dt_fs / 1000.0


def unwrap_frac(frac):
    """Cumulative minimum-image unwrapping of fractional coordinates (T, N, 3)."""
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    return np.concatenate([frac[:1], frac[:1] + np.cumsum(d, axis=0)], axis=0)


def _read_ase_run(md_dir, plan):
    from ase.io.trajectory import Trajectory
    path = os.path.join(md_dir, 'output.traj')
    frac, cell, mom, epot = [], [], [], []
    masses = symbols = None
    with Trajectory(path) as traj:
        for atoms in traj:
            if masses is None:
                masses, symbols = atoms.get_masses(), atoms.get_chemical_symbols()
            frac.append(atoms.get_scaled_positions(wrap=False))
            cell.append(atoms.cell.array.copy())
            mom.append(atoms.get_momenta())
            epot.append(atoms.get_potential_energy())
    frac = unwrap_frac(np.array(frac))
    return dict(frac=frac, cell=np.array(cell), mom=np.array(mom), masses=masses, symbols=symbols,
                epot=np.array(epot, float), dt_fs=float(plan['frame_dt_fs']),
                temperature_K=float(plan['temperature_K']), code='MACE', momenta_from='stored')


def _read_vasp_run(md_dir, plan):
    """vasprun.xml of a VASP MD run. Velocities are not stored, so they are taken from the positions."""
    from pymatgen.io.vasp.outputs import Vasprun
    vr = Vasprun(_vasprun_path(md_dir), parse_dos=False, parse_eigen=False, parse_potcar_file=False)
    steps = vr.ionic_steps
    if len(steps) < 8:
        raise ValueError(f'only {len(steps)} ionic steps in {md_dir}')
    frac = unwrap_frac(np.array([s['structure'].frac_coords for s in steps]))
    cell = np.array([s['structure'].lattice.matrix for s in steps])
    epot = np.array([s['e_0_energy'] for s in steps], float)
    species = steps[0]['structure'].species
    masses = np.array([float(sp.atomic_mass) for sp in species])
    symbols = [sp.symbol for sp in species]
    dt_fs = float(plan.get('vasp_frame_dt_fs', vr.parameters.get('POTIM', 1.0)))
    temperature = float(vr.parameters.get('TEBEG', plan['temperature_K']))

    # central-difference velocities of the fractional coordinates on the mean cell (the motion of the cell itself is
    # not part of the phonon spectrum); frames 1..T-2 keep a velocity
    h_mean = cell.mean(axis=0)
    v = (frac[2:] - frac[:-2]) / (2.0 * dt_fs) @ h_mean                      # A/fs
    masses_amu = masses[None, :, None]
    v = v - (v * masses_amu).sum(axis=1, keepdims=True) / masses.sum()       # remove centre-of-mass velocity
    return dict(frac=frac[1:-1], cell=cell[1:-1], mom=v * masses_amu, masses=masses, symbols=symbols,
                epot=epot[1:-1], dt_fs=dt_fs, temperature_K=temperature, code='VASP',
                momenta_from='finite_difference')


def resolve_equil_ps(plan, equil_ps=None):
    """Thermalisation time (ps discarded at the start of a run): `equil_ps` if given, else the ``equil_ps`` of the plan of
    the run (md.json, set in ``prepare_md_folders``), else ``EQUIL_PS``."""
    if equil_ps is not None:
        return float(equil_ps)
    value = (plan or {}).get('equil_ps')
    return EQUIL_PS if value is None else float(value)


def read_md_run(md_dir, equil_ps=None):
    """Read the production part (first `equil_ps` ps discarded, default: the ``equil_ps`` of md.json) of the run in `md_dir`,
    MACE or VASP."""
    plan = read_md_json(md_dir)
    if plan is None:
        raise MissingData(f'{md_dir} has no md.json')
    equil_ps = resolve_equil_ps(plan, equil_ps)
    state = md_state(md_dir)
    if state != 'done':
        raise MissingData(f'{md_dir}: run is {state}')
    d = _read_vasp_run(md_dir, plan) if _is_vasp_run(md_dir) else _read_ase_run(md_dir, plan)
    n_equil = min(int(round(equil_ps * 1000.0 / d['dt_fs'])), d['frac'].shape[0] - 2)
    sl = slice(n_equil, None)
    return MDData(frac=d['frac'][sl], cell=d['cell'][sl], mom=d['mom'][sl], masses=d['masses'],
                  symbols=d['symbols'], epot=d['epot'][sl], dt_fs=d['dt_fs'], temperature_K=d['temperature_K'],
                  code=d['code'], momenta_from=d['momenta_from'], initial_cell=d['cell'][0],
                  initial_frac=d['frac'][0] % 1.0)


# ---------------------------------------------------------------------------------------------------------------
# vibrational density of states from the velocity autocorrelation, and the quantum F_vib of the spectrum
# ---------------------------------------------------------------------------------------------------------------

def _vacf_spectrum(x, dt_fs, max_lag, pad=4):
    """x: (T, M) mass-weighted velocities (p / sqrt(m)). Returns (omega_meV, S) for omega >= 0."""
    T = x.shape[0]
    n = 1 << (2 * T - 1).bit_length()
    X = np.fft.rfft(x, n=n, axis=0)
    ac = np.fft.irfft((X * np.conj(X)).sum(axis=1), n=n)[:max_lag]
    ac = ac / (T - np.arange(max_lag))                                       # unbiased
    win = 0.5 * (1.0 + np.cos(np.pi * np.arange(max_lag) / max_lag))         # Hann lag window
    ac = ac * win
    L = 1 << ((max_lag * pad) - 1).bit_length()
    S = 2.0 * np.fft.rfft(ac, n=L).real - ac[0]                              # even function: S = c0 + 2 sum c_k cos
    freq = np.fft.rfftfreq(L, d=dt_fs)                                       # cycles/fs
    return freq * H_MEV_FS, S


def compute_vdos(data, max_lag_ps=VDOS_MAX_LAG_PS):
    """Vibrational DOS of a production run: mass-weighted VACF, Hann window, normalised to 3 modes per atom.

    Returns a dict with ``omega_meV``, ``weight`` (modes per atom, sum 3), the resolution and the weight close to the
    Nyquist frequency (non-zero weight there means that the frames are stored too far apart).
    """
    if data.n_frames < 64:
        raise ValueError(f'only {data.n_frames} production frames')
    # the momenta units (ASE time units for the ML-IAP, fs for VASP) only rescale the spectrum, which is normalised below
    x = data.mom / np.sqrt(data.masses)[None, :, None]
    x = x - x.mean(axis=0, keepdims=True)
    max_lag = int(min(data.n_frames // 2, max_lag_ps * 1000.0 / data.dt_fs))
    omega, S = _vacf_spectrum(x.reshape(data.n_frames, -1), data.dt_fs, max_lag)

    nyquist = H_MEV_FS / (2.0 * data.dt_fs)
    resolution = 2.0 * H_MEV_FS / (max_lag * data.dt_fs)
    keep = omega <= nyquist * 0.999
    omega, S = omega[keep], S[keep]

    if data.momenta_from == 'finite_difference':                             # undo the sinc^2 of the central difference
        xx = np.clip(2.0 * np.pi * omega / H_MEV_FS * data.dt_fs, 1e-9, 1.2)
        S = S / (np.sin(xx) / xx) ** 2

    step = max(resolution / 4.0, 0.1)
    grid = np.arange(0.0, omega.max(), step)
    g = np.clip(np.interp(grid, omega, S), 0.0, None)
    w = g * np.gradient(grid)
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError('the DOS has no positive weight')
    w = w * (3.0 / w.sum())
    return {'omega_meV': grid, 'weight': w, 'resolution_meV': float(resolution), 'nyquist_meV': float(nyquist),
            'alias_weight_top10pct': float(w[grid > 0.9 * nyquist].sum() / w.sum())}


def f_oscillator(omega_meV, temperature):
    """Quantum harmonic free energy (eV) of ONE vibrational mode of energy `omega_meV` (meV) at `temperature` (K): omega/2 +
    k T ln(1 - exp(-omega/kT)); the zero-point energy omega/2 at T = 0. The same expression as ``f_vib`` for a single mode."""
    return f_vib([omega_meV], [1.0], temperature) if omega_meV > OMEGA_MIN_MEV else 0.0


def f_vib(omega_meV, weight, temperature):
    """Quantum harmonic vibrational free energy F_vib per atom (eV) of a spectrum (modes with omega <= OMEGA_MIN_MEV skipped).

    At T = 0 it is the zero-point energy, sum of g(omega) omega / 2."""
    omega, weight = np.asarray(omega_meV, float), np.asarray(weight, float)
    m = omega > OMEGA_MIN_MEV
    om, w = omega[m], weight[m]
    if temperature <= 0:
        return float((w * 0.5 * om).sum() * 1e-3)
    x = np.clip(om / (KB_MEV * temperature), 1e-10, 700.0)
    return float((w * (0.5 * om + KB_MEV * temperature * np.log1p(-np.exp(-x)))).sum() * 1e-3)


# ---------------------------------------------------------------------------------------------------------------
# analysis of one run (cached)
# ---------------------------------------------------------------------------------------------------------------

def _slab_vacuum_A(data, n_samples=100):
    """Thickness of the vacuum (largest gap of the atoms along the surface normal, periodic) in a sample of frames."""
    idx = np.unique(np.linspace(0, data.n_frames - 1, min(n_samples, data.n_frames)).astype(int))
    out = []
    for k in idx:
        a, b = data.cell[k][0], data.cell[k][1]
        height = abs(np.linalg.det(data.cell[k])) / np.linalg.norm(np.cross(a, b))      # distance between ab planes
        s = np.sort(data.frac[k][:, 2] % 1.0)
        gaps = np.diff(np.r_[s, s[0] + 1.0])
        out.append(gaps.max() * height)
    return np.array(out)


def _adsorbate_flags(data, adsorbate, n_samples=60):
    """Flags for the H atoms of an adsorbate configuration during the production run (the harmonic spectrum needs atoms
    that vibrate around their sites): desorption, H2 formed from H atoms or dissociated, hopping between sites.

    Distances use the minimum image of the fractional coordinates, which is exact for the cells of at least 15 A along a and b
    that ``prepare_md_folders`` makes and a good approximation along c (a vacuum thicker than the distances tested).
    """
    symbols = np.array(data.symbols)
    h = np.flatnonzero(symbols == 'H')
    rest = np.flatnonzero(symbols != 'H')
    if len(h) == 0:
        return [f'adsorbate {adsorbate!r}: no H atom in the structure']
    idx = np.unique(np.linspace(0, data.n_frames - 1, min(n_samples, data.n_frames)).astype(int))
    nearest_other = np.full(len(h), 0.0)
    nearest_h = np.full(len(h), 0.0)
    for k in idx:
        frac, cell = data.frac[k] % 1.0, data.cell[k]
        d = frac[h][:, None, :] - frac[rest][None, :, :]
        d -= np.round(d)
        nearest_other = np.maximum(nearest_other, np.linalg.norm(d @ cell, axis=-1).min(axis=1))
        if len(h) > 1:
            dh = frac[h][:, None, :] - frac[h][None, :, :]
            dh -= np.round(dh)
            dist = np.linalg.norm(dh @ cell, axis=-1)
            np.fill_diagonal(dist, np.inf)
            nearest_h = np.maximum(nearest_h, dist.min(axis=1)) if adsorbate == 'H2' else nearest_h
            if adsorbate == 'H' and dist.min() < HH_MOLECULE_A:
                return [f'H atoms came within {dist.min():.2f} Å of each other: H2 formed during the run']
    flags = []
    if adsorbate == 'H':
        if nearest_other.max() > ADS_MAX_BOND_A:
            flags.append(f'an H atom was {nearest_other.max():.1f} Å from every other atom (> {ADS_MAX_BOND_A} Å): desorbed')
    elif adsorbate == 'H2' and len(h) > 1 and nearest_h.max() > HH_MOLECULE_A:
        flags.append(f'an H atom was {nearest_h.max():.2f} Å from its nearest H (> {HH_MOLECULE_A} Å): the H2 molecule dissociated')
    # hopping: displacement of each H relative to the other atoms, from the unwrapped fractional coordinates
    disp = (data.frac[idx] - data.frac[idx[0]]) @ data.cell.mean(axis=0)
    relative = disp[:, h, :] - disp[:, rest, :].mean(axis=1, keepdims=True)
    travelled = np.linalg.norm(relative, axis=-1).max()
    if travelled > ADS_HOP_A:
        flags.append(f'an H atom moved {travelled:.1f} Å relative to the surface (> {ADS_HOP_A} Å): it hops between sites, '
                     f'so the harmonic spectrum is not a good description of it')
    return flags


def analyse_md_run(
        md_dir,
        equil_ps=None,
        vdos_max_lag_ps=VDOS_MAX_LAG_PS,
        n_cell_dof=0,
        force=False
):
    """Everything the free energy needs from one finished run; cached in ``md_analysis.json``.

    ``equil_ps`` is the thermalisation time discarded at the start of the run; None (default) uses the one of md.json.

    The result holds what the free energy G(T) = U0 + F_vib of the run is made of (``u0_eV_per_atom``, the spectrum
    ``vdos`` for F_vib, the mean area) and the flags raised by the quality checks.

    ``n_cell_dof``: number of cell degrees of freedom whose kT/2 each is subtracted from <U> in U0 on top of the
    3N - 3 atomic ones. A fluctuating cell (NPT) stores potential energy in the strain too, so counting the strain modes
    (up to 6 for an upper-triangular cell) would be the stricter equipartition count; the default 0 keeps the HeatUp
    prescription (atomic 3N - 3 only). The per-atom effect of the extra modes goes as 1/N, so it does not cancel between a
    slab and a bulk cell of different size.
    """
    plan = read_md_json(md_dir)
    equil_ps = resolve_equil_ps(plan, equil_ps)
    source = _vasprun_path(md_dir) or os.path.join(md_dir, 'output.traj')
    stamp = {'equil_ps': equil_ps, 'vdos_max_lag_ps': vdos_max_lag_ps, 'n_cell_dof': n_cell_dof,
             'adsorbate': plan.get('adsorbate'),
             'source_mtime_ns': os.stat(source).st_mtime_ns, 'source_size': os.path.getsize(source)}
    cache = os.path.join(md_dir, 'md_analysis.json')
    if not force and os.path.exists(cache):
        with open(cache) as json_file:
            cached = json.load(json_file)
        if cached.get('stamp') == stamp:
            return cached

    data = read_md_run(md_dir, equil_ps)
    n, T = data.n_atoms, data.temperature_K
    n_dof = 3 * n - 3 + int(n_cell_dof)
    t_ps = np.arange(data.n_frames) * data.dt_fs / 1000.0
    epot_mean = float(np.mean(data.epot))
    drift = float(np.polyfit(t_ps, data.epot, 1)[0]) if data.n_frames > 3 else 0.0
    ke = 0.5 * ((data.mom ** 2).sum(-1) / data.masses[None, :]).sum(-1)
    if data.momenta_from == 'stored':
        kinetic_T = float(np.mean(2.0 * ke / ((3 * n - 3) * KB_EV)))        # ASE momenta: eV units
    else:
        kinetic_T = None                                                    # finite-difference velocities: not reliable

    sc = plan.get('supercell', [1, 1, 1])
    a, b = data.cell[:, 0, :], data.cell[:, 1, :]
    area_cell = np.linalg.norm(np.cross(a, b), axis=1) / (sc[0] * sc[1])    # area of one original cell, A^2
    lengths = np.linalg.norm(data.cell, axis=2) / np.array(sc)[None, :]
    lengths0 = np.linalg.norm(data.initial_cell, axis=1) / np.array(sc)

    vdos = compute_vdos(data, vdos_max_lag_ps)
    result = {
        'stamp': stamp, 'kind': plan.get('kind'), 'fix_cell': plan.get('fix_cell'), 'adsorbate': plan.get('adsorbate'),
        'code': data.code, 'momenta_from': data.momenta_from,
        'temperature_K': T, 'n_atoms': n, 'n_atoms_cell': plan.get('n_atoms_cell', n), 'supercell': sc,
        'production_ps': data.duration_ps, 'n_frames': data.n_frames,
        'epot_mean_eV': epot_mean, 'epot_drift_eV_per_ps': drift,
        'u0_eV_per_atom': (epot_mean - 0.5 * n_dof * KB_EV * T) / n,
        'kinetic_T_K': kinetic_T,
        'area_A2': float(area_cell.mean()), 'area_std_A2': float(area_cell.std()),
        'volume_per_atom_A3': float(np.mean(np.abs(np.linalg.det(data.cell))) / n),
        'cell_lengths_A': [float(x) for x in lengths.mean(axis=0)],
        'cell_lengths_initial_A': [float(x) for x in lengths0],
        'vdos': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vdos.items()},
        'flags': [],
    }

    flags = result['flags']
    if data.duration_ps < MIN_PRODUCTION_PS:
        flags.append(f'production run of only {data.duration_ps:.1f} ps (< {MIN_PRODUCTION_PS} ps)')
    if abs(drift) * data.duration_ps / n > EPOT_DRIFT_TOL:
        flags.append(f'potential energy drifts {1e3 * abs(drift) * data.duration_ps / n:.2f} meV/atom over the '
                     f'production run (not equilibrated: U0 not converged)')
    if kinetic_T is not None and abs(kinetic_T - T) / T > 0.02:
        flags.append(f'kinetic temperature {kinetic_T:.0f} K differs from the target {T:.0f} K (U0 uses the target)')
    if vdos['alias_weight_top10pct'] > 1e-3:
        flags.append(f"{100 * vdos['alias_weight_top10pct']:.2f}% of the DOS weight is near the Nyquist frequency "
                     f"({vdos['nyquist_meV']:.0f} meV): store frames closer in time")
    if plan.get('kind') == 'slab':
        vacuum = _slab_vacuum_A(data)
        result['vacuum_min_A'], result['vacuum_mean_A'] = float(vacuum.min()), float(vacuum.mean())
        c_change = result['cell_lengths_A'][2] / lengths0[2] - 1.0
        result['c_relative_change'] = float(c_change)
        if vacuum.min() < VACUUM_MIN_A:
            flags.append(f'the vacuum of the slab got as thin as {vacuum.min():.1f} Å (< {VACUUM_MIN_A} Å): the slab '
                         f'interacts with its periodic images')
        if abs(c_change) > C_CHANGE_TOL:
            flags.append(f'the length of c changed by {100 * c_change:+.0f}% during the run (free cell)')
    if plan.get('adsorbate'):
        flags += _adsorbate_flags(data, plan['adsorbate'])

    with open(cache, 'w') as json_file:
        json.dump(result, json_file)
    return result


# ---------------------------------------------------------------------------------------------------------------
# free energy at an operating temperature
# ---------------------------------------------------------------------------------------------------------------

def thermal_terms(
        folder,
        operating_T,
        equil_ps=None,
        vdos_max_lag_ps=VDOS_MAX_LAG_PS,
        n_cell_dof=0
):
    """Free (Gibbs) energy per atom G(T_op) of the structure in `folder`, as a sum of terms, from its finished MD runs.

    G(T_op) = U0 + F_vib(T_op). The terms are collected in a dictionary and G is their sum, so a further contribution
    (a P V term, an entropy, ...) is one more entry and nothing else changes; none is included today.

    The MD run whose temperature is closest to `operating_T` (the lower one in a tie) provides U0, the area and the
    vibrational spectrum. Only F_vib is evaluated at `operating_T`, with that spectrum: the phonons (and U0, A) are assumed
    not to change between the temperature of the run and `operating_T`. A flag is raised when the two differ by more than
    ``T_RUN_TOL_K``. At an MD temperature nothing is assumed. E0 does not enter (see the module notes).

    Args:
        folder (str): Slab or bulk folder with ``md/<T>K`` runs.
        operating_T (float): Temperature of the evaluation (K); need not be an MD temperature.

    Returns:
        dict: 'G_eV_per_atom' (sum of the terms), one entry per term ('U0', 'F_vib'; also all together in 'terms'), 'area_A2'
            (mean area of one cell of the run), 'n_atoms_cell', 'T_run_K' (temperature of the run used), 'md_temperatures_K'
            (finished runs), 'fix_cell' (how the cell was treated in the run), 'flags'.
    """
    done = [temperature for temperature in md_temperatures(folder) if md_state(md_run_dir(folder, temperature)) == 'done']
    if not done:
        raise MissingData(f'no finished MD run in {folder}/md')
    t_run = min(done, key=lambda temperature: abs(temperature - operating_T))      # ascending: a tie gives the lower T
    run = analyse_md_run(md_run_dir(folder, t_run), equil_ps, vdos_max_lag_ps, n_cell_dof)

    flags = [f"{t_run:g} K: {x}" for x in run['flags']]
    if abs(t_run - operating_T) > T_RUN_TOL_K:
        flags.append(f'T = {operating_T:g} K: U0, the area and the vibrational spectrum are those of the MD run at {t_run:g} K '
                     f'(assumed unchanged; only F_vib is evaluated at T)')

    terms = {'U0': run['u0_eV_per_atom'],
             'F_vib': f_vib(run['vdos']['omega_meV'], run['vdos']['weight'], operating_T)}
    return {'G_eV_per_atom': float(sum(terms.values())), **terms, 'terms': dict(terms), 'area_A2': run['area_A2'],
            'n_atoms_cell': run['n_atoms_cell'], 'T_run_K': t_run, 'md_temperatures_K': done, 'fix_cell': run['fix_cell'],
            'flags': flags}


def check_md_consistency(folders, model_load_path):
    """Warnings if the energies entering gamma(T) come from different codes/models (they must be one level of theory)."""
    import libraries.energy_lookup as energy_lookup
    messages, codes = [], {}
    for folder in folders:
        for temperature in md_temperatures(folder):
            md_dir = md_run_dir(folder, temperature)
            if md_state(md_dir) != 'done':
                continue
            plan = read_md_json(md_dir)
            code = 'VASP' if _is_vasp_run(md_dir) else 'MACE'
            codes.setdefault(code, []).append(os.path.basename(folder.rstrip('/')))
            if code == 'MACE' and plan.get('model') not in (None, model_load_path):
                messages.append(f"{md_dir} was run with '{plan.get('model')}' but the 0 K energies use '{model_load_path}'")
        source = energy_lookup.get_energy_source(folder)
        for temperature in md_temperatures(folder)[:1]:
            code = 'VASP' if _is_vasp_run(md_run_dir(folder, temperature)) else 'MACE'
            if (code == 'VASP') != (source == 'DFT'):
                messages.append(f"{os.path.basename(folder.rstrip('/'))}: the MD is {code} but its 0 K energy comes "
                                f"from {source}; the 0 K column is then at another level of theory than G(T)")
    if len(codes) > 1:
        messages.append(f'MD runs come from different codes {sorted(codes)}: the free-energy differences are not meaningful')
    for message in messages:
        print(f'[warning] {message}')
    return messages


def surface_free_energy_ranking(
        slab_folder,
        operating_T,
        model_load_path=None,
        device=None,
        equil_ps=None,
        vdos_max_lag_ps=VDOS_MAX_LAG_PS,
        n_cell_dof=0,
        only_stoichiometric=False
):
    """gamma(T_op) of every slab of a ``slab_v<i>`` folder next to the 0 K surface energy.

    gamma(T) = [G_slab(T) - N g_bulk(T)] / (2 A(T)) with G = U0 + F_vib per atom (see ``thermal_terms`` and the module
    notes) and the area from the NPT run used; ``surfaces.get_surface_energy_of_formation`` does the arithmetic. A slab
    without a finished MD run gets NaN and a flag instead of stopping the ranking; the bulk needs one. The 0 K column is
    the static level, G = E0, through the same function, so ``delta_J_m2`` is the difference between the two levels. The
    columns ``excess_U0_eV`` and ``excess_Fvib_eV`` are the contributions of the two terms of G to the numerator of
    gamma(T), slab minus bulk (2 A gamma(T) = excess_U0 + excess_Fvib, before the conversion to J/m^2).

    Returns:
        pandas.DataFrame indexed by slab folder, sorted by gamma(T): energies in J/m^2, areas, ranks at 0 K and at T,
        the flags of every slab and the slab flags of the 0 K ranking. ``df.attrs['kendall_tau']`` compares the two rankings.
    """
    import pandas as pd
    import libraries.energy_lookup as energy_lookup
    import libraries.surfaces as surfaces
    from libraries.mlip import DEFAULT_DEVICE, DEFAULT_MODEL
    model_load_path = DEFAULT_MODEL if model_load_path is None else model_load_path
    device = DEFAULT_DEVICE if device is None else device

    bulk_folder, names = structure_folders(slab_folder)
    kwargs = dict(equil_ps=equil_ps, vdos_max_lag_ps=vdos_max_lag_ps, n_cell_dof=n_cell_dof)

    # 0 K reference energies, read exactly as in the 0 K ranking
    bulk_data = surfaces.load_json(os.path.join(bulk_folder, 'slab_data.json'))
    n_bulk = bulk_data['number_of_sites']
    e0_bulk = energy_lookup.read_energy(bulk_folder, model_load_path=model_load_path, device=device)
    bulk = thermal_terms(bulk_folder, operating_T, **kwargs)        # MissingData if the bulk has no run
    check_md_consistency([bulk_folder] + [os.path.join(slab_folder, n) for n in names], model_load_path)

    rows = {}
    for name in names:
        folder = os.path.join(slab_folder, name)
        slab_data = surfaces.load_json(os.path.join(folder, 'slab_data.json'))
        n, area0 = slab_data['number_of_sites'], slab_data['surface_area']
        e0 = energy_lookup.read_energy(folder, model_load_path=model_load_path, device=device)
        gamma_0 = surfaces.get_surface_energy_of_formation(e0, e0_bulk * n / n_bulk, area0)
        row = {'surface_energy_0K_J_m2': gamma_0, 'surface_energy_T_J_m2': np.nan, 'area_0K_A2': area0,
               'area_T_A2': np.nan, 'excess_U0_eV': np.nan, 'excess_Fvib_eV': np.nan, 'flags': '',
               'miller_index': str(slab_data['miller_index']), 'shift': slab_data['shift'], 'number_of_sites': n,
               'symmetric': slab_data.get('is_symmetric', 'unknown'), 'polar': slab_data.get('is_polar', 'unknown'),
               'stoichiometric': slab_data.get('is_stoichiometric', 'unknown')}
        try:
            terms = thermal_terms(folder, operating_T, **kwargs)
            row['surface_energy_T_J_m2'] = surfaces.get_surface_energy_of_formation(
                n * terms['G_eV_per_atom'], n * bulk['G_eV_per_atom'], terms['area_A2'])
            row['area_T_A2'] = terms['area_A2']
            # contribution of each term of G to the numerator of gamma(T): 2 A gamma = excess_U0 + excess_Fvib (eV)
            row['excess_U0_eV'] = n * (terms['U0'] - bulk['U0'])
            row['excess_Fvib_eV'] = n * (terms['F_vib'] - bulk['F_vib'])
            flags = list(terms['flags'])
            if terms['T_run_K'] != bulk['T_run_K']:
                flags.append(f"the slab uses the MD run at {terms['T_run_K']:g} K, the bulk the one at {bulk['T_run_K']:g} K")
            row['flags'] = '; '.join(flags)
        except MissingData as exc:
            row['flags'] = f'no finite-temperature result: {exc}'
        rows[name] = row

    df = pd.DataFrame(rows).T
    for col in ('surface_energy_0K_J_m2', 'surface_energy_T_J_m2', 'area_0K_A2', 'area_T_A2', 'excess_U0_eV', 'excess_Fvib_eV'):
        df[col] = df[col].astype(float)
    df['delta_J_m2'] = df['surface_energy_T_J_m2'] - df['surface_energy_0K_J_m2']

    valid = df.dropna(subset=['surface_energy_T_J_m2'])
    if only_stoichiometric:
        valid = valid[valid['stoichiometric'] != 'False']
    df['rank_0K'] = np.nan
    df['rank_T'] = np.nan
    df.loc[valid.index, 'rank_0K'] = valid['surface_energy_0K_J_m2'].rank(method='min')
    df.loc[valid.index, 'rank_T'] = valid['surface_energy_T_J_m2'].rank(method='min')
    df = df.sort_values('surface_energy_T_J_m2')

    tau = np.nan
    if len(valid) >= 2:
        from scipy.stats import kendalltau
        tau = float(kendalltau(valid['surface_energy_0K_J_m2'], valid['surface_energy_T_J_m2'])[0])
    df.attrs['kendall_tau'] = tau
    df.attrs['operating_T'] = operating_T
    return df
