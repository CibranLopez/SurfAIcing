"""Finite-temperature surface energies: NPT folders, MD analysis and the surface free energy gamma(T).

Surface energies at 0 K compare total energies. At finite temperature the quantity to compare is the Helmholtz free
energy, which is obtained here from NPT molecular dynamics with the effective-harmonic scheme (Hellman et al., PRB 87,
104111 (2013)), the same one used for the anharmonic level of HeatUp:

    F(T) = U0(T) + F_vib[omega_eff(T)]                                          (per atom, P = 0 so no PV term)

    U0(T)   = <U_MD> / N - (3N - 3) k T / (2 N)        mean MD potential energy minus the classical harmonic part
    F_vib   = int g(omega) [ omega/2 + k T ln(1 - exp(-omega/kT)) ] domega       quantum oscillators, zero-point included

``g(omega)`` is the vibrational density of states of the MD run: the Fourier transform of the mass-weighted velocity
autocorrelation, normalised to 3 modes per atom. It contains the anharmonic renormalisation of the frequencies at the
temperature and cell of the run ("effective" harmonic spectrum).

The surface free energy is then (same expression as at 0 K, with free energies and the area at temperature T)

    gamma(T) = [ F_slab(T) - N f_bulk(T) ] / (2 A(T)),        f_bulk per atom from an NPT run of the bulk.

Operating temperature. The ranking can be evaluated at any temperature T_op, not only at the MD temperatures:
    * U0 - E0 (and the area) are interpolated in T through the anchor (0 K, 0), PCHIP with two or more MD temperatures,
      linear with one; above the highest MD temperature they are extrapolated linearly and flagged;
    * F_vib(T_op) is evaluated with the spectra of the MD runs and interpolated linearly in T between the two
      neighbouring MD temperatures (constant outside the range, flagged).
    E0 is the 0 K energy of the relaxed structure, read exactly as in the 0 K ranking (``energy_lookup.read_energy``).

Folder convention (next to POSCAR/CONTCAR of every slab and of the bulk):

    <folder>/md/<T>K/POSCAR        supercell of the structure, NOT relaxed (the NPT run takes care of it)
    <folder>/md/<T>K/md.json       what the run is (plan) and, once finished, how it was run
    <folder>/md/<T>K/output.traj   ML-IAP run (written by ``mlip.npt_dynamics``), or
    <folder>/md/<T>K/vasprun.xml   VASP run (ISIF = 3 NPT; every step stored)
    <folder>/md/<T>K/md_analysis.json   cached analysis (U0, area, spectrum, flags)

The cell is fully flexible during the run: nothing is frozen. The analysis only reports (flags) what happened to the
cell, e.g. a change of the length of c or a vacuum thinner than the range of the ML-IAP.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

import numpy as np

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
EQUIL_PS         = 2.0               # ps discarded at the start of every run
VDOS_MAX_LAG_PS  = 4.0               # ps of the velocity autocorrelation used for the spectrum
MIN_PRODUCTION_PS = 5.0              # shorter production runs are flagged
EPOT_DRIFT_TOL   = 5e-4              # eV/atom, drift of the potential energy over the production run that is flagged
VACUUM_MIN_A     = 6.0               # a slab whose vacuum gets thinner than this is flagged (range of the MACE models)
C_CHANGE_TOL     = 0.10              # relative change of the length of c of a slab that is flagged


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
        max_atoms=1000
):
    """Create ``<source_folder>/md/<T>K`` with the supercell POSCAR and the plan of one NPT run per temperature.

    The structure is the (unrelaxed) POSCAR of `source_folder` (CONTCAR if there is no POSCAR). It is repeated until
    the lattice vectors are at least `min_length` long: a and b only for a slab, all three for the bulk. A POTCAR
    present in `source_folder` is copied (the species order does not change).

    Args:
        source_folder (str): Folder of a slab or of the bulk.
        kind (str): 'slab' or 'bulk'.
        temperatures (sequence): MD temperatures in K.
        min_length (float): Minimum lattice vector length in Angstrom.
        n_steps, timestep, ttime, ptime, barostat_modulus, loginterval, pressure: see ``mlip.npt_dynamics``.
        overwrite (bool): Existing md folders are kept (reused) unless overwrite=True, which regenerates them.
        max_atoms (int): A warning is printed if the supercell has more atoms.

    Returns:
        list[str]: the md folders.
    """
    from ase.io.vasp import write_vasp
    from pymatgen.io.ase import AseAtomsAdaptor

    if kind not in ('slab', 'bulk'):
        raise ValueError(f"kind must be 'slab' or 'bulk', got {kind!r}")
    source = None
    for candidate in ('POSCAR', 'CONTCAR'):
        if os.path.exists(os.path.join(source_folder, candidate)):
            source = os.path.join(source_folder, candidate)
            break
    if source is None:
        raise FileNotFoundError(f'Neither POSCAR nor CONTCAR found in {source_folder}')

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
    import pandas as pd
    bulk_folder, names = structure_folders(slab_folder)
    rows = []
    for name, folder in [('bulk', bulk_folder)] + [(n, os.path.join(slab_folder, n)) for n in names]:
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
                                output_folder=md_dir)
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
    todo = []
    for folder in [bulk_folder] + [os.path.join(slab_folder, n) for n in names]:
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


def read_md_run(md_dir, equil_ps=EQUIL_PS):
    """Read the production part (first `equil_ps` ps discarded) of the run in `md_dir`, MACE or VASP."""
    plan = read_md_json(md_dir)
    if plan is None:
        raise MissingData(f'{md_dir} has no md.json')
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


def f_vib(omega_meV, weight, temperature):
    """Quantum harmonic vibrational free energy per atom (eV) of a spectrum (modes with omega <= OMEGA_MIN_MEV skipped)."""
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


def analyse_md_run(
        md_dir,
        equil_ps=EQUIL_PS,
        vdos_max_lag_ps=VDOS_MAX_LAG_PS,
        n_cell_dof=0,
        force=False
):
    """Everything the free energy needs from one finished run; cached in ``md_analysis.json``.

    ``n_cell_dof``: number of cell degrees of freedom whose kT/2 each is subtracted from <U> in U0 on top of the
    3N - 3 atomic ones. The default 0 keeps the HeatUp prescription (see the discussion in the module notes).
    """
    plan = read_md_json(md_dir)
    source = _vasprun_path(md_dir) or os.path.join(md_dir, 'output.traj')
    stamp = {'equil_ps': equil_ps, 'vdos_max_lag_ps': vdos_max_lag_ps, 'n_cell_dof': n_cell_dof,
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
        'stamp': stamp, 'kind': plan.get('kind'), 'code': data.code, 'momenta_from': data.momenta_from,
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

    with open(cache, 'w') as json_file:
        json.dump(result, json_file)
    return result


# ---------------------------------------------------------------------------------------------------------------
# free energy at an operating temperature
# ---------------------------------------------------------------------------------------------------------------

def _through_origin(temperature, md_temperatures_K, offsets, flags, what):
    """Value at `temperature` of a quantity known at the MD temperatures as an offset from its 0 K value.

    Interpolated through the anchor (0 K, 0): linear with one MD temperature, PCHIP with more; linear
    extrapolation above the highest one (flagged).
    """
    from scipy.interpolate import PchipInterpolator
    tt = np.r_[0.0, np.asarray(md_temperatures_K, float)]
    cc = np.r_[0.0, np.asarray(offsets, float)]
    if temperature <= tt[-1]:
        if len(tt) == 2:
            return float(np.interp(temperature, tt, cc))
        return float(PchipInterpolator(tt, cc)(temperature))
    slope = (cc[-1] - cc[-2]) / (tt[-1] - tt[-2])
    flags.append(f'{what} extrapolated linearly from {tt[-1]:g} K to {temperature:g} K (no MD run at or above T)')
    return float(cc[-1] + slope * (temperature - tt[-1]))


def thermal_terms(
        folder,
        operating_T,
        e0_per_atom,
        area0=None,
        equil_ps=EQUIL_PS,
        vdos_max_lag_ps=VDOS_MAX_LAG_PS,
        n_cell_dof=0
):
    """F(T_op) = E0 + [U0(T_op) - E0] + F_vib(T_op) per atom for the structure in `folder`, from its finished MD runs.

    Args:
        folder (str): Slab or bulk folder with ``md/<T>K`` runs.
        operating_T (float): Temperature of the evaluation (K); need not be an MD temperature.
        e0_per_atom (float): 0 K energy per atom of the relaxed structure, in eV (same level of theory as the MD).
        area0 (float): 0 K area of the slab cell (A^2); if given, the area at T_op is returned too.

    Returns:
        dict: 'F_eV_per_atom', 'E0', 'dU0', 'F_vib', 'area_A2', 'n_atoms_cell', 'md_temperatures_K', 'flags'.
    """
    runs = []
    for temperature in md_temperatures(folder):
        md_dir = md_run_dir(folder, temperature)
        if md_state(md_dir) == 'done':
            runs.append(analyse_md_run(md_dir, equil_ps, vdos_max_lag_ps, n_cell_dof))
    if not runs:
        raise MissingData(f'no finished MD run in {folder}/md')
    runs.sort(key=lambda r: r['temperature_K'])
    t_md = [r['temperature_K'] for r in runs]
    flags = [f"{r['temperature_K']:g} K: {x}" for r in runs for x in r['flags']]

    d_u0 = _through_origin(operating_T, t_md, [r['u0_eV_per_atom'] - e0_per_atom for r in runs], flags, 'U0')
    curves = [f_vib(r['vdos']['omega_meV'], r['vdos']['weight'], operating_T) for r in runs]
    fv = float(np.interp(operating_T, t_md, curves))
    if operating_T < t_md[0] or operating_T > t_md[-1]:
        flags.append(f'T = {operating_T:g} K is outside the MD range [{t_md[0]:g}, {t_md[-1]:g}] K: the vibrational '
                     f'spectrum is that of the nearest run')

    out = {'E0': e0_per_atom, 'dU0': d_u0, 'F_vib': fv, 'F_eV_per_atom': e0_per_atom + d_u0 + fv,
           'n_atoms_cell': runs[0]['n_atoms_cell'], 'md_temperatures_K': t_md, 'flags': flags, 'area_A2': None}
    if area0 is not None:
        out['area_A2'] = area0 + _through_origin(operating_T, t_md, [r['area_A2'] - area0 for r in runs], [], 'area')
    return out


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
                                f"from {source}; the anchor U0 - E0 mixes levels of theory")
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
        equil_ps=EQUIL_PS,
        vdos_max_lag_ps=VDOS_MAX_LAG_PS,
        n_cell_dof=0,
        only_stoichiometric=False
):
    """gamma(T_op) of every slab of a ``slab_v<i>`` folder next to the 0 K surface energy.

    gamma(T) = [F_slab(T) - N f_bulk(T)] / (2 A(T)) with F = E0 + (U0 - E0) + F_vib per atom (see the module notes) and
    the area A(T) from the NPT runs. A slab without a finished MD run gets NaN and a flag instead of stopping the
    ranking; the bulk needs one. The 0 K column is computed from the same E0, so the difference between the two columns
    is purely thermal.

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
    bulk = thermal_terms(bulk_folder, operating_T, e0_bulk / n_bulk, **kwargs)        # MissingData if the bulk has no run
    check_md_consistency([bulk_folder] + [os.path.join(slab_folder, n) for n in names], model_load_path)

    rows = {}
    for name in names:
        folder = os.path.join(slab_folder, name)
        slab_data = surfaces.load_json(os.path.join(folder, 'slab_data.json'))
        n, area0 = slab_data['number_of_sites'], slab_data['surface_area']
        e0 = energy_lookup.read_energy(folder, model_load_path=model_load_path, device=device)
        gamma_0 = surfaces.get_surface_energy_of_formation(e0, e0_bulk * n / n_bulk, area0)
        row = {'surface_energy_0K_J_m2': gamma_0, 'surface_energy_T_J_m2': np.nan, 'area_0K_A2': area0,
               'area_T_A2': np.nan, 'dU0_eV': np.nan, 'dFvib_eV': np.nan, 'flags': '',
               'miller_index': str(slab_data['miller_index']), 'shift': slab_data['shift'], 'number_of_sites': n,
               'symmetric': slab_data.get('is_symmetric', 'unknown'), 'polar': slab_data.get('is_polar', 'unknown'),
               'stoichiometric': slab_data.get('is_stoichiometric', 'unknown')}
        try:
            terms = thermal_terms(folder, operating_T, e0 / n, area0=area0, **kwargs)
            row['surface_energy_T_J_m2'] = surfaces.get_surface_energy_of_formation(
                n * terms['F_eV_per_atom'], n * bulk['F_eV_per_atom'], terms['area_A2'])
            row['area_T_A2'] = terms['area_A2']
            row['dU0_eV'] = n * (terms['dU0'] - bulk['dU0'])
            row['dFvib_eV'] = n * (terms['F_vib'] - bulk['F_vib'])
            flags = list(terms['flags'])
            if terms['md_temperatures_K'] != bulk['md_temperatures_K']:
                flags.append(f"slab MD temperatures {terms['md_temperatures_K']} differ from the bulk ones "
                             f"{bulk['md_temperatures_K']}")
            row['flags'] = '; '.join(flags)
        except MissingData as exc:
            row['flags'] = f'no finite-temperature result: {exc}'
        rows[name] = row

    df = pd.DataFrame(rows).T
    for col in ('surface_energy_0K_J_m2', 'surface_energy_T_J_m2', 'area_0K_A2', 'area_T_A2', 'dU0_eV', 'dFvib_eV'):
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
