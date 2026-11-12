import os
from pathlib import Path

import numpy as np

from mace.calculators import mace_mp, MACECalculator

from ase import units
from ase.io.vasp import read_vasp, write_vasp
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch
from ase.constraints import FixAtoms

try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/mace/mace-mpa-0-medium.model")
#DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/mace/MACE_MPtrj_20229model")


def _make_mace_calculator(model_load_path, device="cuda", dispersion=False, default_dtype="float64"):
    """
    model_load_path can be:
      - local file path to a .model file (e.g. ~/.cache/mace/mace-mpa-0-medium.model)
      - foundation model name (e.g. "large", "medium-mpa-0") -> uses mace_mp
    """
    p = Path(str(model_load_path)).expanduser()

    # Local file case
    if p.is_file() or str(model_load_path).endswith(".model"):
        if not p.is_file():
            raise FileNotFoundError(
                f"[MACE] Model file not found: {p}\n"
                f"Put your file here: {DEFAULT_MODEL_PATH}\n"
                f"Or pass the correct absolute path as model_load_path."
            )

        if os.environ.get("MACE_VERBOSE", "0") == "1":
            print(f"[MACE] USING LOCAL MODEL FILE: {p}")

        # MACECalculator API differs across versions; try both arg names
        try:
            return MACECalculator(
                model_paths=str(p),
                device=device,
                dispersion=dispersion,
                default_dtype=default_dtype,
            )
        except TypeError:
            return MACECalculator(
                model_path=str(p),
                device=device,
                dispersion=dispersion,
                default_dtype=default_dtype,
            )

    # Foundation model name case
    if os.environ.get("MACE_VERBOSE", "0") == "1":
        print(f"[MACE] USING FOUNDATION MODEL NAME: {model_load_path}")

    return mace_mp(
        model=model_load_path,
        device=device,
        dispersion=dispersion,
        default_dtype=default_dtype,
    )


def _min_distance(atoms) -> float:
    """Minimum pair distance (Å)."""
    d = atoms.get_all_distances(mic=True)
    d[d == 0.0] = np.inf
    return float(d.min())


def _freeze_bottom_by_fraction(atoms, frac: float):
    """
    Freeze atoms with z <= quantile(z, frac).
    frac=0.35 means freeze bottom 35% in z.
    """
    z = atoms.positions[:, 2]
    zcut = float(np.quantile(z, frac))
    mask = z <= zcut
    atoms.set_constraint(FixAtoms(mask=mask))
    return zcut, int(mask.sum())


def _report_fmax_atom(atoms, tag=""):
    F = atoms.get_forces()
    fn = np.linalg.norm(F, axis=1)
    i = int(np.argmax(fn))
    zi = atoms.positions[i, 2]
    print(f"[{tag}] fmax={fn[i]:.6f} eV/Å  atom={i} {atoms[i].symbol}  z={zi:.3f} Å")


def structural_relaxation(
        path_to_structure,
        model_load_path=DEFAULT_MODEL_PATH,
        device="cuda",
        dispersion=False,
        relax_cell=False,
        fmax=0.04,
        output_folder="./",
        steps=3000,
        # --- slab-specific robustness knobs ---
        freeze_bottom_frac=0.00,      # set None or 0 to disable
        check_min_distance=True,
        min_distance_warn=0.80,       # Å
        # Stage 1 (robust untangling)
        do_stage1_fire=True,
        stage1_fmax=0.05,
        stage1_steps=300,
        # Stage 2 (fast finishing)
        stage2_optimizer="lbfgs",      # "lbfgs" | "bfgs_linesearch" | "bfgs"
        stage2_maxstep=0.01,
        # Optional MD kick (escape a bad basin)
        md_kick=False,
        md_temperature=500,           # K
        md_steps=300,
        md_timestep_fs=1.0,
        md_friction=0.01,             # 1/fs
        # Logging
        log_interval=25,
):
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    atoms = read_vasp(file=path_to_structure)

    atoms.calc = _make_mace_calculator(
        model_load_path=model_load_path,
        device=device,
        dispersion=dispersion,
        default_dtype="float64",
    )

    if check_min_distance:
        mind = _min_distance(atoms)
        if mind < min_distance_warn:
            print(f"[WARN] min distance = {mind:.3f} Å (< {min_distance_warn:.2f} Å). Likely overlap / bad termination.")

    # Freeze bottom of slab (recommended for slabs; disable for bulk)
    if freeze_bottom_frac is not None and float(freeze_bottom_frac) > 0.0:
        zcut, nfix = _freeze_bottom_by_fraction(atoms, float(freeze_bottom_frac))
        print(f"[constraint] froze {nfix} atoms with z <= {zcut:.3f} Å")

    # Optional MD kick
    if md_kick:
        MaxwellBoltzmannDistribution(atoms, md_temperature * units.kB)
        dyn_md = Langevin(
            atoms,
            timestep=md_timestep_fs * units.fs,
            temperature=md_temperature * units.kB,
            friction=md_friction / units.fs,
            trajectory=str(out / "md.traj"),
            logfile=str(out / "md.log"),
        )
        dyn_md.run(md_steps)
        _report_fmax_atom(atoms, tag="after_md")

    # Cell relaxation wrapper (not recommended for slabs with vacuum unless you know why)
    atoms_opt = ExpCellFilter(atoms) if relax_cell else atoms

    # Stage 1: FIRE (robust)
    if do_stage1_fire:
        opt1 = FIRE(atoms_opt, trajectory=str(out / "stage1.traj"), logfile=str(out / "stage1.log"))
        opt1.attach(lambda: _report_fmax_atom(atoms, tag="FIRE"), interval=log_interval)
        opt1.run(fmax=stage1_fmax, steps=stage1_steps)

    # Stage 2: finishing optimizer
    stage2 = str(stage2_optimizer).lower().strip()
    if stage2 == "lbfgs":
        opt2 = LBFGS(
            atoms_opt,
            trajectory=str(out / "run.traj"),
            logfile=str(out / "opt.log"),
            maxstep=stage2_maxstep,
        )
    elif stage2 in ("bfgs_linesearch", "bfgsline", "bfgslinesearch"):
        opt2 = BFGSLineSearch(
            atoms_opt,
            trajectory=str(out / "run.traj"),
            logfile=str(out / "opt.log"),
            maxstep=stage2_maxstep,
        )
    elif stage2 == "bfgs":
        opt2 = BFGS(
            atoms_opt,
            trajectory=str(out / "run.traj"),
            logfile=str(out / "opt.log"),
            maxstep=stage2_maxstep,
        )
    else:
        raise ValueError(f"Unknown stage2_optimizer='{stage2_optimizer}'. Use 'lbfgs', 'bfgs_linesearch', or 'bfgs'.")

    opt2.attach(lambda: _report_fmax_atom(atoms, tag=stage2.upper()), interval=log_interval)
    opt2.run(fmax=fmax, steps=steps)

    if relax_cell:
        atoms = atoms_opt.atoms  # unwrap ExpCellFilter

    write_vasp(str(out / "CONTCAR"), atoms=atoms, direct=True, sort=True)
    return atoms


def single_shot_energy_calculation(
        path_to_structure,
        model_load_path=DEFAULT_MODEL_PATH,
        device="cuda",
        dispersion=False
):
    atoms = read_vasp(file=path_to_structure)

    atoms.calc = _make_mace_calculator(
        model_load_path=model_load_path,
        device=device,
        dispersion=dispersion,
        default_dtype="float64",
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    return energy, forces, stress


def molecular_dynamics(
        path_to_structure,
        model_load_path=DEFAULT_MODEL_PATH,
        device="cuda",
        dispersion=False,
        temperature=300,
        timestep=1,
        friction=0.001,
        n_steps=200,
        output_folder="./"
):
    atoms = read_vasp(file=path_to_structure)

    atoms.calc = _make_mace_calculator(
        model_load_path=model_load_path,
        device=device,
        dispersion=dispersion,
        default_dtype="float64",
    )

    timestep *= units.fs
    temperature *= units.kB
    friction *= 1 / units.fs

    MaxwellBoltzmannDistribution(atoms, temperature)

    dyn = Langevin(
        atoms,
        timestep=timestep,
        temperature=temperature,
        friction=friction,
        trajectory=f"{output_folder}/run.traj",
        logfile=f"{output_folder}/md.log",
    )
    dyn.run(n_steps)

    write_vasp(f"{output_folder}/CONTCAR", atoms=atoms, direct=True, sort=True)
    return atoms

