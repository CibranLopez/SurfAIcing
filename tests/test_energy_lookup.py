import json
import os

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io.vasp import read_vasp, write_vasp

import libraries.energy_lookup as energy_lookup


class FakeVasprun:
    """Stand-in for pymatgen's Vasprun."""
    energy = -12.5

    def __init__(self, path):
        self.final_energy = type(self).energy


class BrokenVasprun:
    def __init__(self, path):
        raise ValueError('truncated xml')


def cu_atoms(displace=False):
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    if displace:
        atoms.positions[0] += [0.3, -0.2, 0.1]
    return atoms


def single_point(path):
    atoms = read_vasp(str(path))
    atoms.calc = EMT()
    return atoms.get_potential_energy()


@pytest.fixture
def folder_with(tmp_path):
    """Factory: a folder holding the requested files."""
    def make(name='calc', poscar=False, contcar=False, vasprun=False, vasprun_gz=False, single_shot=None,
             displace=False):
        folder = tmp_path / name
        folder.mkdir()
        if poscar:
            write_vasp(str(folder / 'POSCAR'), cu_atoms(displace), direct=True)
        if contcar:
            write_vasp(str(folder / 'CONTCAR'), cu_atoms(displace), direct=True)
        if vasprun:
            (folder / 'vasprun.xml').write_text('')
        if vasprun_gz:
            (folder / 'vasprun.xml.gz').write_text('')
        if single_shot is not None:
            np.savetxt(str(folder / 'single_shot_energy'), [single_shot])
        return folder
    return make


class TestGetEnergySource:
    def test_each_source(self, folder_with):
        assert energy_lookup.get_energy_source(str(folder_with('a', vasprun=True))) == 'DFT'
        assert energy_lookup.get_energy_source(str(folder_with('b', vasprun_gz=True))) == 'DFT'
        assert energy_lookup.get_energy_source(str(folder_with('c', single_shot=-1.0))) == 'precomputed'
        assert energy_lookup.get_energy_source(str(folder_with('d', contcar=True))) == 'ML-IAP'
        assert energy_lookup.get_energy_source(str(folder_with('e', poscar=True))) == 'ML-IAP'

    def test_nothing_available(self, folder_with):
        assert energy_lookup.get_energy_source(str(folder_with('empty'))) is None

    def test_priority_is_dft_then_precomputed_then_mliap(self, folder_with):
        everything = folder_with('all', poscar=True, contcar=True, vasprun=True, single_shot=-1.0)
        assert energy_lookup.get_energy_source(str(everything)) == 'DFT'
        no_dft = folder_with('no_dft', poscar=True, contcar=True, single_shot=-1.0)
        assert energy_lookup.get_energy_source(str(no_dft)) == 'precomputed'


class TestReadEnergy:
    def test_dft_energy_from_vasprun(self, folder_with, monkeypatch):
        monkeypatch.setattr(energy_lookup, 'Vasprun', FakeVasprun)
        folder = folder_with(vasprun=True, contcar=True)
        assert energy_lookup.read_energy(str(folder)) == pytest.approx(-12.5)

    def test_compressed_vasprun(self, folder_with, monkeypatch):
        monkeypatch.setattr(energy_lookup, 'Vasprun', FakeVasprun)
        assert energy_lookup.read_energy(str(folder_with(vasprun_gz=True))) == pytest.approx(-12.5)

    def test_vasprun_has_priority_over_precomputed_and_mliap(self, folder_with, monkeypatch, emt_backend):
        monkeypatch.setattr(energy_lookup, 'Vasprun', FakeVasprun)
        folder = folder_with(vasprun=True, contcar=True, single_shot=-99.0)
        assert energy_lookup.read_energy(str(folder)) == pytest.approx(-12.5)
        assert emt_backend == []                                    # the potential was never loaded

    def test_unreadable_vasprun_gives_nan_and_does_not_silently_switch_method(self, folder_with, monkeypatch, capsys, emt_backend):
        monkeypatch.setattr(energy_lookup, 'Vasprun', BrokenVasprun)
        folder = folder_with(vasprun=True, contcar=True)
        assert np.isnan(energy_lookup.read_energy(str(folder)))
        assert 'Error reading vasprun.xml' in capsys.readouterr().out
        assert emt_backend == []

    def test_precomputed_energy_is_a_float(self, folder_with):
        energy = energy_lookup.read_energy(str(folder_with(single_shot=-3.25)))
        assert isinstance(energy, float)
        assert energy == pytest.approx(-3.25)

    def test_precomputed_has_priority_over_mliap(self, folder_with, emt_backend):
        folder = folder_with(contcar=True, single_shot=-3.25)
        assert energy_lookup.read_energy(str(folder)) == pytest.approx(-3.25)
        assert emt_backend == []

    def test_malformed_precomputed_file_gives_nan(self, folder_with, capsys):
        folder = folder_with()
        (folder / 'single_shot_energy').write_text('not a number\n')
        assert np.isnan(energy_lookup.read_energy(str(folder)))
        assert 'Error reading single_shot_energy' in capsys.readouterr().out

    def test_contcar_is_evaluated_as_it_is(self, folder_with, emt_backend):
        folder = folder_with(contcar=True, displace=True)           # a CONTCAR with forces: it is trusted as relaxed
        assert energy_lookup.read_energy(str(folder)) == pytest.approx(single_point(folder / 'CONTCAR'))
        assert not (folder / 'run.traj').exists()                   # no relaxation was run
        assert not (folder / 'mlip_provenance.json').exists()

    def test_poscar_is_relaxed_at_fixed_cell_before_the_single_point(self, folder_with, emt_backend):
        folder = folder_with(poscar=True, displace=True)
        energy = energy_lookup.read_energy(str(folder))
        assert (folder / 'CONTCAR').exists()
        info = json.loads((folder / 'mlip_provenance.json').read_text())
        assert info['relax_cell'] is False
        assert energy == pytest.approx(single_point(folder / 'CONTCAR'))
        assert energy < single_point(folder / 'POSCAR')             # relaxation lowers the energy
        np.testing.assert_allclose(read_vasp(str(folder / 'CONTCAR')).cell, read_vasp(str(folder / 'POSCAR')).cell)

    def test_model_and_device_are_forwarded(self, folder_with, emt_backend):
        energy_lookup.read_energy(str(folder_with(contcar=True)), model_load_path='m.model', device='cuda')
        assert emt_backend[-1]['model'] == 'm.model' and emt_backend[-1]['device'] == 'cuda'

    def test_empty_folder_gives_nan(self, folder_with):
        assert np.isnan(energy_lookup.read_energy(str(folder_with())))

    def test_structure_file_that_disappears_gives_nan_and_a_message(self, folder_with, monkeypatch, capsys):
        import libraries.vasp_files as vasp_files
        monkeypatch.setattr(vasp_files, 'get_structure_file', lambda folder: None)
        assert np.isnan(energy_lookup.read_energy(str(folder_with(poscar=True))))
        assert 'No structure file found' in capsys.readouterr().out

    def test_failure_to_load_the_potential_gives_nan_and_a_message(self, folder_with, broken_backend, capsys):
        assert np.isnan(energy_lookup.read_energy(str(folder_with(contcar=True))))
        out = capsys.readouterr().out
        assert 'Error loading model or structure' in out and 'RuntimeError' in out


class TestRelaxStructure:
    def test_writes_the_relaxed_structure(self, folder_with, emt_backend):
        folder = folder_with(poscar=True, displace=True)
        energy_lookup.relax_structure(str(folder / 'POSCAR'), relax_cell=False, output_folder=str(folder))
        assert single_point(folder / 'CONTCAR') < single_point(folder / 'POSCAR')

    def test_cell_relaxation_option(self, tmp_path, emt_backend):
        write_vasp(str(tmp_path / 'POSCAR'), bulk('Cu', 'fcc', a=3.9, cubic=True), direct=True)
        energy_lookup.relax_structure(str(tmp_path / 'POSCAR'), relax_cell=True, output_folder=str(tmp_path))
        assert read_vasp(str(tmp_path / 'CONTCAR')).get_volume() < read_vasp(str(tmp_path / 'POSCAR')).get_volume()

    def test_errors_are_raised_not_swallowed(self, folder_with, broken_backend):
        """Regression: the docstring promises that failures are visible, but a bare except used to hide them."""
        folder = folder_with(poscar=True)
        with pytest.raises(RuntimeError, match='cannot load the model'):
            energy_lookup.relax_structure(str(folder / 'POSCAR'), output_folder=str(folder))
        assert not (folder / 'CONTCAR').exists()

    def test_missing_structure_raises(self, tmp_path, emt_backend):
        with pytest.raises(FileNotFoundError):
            energy_lookup.relax_structure(str(tmp_path / 'POSCAR'), output_folder=str(tmp_path))


def test_read_provenance(folder_with):
    folder = folder_with('prov', contcar=True)
    assert energy_lookup._read_provenance(str(folder)) is None
    (folder / 'mlip_provenance.json').write_text(json.dumps({'model': 'a.model'}))
    assert energy_lookup._read_provenance(str(folder)) == {'model': 'a.model'}


class TestCheckEnergySources:
    def test_consistent_sources_are_silent(self, folder_with, capsys):
        folders = [str(folder_with(f'f{i}', contcar=True)) for i in range(3)]
        sources = energy_lookup.check_energy_sources(folders)
        assert sources == {folder: 'ML-IAP' for folder in folders}
        assert capsys.readouterr().out == ''

    def test_mixing_dft_and_mliap_warns_and_lists_the_members(self, folder_with, capsys):
        dft = folder_with('slab_dft', vasprun=True)
        mliap = folder_with('bulk_mliap', contcar=True)
        sources = energy_lookup.check_energy_sources([str(dft), str(mliap)], label='surface-energy ranking')
        out = capsys.readouterr().out
        assert sources == {str(dft): 'DFT', str(mliap): 'ML-IAP'}
        assert '[warning] surface-energy ranking: energies come from different levels of theory' in out
        assert "DFT: ['slab_dft']" in out and "ML-IAP: ['bulk_mliap']" in out

    def test_long_member_lists_are_truncated(self, folder_with, capsys):
        folders = [str(folder_with(f'm{i}', contcar=True)) for i in range(8)] + [str(folder_with('d', vasprun=True))]
        energy_lookup.check_energy_sources(folders)
        assert ' ...' in capsys.readouterr().out

    def test_folders_without_data_do_not_trigger_the_warning(self, folder_with, capsys):
        energy_lookup.check_energy_sources([str(folder_with('a', contcar=True)), str(folder_with('b'))])
        assert capsys.readouterr().out == ''

    def test_relaxation_with_another_model_warns(self, folder_with, capsys):
        folder = folder_with('slab', contcar=True)
        (folder / 'mlip_provenance.json').write_text(json.dumps({'model': 'old.model'}))
        energy_lookup.check_energy_sources([str(folder)], label='adsorption energy', model_load_path='new.model')
        out = capsys.readouterr().out
        assert "was relaxed with 'old.model'" in out and "'new.model'" in out

    def test_relaxation_with_the_same_model_is_silent(self, folder_with, capsys):
        folder = folder_with('slab', contcar=True)
        (folder / 'mlip_provenance.json').write_text(json.dumps({'model': 'same.model'}))
        energy_lookup.check_energy_sources([str(folder)], model_load_path='same.model')
        assert capsys.readouterr().out == ''

    def test_model_check_does_not_apply_to_dft_folders(self, folder_with, capsys):
        folder = folder_with('slab', vasprun=True)
        (folder / 'mlip_provenance.json').write_text(json.dumps({'model': 'old.model'}))
        energy_lookup.check_energy_sources([str(folder)], model_load_path='new.model')
        assert capsys.readouterr().out == ''
