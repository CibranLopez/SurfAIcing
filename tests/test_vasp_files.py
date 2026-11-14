import numpy as np
import pytest

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints

import libraries.vasp_files as vasp_files
from tests.helpers import write_poscar


def cubic_structure(a=4.0, c=None, species=('Cu', 'Cu')):
    lattice = Lattice.tetragonal(a, c or a)
    return Structure(lattice, list(species), [[0, 0, 0], [0.5, 0.5, 0.5]])


class FakeVasprun:
    """Stand-in for pymatgen's Vasprun, so no real vasprun.xml is needed."""
    structure = None

    def __init__(self, path):
        type(self).opened = path
        self.final_structure = type(self).structure


class BrokenVasprun:
    def __init__(self, path):
        raise ValueError('truncated xml')


class TestGetStructureFile:
    def test_contcar_has_priority_over_poscar(self, tmp_path):
        write_poscar(cubic_structure(), tmp_path, 'POSCAR')
        write_poscar(cubic_structure(), tmp_path, 'CONTCAR')
        assert vasp_files.get_structure_file(str(tmp_path)) == str(tmp_path / 'CONTCAR')

    def test_poscar_when_there_is_no_contcar(self, tmp_path):
        write_poscar(cubic_structure(), tmp_path, 'POSCAR')
        assert vasp_files.get_structure_file(str(tmp_path)) == str(tmp_path / 'POSCAR')

    def test_none_when_there_is_no_structure_file(self, tmp_path):
        assert vasp_files.get_structure_file(str(tmp_path)) is None


class TestReadStructure:
    def test_reads_poscar_without_vasprun(self, tmp_path):
        structure = cubic_structure()
        write_poscar(structure, tmp_path)
        result = vasp_files.read_structure(str(tmp_path))
        assert result.composition == structure.composition
        np.testing.assert_allclose(result.lattice.matrix, structure.lattice.matrix, atol=1e-8)

    def test_contcar_is_preferred_to_poscar(self, tmp_path):
        write_poscar(cubic_structure(a=4.0), tmp_path, 'POSCAR')
        write_poscar(cubic_structure(a=4.2), tmp_path, 'CONTCAR')
        assert vasp_files.read_structure(str(tmp_path)).lattice.a == pytest.approx(4.2)

    def test_vasprun_is_preferred_to_contcar(self, tmp_path, monkeypatch):
        write_poscar(cubic_structure(a=4.0), tmp_path, 'CONTCAR')
        (tmp_path / 'vasprun.xml').write_text('')
        FakeVasprun.structure = cubic_structure(a=5.0)
        monkeypatch.setattr(vasp_files, 'Vasprun', FakeVasprun)
        assert vasp_files.read_structure(str(tmp_path)).lattice.a == pytest.approx(5.0)

    def test_compressed_vasprun_is_found(self, tmp_path, monkeypatch):
        (tmp_path / 'vasprun.xml.gz').write_text('')
        FakeVasprun.structure = cubic_structure(a=5.5)
        monkeypatch.setattr(vasp_files, 'Vasprun', FakeVasprun)
        assert vasp_files.read_structure(str(tmp_path)).lattice.a == pytest.approx(5.5)
        assert FakeVasprun.opened.endswith('vasprun.xml.gz')

    def test_unreadable_vasprun_falls_back_to_the_structure_file(self, tmp_path, monkeypatch, capsys):
        write_poscar(cubic_structure(a=4.0), tmp_path, 'POSCAR')
        (tmp_path / 'vasprun.xml').write_text('<broken')
        monkeypatch.setattr(vasp_files, 'Vasprun', BrokenVasprun)
        assert vasp_files.read_structure(str(tmp_path)).lattice.a == pytest.approx(4.0)
        assert 'Error reading vasprun.xml' in capsys.readouterr().out

    def test_none_for_an_empty_folder(self, tmp_path):
        assert vasp_files.read_structure(str(tmp_path)) is None


class TestVolumeAndLattice:
    def test_read_volume(self, tmp_path):
        structure = cubic_structure(a=4.0, c=10.0)
        write_poscar(structure, tmp_path)
        assert vasp_files.read_volume(str(tmp_path)) == pytest.approx(160.0)

    def test_read_lattice_vectors_are_rows(self, tmp_path):
        structure = Structure(Lattice([[4, 0, 0], [1, 5, 0], [0, 0, 9]]), ['Cu'], [[0, 0, 0]])
        write_poscar(structure, tmp_path)
        np.testing.assert_allclose(vasp_files.read_lattice_vectors(str(tmp_path)), [[4, 0, 0], [1, 5, 0], [0, 0, 9]])

    def test_missing_data_returns_none_and_explains(self, tmp_path, capsys):
        assert vasp_files.read_volume(str(tmp_path)) is None
        assert vasp_files.read_lattice_vectors(str(tmp_path)) is None
        out = capsys.readouterr().out
        assert 'Could not determine structure volume' in out
        assert 'Could not determine lattice vectors' in out


class TestGenerateKpoints:
    def test_density_scales_with_reciprocal_length(self, tmp_path):
        poscar = write_poscar(cubic_structure(a=4.0, c=20.0), tmp_path)
        kpoints_file = str(tmp_path / 'KPOINTS')
        vasp_files.generate_kpoints(poscar, kpoints_file, kpoints_density=[20, 20, 20])
        # 20 / a  ->  5 points for a = 4 A and 1 point for c = 20 A
        assert Kpoints.from_file(kpoints_file).kpts[0] == (5, 5, 1)

    def test_density_is_set_per_direction(self, tmp_path):
        poscar = write_poscar(cubic_structure(a=4.0, c=4.0), tmp_path)
        kpoints_file = str(tmp_path / 'KPOINTS')
        vasp_files.generate_kpoints(poscar, kpoints_file, kpoints_density=(8, 12, 20))
        assert Kpoints.from_file(kpoints_file).kpts[0] == (2, 3, 5)

    def test_default_arguments_write_a_kpoints_file_in_the_working_directory(self, tmp_path, monkeypatch):
        write_poscar(cubic_structure(a=4.0), tmp_path)
        monkeypatch.chdir(tmp_path)
        vasp_files.generate_kpoints()
        assert (tmp_path / 'KPOINTS').exists()

    def test_missing_poscar_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vasp_files.generate_kpoints(str(tmp_path / 'POSCAR'), str(tmp_path / 'KPOINTS'))
