import numpy as np
import pytest

from pymatgen.core import Lattice, Structure

from libraries import alignment


def structure_with_c(fractional_c, c_length=30.0):
    """Cubic-in-plane cell with one atom at each given fractional height along c."""
    coords = [[0.0, 0.0, z] for z in fractional_c]
    return Structure(Lattice.tetragonal(3.0, c_length), ['Cu'] * len(coords), coords)


def step_potential(n, slab_start, slab_end, slab_value=-5.0, vacuum_value=0.0):
    """Potential equal to slab_value inside [slab_start, slab_end] (fractional, periodic) and vacuum_value elsewhere."""
    z = (np.arange(n) + 0.5) / n
    inside = ((z - slab_start) % 1.0) <= ((slab_end - slab_start) % 1.0)
    return np.where(inside, slab_value, vacuum_value)


class TestMidpointMode:
    def test_regions_are_the_two_halves_without_the_edges(self):
        potential = np.arange(100.0)
        shifted, bulk, vacuum = alignment.get_regions(potential, None, 0.25, mode='midpoint')
        np.testing.assert_array_equal(shifted, potential)           # the potential is not moved
        assert bulk == (12, 38)
        assert vacuum == (62, 88)

    def test_odd_number_of_points(self):
        _, bulk, vacuum = alignment.get_regions(np.zeros(101), None, 0.25, mode='midpoint')
        assert bulk == (12, 38)
        assert vacuum == (62, 89)

    def test_no_discard_gives_complete_halves(self):
        _, bulk, vacuum = alignment.get_regions(np.zeros(100), None, 0.0, mode='midpoint')
        assert bulk == (0, 50)
        assert vacuum == (50, 100)

    def test_structure_is_not_used(self):
        # structure=None must be accepted in the legacy mode
        alignment.get_regions(np.zeros(10), None, 0.1, mode='midpoint')

    def test_slab_in_the_second_half_swaps_the_regions(self):
        """Documented limitation of the legacy mode: it assumes the slab sits in the first half of the cell."""
        n = 200
        potential = step_potential(n, 0.55, 0.90)
        _, (b0, b1), (v0, v1) = alignment.get_regions(potential, None, 0.25, mode='midpoint')
        assert np.mean(potential[b0:b1]) == pytest.approx(0.0)      # the "bulk" region is actually vacuum
        assert np.mean(potential[v0:v1]) == pytest.approx(-5.0)


class TestAutoMode:
    def test_slab_in_the_first_half(self):
        n = 200
        structure = structure_with_c([0.1, 0.25, 0.4])
        potential = np.arange(float(n))
        shifted, bulk, vacuum = alignment.get_regions(potential, structure, 0.25, mode='auto')
        # the slab (thickness 0.3 -> 60 points) is moved to the beginning of the array
        assert shifted[0] == 20
        assert bulk == (15, 45)
        assert vacuum == (95, 165)

    def test_default_mode_is_auto(self):
        structure = structure_with_c([0.1, 0.4])
        potential = np.arange(100.0)
        expected = alignment.get_regions(potential, structure, 0.25, mode='auto')
        result = alignment.get_regions(potential, structure, 0.25)
        np.testing.assert_array_equal(result[0], expected[0])
        assert result[1:] == expected[1:]

    @pytest.mark.parametrize('slab_start', [0.0, 0.1, 0.35, 0.55, 0.8, 0.95])
    def test_regions_follow_the_slab_wherever_it_sits(self, slab_start):
        """The bulk-like region is inside the slab and the vacuum region outside it, also across the cell boundary."""
        n = 400
        thickness = 0.3
        fractional_c = [(slab_start + f * thickness) % 1.0 for f in (0.0, 0.5, 1.0)]
        structure = structure_with_c(fractional_c)
        potential = step_potential(n, slab_start, slab_start + thickness)
        shifted, (b0, b1), (v0, v1) = alignment.get_regions(potential, structure, 0.25, mode='auto')
        assert np.all(shifted[b0:b1] == -5.0)
        assert np.all(shifted[v0:v1] == 0.0)

    def test_the_potential_is_only_rolled(self):
        n = 100
        structure = structure_with_c([0.6, 0.7, 0.8])
        potential = np.random.default_rng(0).normal(size=n)
        shifted, _, _ = alignment.get_regions(potential, structure, 0.25, mode='auto')
        np.testing.assert_array_equal(np.sort(shifted), np.sort(potential))
        np.testing.assert_array_equal(np.roll(shifted, 60), potential)

    def test_slab_crossing_the_boundary_has_the_same_thickness_as_when_it_does_not(self):
        n = 200
        inside = structure_with_c([0.1, 0.2, 0.3])
        crossing = structure_with_c([0.9, 0.0, 0.1])
        _, bulk_a, vac_a = alignment.get_regions(np.zeros(n), inside, 0.25, mode='auto')
        _, bulk_b, vac_b = alignment.get_regions(np.zeros(n), crossing, 0.25, mode='auto')
        assert bulk_a == bulk_b
        assert vac_a == vac_b

    @pytest.mark.parametrize('discard_ratio', [0.0, 0.1, 0.25, 0.4])
    @pytest.mark.parametrize('mode', ['midpoint', 'auto'])
    def test_regions_are_ordered_and_inside_the_array(self, mode, discard_ratio):
        n = 300
        structure = structure_with_c([0.05, 0.2, 0.35])
        _, (b0, b1), (v0, v1) = alignment.get_regions(np.zeros(n), structure, discard_ratio, mode=mode)
        assert 0 <= b0 < b1 <= v0 < v1 <= n

    def test_larger_discard_ratio_gives_smaller_regions(self):
        structure = structure_with_c([0.1, 0.4])
        _, bulk_small, vac_small = alignment.get_regions(np.zeros(500), structure, 0.1, mode='auto')
        _, bulk_large, vac_large = alignment.get_regions(np.zeros(500), structure, 0.3, mode='auto')
        assert bulk_large[1] - bulk_large[0] < bulk_small[1] - bulk_small[0]
        assert vac_large[1] - vac_large[0] < vac_small[1] - vac_small[0]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match='mode must be'):
        alignment.get_regions(np.zeros(10), structure_with_c([0.1, 0.2]), 0.25, mode='nearest')


class TestGetBandAlignment:
    """get_band_alignment: the formula of compute-band-alignments.ipynb on synthetic slabs with a known answer."""

    n = 400
    slab = (0.1, 0.4)                                               # fractional extent of the test slab

    def inputs(self, slab_value=-3.0, vacuum_value=2.0, start=None):
        start = self.slab[0] if start is None else start
        thickness = self.slab[1] - self.slab[0]
        structure = structure_with_c([(start + f * thickness) % 1.0 for f in (0.0, 0.5, 1.0)])
        return step_potential(self.n, start, start + thickness, slab_value, vacuum_value), structure

    def align(self, potential, structure, vbm_dft=1.5, band_gap=1.2, bulk_average=-4.0, **kwargs):
        return alignment.get_band_alignment(potential, structure, vbm_dft, band_gap, bulk_average, **kwargs)

    def test_known_values(self):
        potential, structure = self.inputs(slab_value=-3.0, vacuum_value=2.0)
        result = self.align(potential, structure)
        # VBM_vac = VBM_DFT + (V_bulk-like - V_bulk) - V_vac = 1.5 + (-3 + 4) - 2
        assert result.valence_band == pytest.approx(0.5)
        assert result.conduction_band == pytest.approx(1.7)
        assert result.bulk_like_potential == pytest.approx(-3.0)
        assert result.vacuum_potential == pytest.approx(2.0)

    def test_conduction_band_is_the_valence_band_plus_the_gap(self):
        potential, structure = self.inputs()
        for gap in (0.0, 0.7, 3.4):
            result = self.align(potential, structure, band_gap=gap)
            assert result.conduction_band - result.valence_band == pytest.approx(gap)

    def test_ionization_potential_and_electron_affinity(self):
        potential, structure = self.inputs(slab_value=-9.0, vacuum_value=0.0)
        result = self.align(potential, structure, vbm_dft=3.0, band_gap=1.5, bulk_average=0.0)
        assert result.valence_band == pytest.approx(-6.0)           # 3 + (-9 - 0) - 0
        assert result.ionization_potential == pytest.approx(6.0)
        assert result.electron_affinity == pytest.approx(4.5)
        assert result.ionization_potential - result.electron_affinity == pytest.approx(1.5)   # IP - EA = gap

    def test_a_constant_shift_of_the_slab_potential_changes_nothing(self):
        """Only potential differences matter: the arbitrary zero of the LOCPOT must drop out."""
        potential, structure = self.inputs()
        reference = self.align(potential, structure)
        shifted = self.align(potential + 7.3, structure)
        assert shifted.valence_band == pytest.approx(reference.valence_band)
        assert shifted.conduction_band == pytest.approx(reference.conduction_band)

    def test_the_bulk_reference_potential_enters_with_a_negative_sign(self):
        potential, structure = self.inputs()
        base = self.align(potential, structure, bulk_average=-4.0)
        assert self.align(potential, structure, bulk_average=-3.0).valence_band == pytest.approx(base.valence_band - 1.0)

    def test_the_dft_band_edge_enters_with_a_positive_sign(self):
        potential, structure = self.inputs()
        base = self.align(potential, structure, vbm_dft=1.5)
        moved = self.align(potential, structure, vbm_dft=2.0)
        assert moved.valence_band == pytest.approx(base.valence_band + 0.5)
        assert moved.conduction_band == pytest.approx(base.conduction_band + 0.5)

    def test_a_deeper_vacuum_level_raises_the_edges(self):
        """The same bulk-like potential with a higher vacuum level puts the bands deeper below it."""
        potential, structure = self.inputs(vacuum_value=2.0)
        low = self.align(potential, structure)
        high = self.align(*self.inputs(vacuum_value=3.0))
        assert high.valence_band == pytest.approx(low.valence_band - 1.0)

    @pytest.mark.parametrize('start', [0.0, 0.2, 0.55, 0.7, 0.95])
    def test_independent_of_where_the_slab_sits_in_the_cell(self, start):
        reference = self.align(*self.inputs())
        result = self.align(*self.inputs(start=start))
        assert result.valence_band == pytest.approx(reference.valence_band)
        assert result.conduction_band == pytest.approx(reference.conduction_band)

    def test_regions_and_rolled_potential_are_returned_for_plotting(self):
        potential, structure = self.inputs()
        result = self.align(potential, structure)
        shifted, bulk, vacuum = alignment.get_regions(potential, structure, 0.25, 'auto')
        np.testing.assert_array_equal(result.potential, shifted)
        assert result.bulk_region == bulk and result.vacuum_region == vacuum
        assert np.mean(result.potential[slice(*result.bulk_region)]) == pytest.approx(result.bulk_like_potential)
        assert np.mean(result.potential[slice(*result.vacuum_region)]) == pytest.approx(result.vacuum_potential)

    def test_discard_ratio_removes_the_edges_from_the_averages(self):
        """A slab whose potential rises towards its surfaces: discarding the edges lowers the bulk-like average."""
        n = 400
        potential, structure = self.inputs()
        z = (np.arange(n) + 0.5) / n
        potential = potential + np.where((z > 0.1) & (z < 0.4), 5.0 * np.abs(z - 0.25) / 0.15, 0.0)   # 0 at the centre, 5 at the edges
        wide = self.align(potential, structure, discard_ratio=0.0)
        narrow = self.align(potential, structure, discard_ratio=0.4)
        assert narrow.bulk_like_potential < wide.bulk_like_potential

    def test_midpoint_mode(self):
        potential = step_potential(self.n, 0.0, 0.5, -3.0, 2.0)
        result = alignment.get_band_alignment(potential, None, 1.5, 1.2, -4.0, mode='midpoint')
        assert result.valence_band == pytest.approx(0.5)

    def test_unknown_mode_raises(self):
        potential, structure = self.inputs()
        with pytest.raises(ValueError, match='mode must be'):
            self.align(potential, structure, mode='nearest')

    def test_discard_ratio_that_leaves_nothing_of_the_bulk_like_region_raises(self):
        potential, structure = self.inputs()
        with pytest.raises(ValueError, match='bulk-like region is empty'):
            self.align(potential, structure, discard_ratio=0.5)

    def test_empty_vacuum_region_raises(self, monkeypatch):
        monkeypatch.setattr(alignment, 'get_regions', lambda potential, structure, discard, mode: (potential, (0, 10), (50, 50)))
        with pytest.raises(ValueError, match='vacuum region is empty'):
            self.align(np.zeros(100), None)

    def test_results_are_plain_floats(self):
        result = self.align(*self.inputs())
        for value in (result.valence_band, result.conduction_band, result.bulk_like_potential, result.vacuum_potential,
                      result.ionization_potential, result.electron_affinity):
            assert type(value) is float

    def test_result_is_immutable(self):
        result = self.align(*self.inputs())
        with pytest.raises(Exception):                              # dataclasses.FrozenInstanceError
            result.valence_band = 0.0

    def test_the_input_potential_is_not_modified(self):
        potential, structure = self.inputs()
        before = potential.copy()
        self.align(potential, structure)
        np.testing.assert_array_equal(potential, before)
