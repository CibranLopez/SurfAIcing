import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt

from libraries import plotting


@pytest.fixture(autouse=True)
def no_show(monkeypatch):
    """plt.show() would try to open a window; keep the figure so that it can be inspected."""
    monkeypatch.setattr(plotting.plt, 'show', lambda: None)


def frame(values, labels):
    return pd.DataFrame([values], columns=labels)


class TestFormatSlabLabel:
    @pytest.mark.parametrize('label, expected', [
        ('0_0_1_i_3',   r'$\mathregular{(001)_{03}}$'),     # index padded to two digits
        ('1_1_1_i_24',  r'$\mathregular{(111)_{24}}$'),
        ('1_-1_0_i_10', r'$\mathregular{(1-10)_{10}}$'),
        ('1_0_0_7',     r'$\mathregular{(100)_{7}}$'),      # legacy four-field names are not padded
    ])
    def test_known_formats(self, label, expected):
        assert plotting.format_slab_label(label) == expected

    @pytest.mark.parametrize('label', ['custom', 'a_b', '1_1_0_1_i_3', '0_0_1_x_3', '0_0_1_i_x'])
    def test_unknown_formats_are_returned_unchanged(self, label):
        assert plotting.format_slab_label(label) == label


class TestPlotRanking:
    def test_slabs_are_sorted_by_ascending_energy(self, tmp_path):
        df = frame([0.9, 0.2, 0.5], ['0_0_1_i_0', '1_1_1_i_1', '1_0_0_i_2'])
        plotting.plot_ranking(df, filename=str(tmp_path / 'ranking.png'))
        ax = plt.gcf().axes[0]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.2, 0.5, 0.9])
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            r'$\mathregular{(111)_{01}}$', r'$\mathregular{(100)_{02}}$', r'$\mathregular{(001)_{00}}$']

    def test_file_is_saved(self, tmp_path):
        out = tmp_path / 'ranking.png'
        plotting.plot_ranking(frame([0.3, 0.1], ['0_0_1_i_0', '1_1_1_i_1']), filename=str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_nan_slabs_are_left_out_and_reported(self, tmp_path, capsys):
        df = frame([0.3, np.nan, 0.1], ['0_0_1_i_0', '1_1_1_i_1', '1_0_0_i_2'])
        plotting.plot_ranking(df, filename=str(tmp_path / 'r.png'))
        assert '1 slab(s) without a valid energy' in capsys.readouterr().out
        assert len(plt.gcf().axes[0].lines[0].get_ydata()) == 2

    def test_no_message_when_every_energy_is_valid(self, tmp_path, capsys):
        plotting.plot_ranking(frame([0.3, 0.1], ['0_0_1_i_0', '1_1_1_i_1']), filename=str(tmp_path / 'r.png'))
        assert 'left out' not in capsys.readouterr().out

    def test_all_nan_raises_a_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match='nothing to plot'):
            plotting.plot_ranking(frame([np.nan], ['0_0_1_i_0']), filename=str(tmp_path / 'r.png'))

    def test_unrecognised_labels_do_not_break_the_plot(self, tmp_path):
        df = frame([0.5, 0.3], ['slab_a', '0_0_1_i_3'])
        plotting.plot_ranking(df, filename=str(tmp_path / 'r.png'))
        labels = [t.get_text() for t in plt.gcf().axes[0].get_xticklabels()]
        assert labels == [r'$\mathregular{(001)_{03}}$', 'slab_a']

    def test_positive_energies_start_the_axis_at_zero(self, tmp_path):
        plotting.plot_ranking(frame([0.5, 0.3], ['0_0_1_i_0', '1_1_1_i_1']), filename=str(tmp_path / 'r.png'))
        assert plt.gcf().axes[0].get_ylim()[0] == 0

    def test_negative_energies_extend_the_axis_below_the_minimum(self, tmp_path):
        plotting.plot_ranking(frame([-0.4, 0.3], ['0_0_1_i_0', '1_1_1_i_1']), filename=str(tmp_path / 'r.png'))
        assert plt.gcf().axes[0].get_ylim()[0] == pytest.approx(-0.44)

    def test_label_and_size_are_configurable(self, tmp_path):
        plotting.plot_ranking(frame([0.3, 0.1], ['0_0_1_i_0', '1_1_1_i_1']), filename=str(tmp_path / 'r.png'),
                              figsize=(4, 3), ylabel='gamma')
        figure = plt.gcf()
        assert tuple(figure.get_size_inches()) == (4, 3)
        assert figure.axes[0].get_ylabel() == 'gamma'
