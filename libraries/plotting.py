import numpy             as np
import matplotlib.pyplot as plt
import multiprocessing   as mp
import re
import subprocess
import os

from mace.calculators            import mace_mp
from ase.md                      import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase                         import units
from ase.io.vasp                 import read_vasp, write_vasp
from ase.optimize                import BFGS
from ase.constraints             import ExpCellFilter


linewidth    = 0.5
footnotesize = 8


def plot_ranking(
    slab_energies,
    filename='ranking.eps',
    figsize=(15, 5),
    ylabel=r'$\Delta E$ (eV/atom/Å^2)'
):
    """Plot the ranking of the slabs based on their energy differences.

    Args:
        slab_energies (pd.DataFrame): Energy differences of the slabs.
        filename      (str):          Filename to save the plot.
        figsize       (tuple):        Size of the figure.
        ylabel        (str):          Label for the y-axis.

    Returns:
        None
    """
    # Sorted in ascendent order
    min_arg      = np.argsort(slab_energies.values)[0]
    local_minima = slab_energies.columns[min_arg]
    energies     = slab_energies.values[0][min_arg]

    # Energy differences in eV/supercell
    plt.figure(figsize=figsize)
    plt.plot(energies, 'o-')
    plt.xticks(range(len(local_minima)), local_minima, rotation='vertical')
    plt.ylabel(ylabel)
    plt.savefig(filename, dpi=50, bbox_inches='tight')
    plt.show()