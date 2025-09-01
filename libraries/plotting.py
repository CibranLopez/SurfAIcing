import numpy             as np
import matplotlib.pyplot as plt


def plot_ranking(
    surface_energies_of_formation,
    filename='ranking.eps',
    figsize=(15, 5),
    ylabel=r'$E_{\text{surface}}$ (eV/atom/Å$^2$)',
    dpi=50
):
    """Plot the ranking of the slabs based on their energy differences.

    Args:
        surface_energies_of_formation (pd.DataFrame): Energy differences of the slabs.
        filename      (str):          Filename to save the plot.
        figsize       (tuple):        Size of the figure.
        ylabel        (str):          Label for the y-axis.

    Returns:
        None
    """
    # Sorted in ascendent order
    min_arg      = np.argsort(surface_energies_of_formation.values)[0]
    local_minima = surface_energies_of_formation.columns[min_arg]
    energies     = surface_energies_of_formation.values[0][min_arg]

    # Energy differences in eV/supercell
    plt.figure(figsize=figsize)
    plt.plot(energies, 'o-')
    plt.xticks(range(len(local_minima)), local_minima, rotation='vertical')
    plt.ylabel(ylabel)
    #plt.ylim(0, 1)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()