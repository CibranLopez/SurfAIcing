from skopt import gp_minimize

from ase.build import stack
from ase.io import read

# Load two slabs (e.g., from CIFs or pymatgen)
slab1 = read("MoS2.cif")
slab2 = read("WS2.cif")


def energy_function(angle, distance):
    
    # Stack with a 30° rotation and 5Å vacuum
    heterostructure = stack(slab1, slab2, axis=2, rotate=30, vacuum=5.0)

    save heterostructure

    energy = sol.read_energy(path to heterostructure)

    return energy

res = gp_minimize(
    energy_function,
    dimensions=[(0, 30), (2.0, 6.0)],  # Angle and distance ranges
    n_calls=50,
    random_state=42
)
print(f"Best angle: {res.x[0]}°, distance: {res.x[1]}Å")
