from skopt import gp_minimize

def energy_function(angle, distance):
    heterostructure = generate_heterostructure(angle, distance)
    return compute_energy(heterostructure)  # Replace with ML/DFT energy

res = gp_minimize(
    energy_function,
    dimensions=[(0, 30), (2.0, 6.0)],  # Angle and distance ranges
    n_calls=50,
    random_state=42
)
print(f"Best angle: {res.x[0]}°, distance: {res.x[1]}Å")
