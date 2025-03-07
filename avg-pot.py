import numpy              as np
import matplotlib.pyplot  as plt

# Read data
mac_pot = np.loadtxt('PLANAR_AVERAGE.dat')

plt.plot(mac_pot[:, 0], mac_pot[:, 1], label='Avg. potential')

mid_idx = int(0.5*len(mac_pot))

bulk_mac_pot   = mac_pot[:mid_idx]
vacuum_mac_pot = mac_pot[mid_idx:]

# Bulk
idx_ref = int(0.25*len(bulk_mac_pot))
bulk_valid_mac_pot = bulk_mac_pot[idx_ref:-idx_ref]
mean_avg_pot_bulk = np.mean(bulk_valid_mac_pot[: ,1])

print(f'Mean vacuum: {mean_avg_pot_bulk}')
plt.plot(bulk_valid_mac_pot[:, 0], bulk_valid_mac_pot[:, 1], label='Bulk')
plt.plot([bulk_valid_mac_pot[0, 0], bulk_valid_mac_pot[-1, 0]], [mean_avg_pot_bulk, mean_avg_pot_bulk])

# Vacuum
idx_ref = int(0.25*len(vacuum_mac_pot))
vacuum_valid_mac_pot = vacuum_mac_pot[idx_ref:-idx_ref]
mean_avg_pot_vacuum = np.mean(vacuum_valid_mac_pot[: ,1])

print(f'Mean vacuum: {mean_avg_pot_vacuum}')
plt.plot(vacuum_valid_mac_pot[:, 0], vacuum_valid_mac_pot[:, 1], label='Vacuum')
plt.plot([vacuum_valid_mac_pot[0, 0], vacuum_valid_mac_pot[-1, 0]], [mean_avg_pot_vacuum, mean_avg_pot_vacuum])

plt.show()
