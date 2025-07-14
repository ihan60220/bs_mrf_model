"""
Implements the optimizer for the 1D MRF using Iterated Conditional Modes
"""

import numpy as np

def penalty(energy_band, arpes_mapping, energy_val, idx, eta):
    """
    defines the local penalty function for some node in the mrf chain
    """
    for idx, node in enumerate(energy_band):
            # objective function

            if idx == 0:
                penalty = -np.log(arpes_mapping[idx]) +  (energy_val - energy_band[idx + 1])**2 / (2 * eta**2)

            elif idx == len(energy_band) - 1:
                penalty = -np.log(arpes_mapping[idx]) +  (energy_val - energy_band[idx - 1])**2 / (2 * eta**2)

            else:
                # assuming node is not an edge node
                penalty = -np.log(arpes_mapping[idx]) + (energy_val - energy_band[idx - 1])**2 / (2 * eta**2) + (node - energy_band[idx + 1])**2 / (2 * eta**2)

    return penalty


def iterated_conditional_modes(arpes_mapping, bin_size, initialization, num_iter, eta):
    """
    num_iter - number of iterations of coordinate descent to perform [int]
    energy_band - an array containing the 1D MRF values [2D array]
    intial_values - the initialization for the MRF optimization
    """
    
    # determing the binning intervals of the arpes mapping and bounds
    min = np.min(arpes_mapping)
    max = np.max(arpes_mapping)

    # create new binned energy values bounded by max and min
    binned_energy = np.arange(min, max, bin_size)

    # initialize the energy band
    energy_band = initialization.copy()

    # vectorize the penalty function
    v_penalty = np.vectorize(penalty)

    for _ in range(num_iter):
        # perform coordinate descent on each node
        for i in range(energy_band.size):
            # edge cases
            if i == 0:
                 # only look to the right
                applied_energy = v_penalty(binned_energy)
                correct_energy = binned_energy(np.argmin(applied_energy))

            elif i == energy_band.size - 1:
                 # only look to the left
                 applied_energy = v_penalty(binned_energy)
                 correct_energy = binned_energy(np.argmin(applied_energy))
            else:
                 applied_energy = v_penalty(binned_energy)
                 correct_energy = binned_energy(np.argmin(applied_energy))
             