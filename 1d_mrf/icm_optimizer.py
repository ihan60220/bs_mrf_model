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


def iterated_conditional_modes(num_iter, arpes_mapping, initial_values, eta):
    """
    num_iter - number of iterations of coordinate descent to perform [int]
    energy_band - an array containing the 1D MRF values [2D array]
    intial_values - the initialization for the MRF optimization
    """
    
    # initialize the energy band
    energy_band = initial_values.copy()

    for _ in range(num_iter):
        # perform coordinate descent on each node
        pass