import pickle

# unpickle all the dft bands from earlier runs


"""
dft_bands = []

for i in range(14):
    with open(f"./dft_bands/init_band_n={i}_epoch=1.pkl", 'rb') as dft_file:
        dft_band = pickle.load(dft_file)
        print(dft_band)

        dft_bands.append(dft_band)

print(dft_bands)
"""

import numpy as np
import scipy.io

dft_bands = np.asarray(scipy.io.loadmat('../data/theory/WSe2_HSE06_bands.mat')['evb'])

print(dft_bands.shape)


bdi = aly.bandpath_map(np.moveaxis(init_bands, 0, 2), pathr=paths['rowInds'], pathc=paths['colInds'], eaxis=2)
    reconbands["init"] = bdi.T
    bdi = aly.bandpath_map(np.moveaxis(recon_bands, 0, 2), pathr=paths['rowInds'], pathc=paths['colInds'], eaxis=2)
    reconbands["recon"] = bdi.T