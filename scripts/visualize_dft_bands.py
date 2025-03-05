import warnings as wn
wn.filterwarnings("ignore")

import os
import numpy as np
from fuller.mrfRec import MrfRec
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpes import fprocessing as fp, analysis as aly
from scipy import io as sio
from fuller.utils import loadHDF

# Load preprocessed photoemission data
bcsm = np.load(r'../data/processed/hslines/WSe2_vcut.npy')
Evals = fp.readBinnedhdf5(r'../data/pes/3_smooth.h5')['E']
ehi, elo = Evals[0], Evals[469]

paths = np.load(r'../data/processed/hslines/WSe2_kpath.npz')

dftbands = sio.loadmat('../data/theory/hslines/initials_DFT_G-M.mat')   
hse_th_shift = dftbands['HSE'][100:125, 0].max()

reconbands = {}

num_bands = 30 # default

bands = np.empty((num_bands, 256, 256))
for i in range(num_bands):
    band = loadHDF(f'../results/band_data/mrf_rec_{i}.h5')
    kx, ky, Eb = band['kx'], band['ky'], band['Eb']
    Eb.reshape(256, 256)
    bands[i] = Eb
    
bdi = aly.bandpath_map(np.moveaxis(bands, 0, 2), pathr=paths['rowInds'], pathc=paths['colInds'], eaxis=2)
reconbands["recon"] = bdi.T

"""
kx, ky, E = MrfRec.loadBandsMat('../data/theory/WSe2_PBEsol_bands.mat')
# initialization bands
print(kx.shape, ky.shape, E.shape)
for i in range(num_bands):
    print(E[i])
"""  


pos = paths['pathInds']
pos[-1] -= 1

ff, axa = plt.subplots(1, 1, figsize=(10.5, 8))
im = axa.imshow(bcsm, cmap='YlOrBr', extent=[0, 185, elo, ehi], aspect=12)

for ib in range(num_bands): # normally set to 14
    axa.plot(reconbands['recon'][:,ib], color='cyan', zorder=1)

axa.tick_params(axis='y', length=8, width=2, labelsize=15)
axa.tick_params(axis='x', length=0, labelsize=15, pad=8)
axa.set_ylim([elo, ehi])
axa.set_xticks(pos)
axa.set_xticklabels(['$\overline{\Gamma}$', '$\overline{\mathrm{M}}$',
                       '$\overline{\mathrm{K}}$', '$\overline{\Gamma}$'])
axa.set_ylabel('Energy (eV)', fontsize=20)
for p in pos[:-1]:
        axa.axvline(x=p, c='k', ls='--', lw=2, dashes=[4, 2])
        
axa.set_title('Reconstruction', fontsize=15, x=0.8, y=0.9)
cax = inset_axes(axa, width="3%", height="30%", bbox_to_anchor=(220, 90, 440, 200))
cb = plt.colorbar(im, cax=cax, ticks=[])
cb.ax.set_ylabel('Intensity', fontsize=15, rotation=-90, labelpad=17)

plt.savefig('../results/reconstruction/multiple_bands.png')