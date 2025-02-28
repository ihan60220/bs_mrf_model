import warnings as wn
wn.filterwarnings("always")

import numpy as np
from fuller.mrfRec import MrfRec
from fuller.generator import rotosymmetrize
from fuller.utils import saveHDF
from mpes import analysis as aly, fprocessing as fp

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# high symmetry points need to be defined for graphic Brioullin zone

fdata = fp.readBinnedhdf5('./data/pes/1_sym.h5')
mc = aly.MomentumCorrector(fdata['V'])

mc.selectSlice2D(selector=slice(30, 32), axis=2)
mc.featureExtract(mc.slice, method='daofind', sigma=6, fwhm=20, symscores=False)

# False detection filter, if needed
try:
    mc.pouter_ord = mc.pouter_ord[[0,1,3,5,6,9],:]
except:
    pass

# Define high-symmetry points
G = mc.pcent # Gamma point
K = mc.pouter_ord[0,:] # K point
K1 = mc.pouter_ord[1,:] # K' point
M = (K + K1) / 2 # M point

# Define cutting path
pathPoints = np.asarray([G, M, K, G])
nGM, nMK, nKG = 70, 39, 79
segPoints = [nGM, nMK, nKG]
rowInds, colInds, pathInds = aly.points2path(pathPoints[:,0], pathPoints[:,1], npoints=segPoints)
nSegPoints = len(rowInds)

# Define plotting function

def plot_path(mrf, vmax, save_path):
    # Normalize data
    imNorm = mrf.I / mrf.I.max()

    # Sample the data along high-symmetry lines (k-path) connecting the corresponding high-symmetry points
    pathDiagram = aly.bandpath_map(imNorm, pathr=rowInds, pathc=colInds, eaxis=2)

    Evals = mrf.E
    ehi, elo = Evals[0], Evals[449]

    f, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(pathDiagram[:450, :], cmap='Blues', aspect=10.9, extent=[0, nSegPoints, elo, ehi], vmin=0, vmax=vmax)
    ax.set_xticks(pathInds)
    ax.set_xticklabels(['$\overline{\Gamma}$', '$\overline{\mathrm{M}}$',
                        '$\overline{\mathrm{K}}$', '$\overline{\Gamma}$'], fontsize=15)
    for p in pathInds[:-1]:
        ax.axvline(x=p, c='r', ls='--', lw=2, dashes=[4, 2])
    # ax.axhline(y=0, ls='--', color='r', lw=2)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('Energy (eV)', fontsize=15, rotation=-90, labelpad=20)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.tick_params(which='both', axis='y', length=8, width=2, labelsize=15)
    
    plt.savefig(save_path, dpi=200)

# Load data
data = fp.readBinnedhdf5('./data/pes/0_binned.h5')
I = data['V']
E = data['E']
kx = data['kx']
ky = data['ky']

# Create reconstruction object from data file
mrf = MrfRec(E=E, kx=kx, ky=ky, I=I)

# preprocessing steps
print("symmetrizing...")
mrf.symmetrizeI()
plot_path(mrf, 0.5, './results/symmetrized')
print("normalizing...")
mrf.normalizeI(kernel_size=(20, 20, 25), n_bins=256, clip_limit=0.15, use_gpu=False)
plot_path(mrf, 0.5, './results/normalized')
print("smoothing...")
mrf.smoothenI(sigma=(.8, .8, 1.))
plot_path(mrf, 1, './results/smoothened')