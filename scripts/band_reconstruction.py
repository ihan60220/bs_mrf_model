# Import packages
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from mpes import analysis as aly, fprocessing as fp
from fuller.mrfRec import MrfRec
from fuller.utils import loadHDF


# preprocessing steps for graphing along high symmetry points

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

# plotting function for high symmetry points
def plot_path(mrf, vmax, save_path, fname):
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

    full_path = save_path + fname
    
    plt.savefig(full_path, dpi=200)


# Load preprocessed data
print("loading preprocessed data...")
data = loadHDF('./data/pes/3_smooth.h5')
E = data['E']
kx = data['kx']
ky = data['ky']
I = data['V']

# Create MRF model
print("creating mrf model...")
mrf = MrfRec(E=E, kx=kx, ky=ky, I=I, eta=.12)
mrf.I_normalized = True

# debugging

print(E.size, kx.size, ky.size)

# Initialize mrf model with band structure approximation from DFT
print("initializing E_0 to DFT calculations...")
path_dft = './data/theory/WSe2_HSE06_bands.mat'

band_index = 15  # there is a total of 80 different bands
offset = 0.4    # default was -0.1
k_scale = 1.0

kx_dft, ky_dft, E_dft = mrf.loadBandsMat(path_dft)
print("band structure shape:", E_dft.shape)

# possible modify source to train multiple bands at once
mrf.initializeBand(kx=kx_dft, ky=ky_dft, Eb=E_dft[band_index,...], offset=offset, kScale=k_scale, flipKAxes=True)

# Plot slices with initialiation to check offset and scale
print("plotting initialized mrf...")
mrf.plotI(ky=0, plotBandInit=True, cmapName='YlGn', bandColor='tab:orange', initColor='m')
plt.savefig("./results/reconstruction/init_ky_0")
mrf.plotI(kx=0, plotBandInit=True, cmapName='YlGn', bandColor='tab:orange', initColor='m')
plt.savefig("./results/reconstruction/init_kx_0")

# Run optimization to perform reconstruction
print("training model...")
eta = 0.1 # default 0.1
n_epochs = 150 # default 150

mrf.eta = eta
mrf.iter_para(n_epochs)
#mrf.iter_seq(n_epochs)  # doing sequentially

# Plot results
print("plotting reconstructed bands...")
mrf.plotBands(surfPlot=True)
plt.savefig("./results/reconstruction/bs_surface")
mrf.plotI(ky=0, plotBand=True, plotBandInit=True, cmapName='YlGn', bandColor='tab:orange', initColor='m')
plt.savefig("./results/reconstruction/band_ky_0")
mrf.plotI(kx=0, plotBand=True, plotBandInit=True, cmapName='YlGn', bandColor='tab:orange', initColor='m')
plt.savefig("./results/reconstruction/band_kx_0")

# Save results
path_save = 'results/reconstruction/'
mrf.saveBand(f"{path_save}mrf_rec_{band_index}.h5", index=band_index)

# think about maybe serializing the mrf for later use