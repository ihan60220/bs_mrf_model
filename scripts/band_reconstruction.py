import pickle
import matplotlib.pyplot as plt
from fuller.mrfRec import MrfRec
from fuller.utils import loadHDF
from visualize_dft_bands import plot_bs

# Load preprocessed data
print("loading preprocessed data...")
data = loadHDF('../results/preprocessing/WSe2_preprocessed.h5')
E = data['E']
kx = data['kx']
ky = data['ky']
I = data['V']

print(E.shape, kx.shape, ky.shape, I.shape)

# Create MRF model
print("creating mrf model...")
mrf = MrfRec(E=E, kx=kx, ky=ky, I=I, eta=.12)
mrf.I_normalized = True

# Initialize mrf model with band structure approximation from DFT
print("initializing E_0 to DFT calculations...")
path_dft = '../data/theory/WSe2_PBEsol_bands.mat'


def reconstruct(**kwargs):
    """
    Wrapper function for reconstructing band structure given some parameters
    """

    # there is a total of 80 different bands
    offset = kwargs['offset']
    k_scale = kwargs['k_scale']

    kx_dft, ky_dft, E_dft = mrf.loadBandsMat(path_dft)
    print("band structure shape:", E_dft.shape)

    # possible modify source to train multiple bands at once
    mrf.initializeBand(kx=kx_dft, ky=ky_dft, Eb=E_dft[band_index,...], offset=offset, kScale=k_scale, flipKAxes=True)

    # save the E0 as either an h5 or pickled file
    with open(f'../results/band_data/init_band_{band_index}.pkl', 'wb') as f:
      pickle.dump(mrf.E0, f) # serialize the list

    # Run optimization to perform reconstruction
    print("training model...")

    eta = kwargs['eta']
    n_epochs = kwargs['n_epochs']

    mrf.eta = eta
    mrf.iter_para(n_epochs, updateLogP=True)

    # Plot results
    print("plotting reconstructed bands...")
    mrf.plotBands(surfPlot=True)
    plt.tight_layout()
    plt.savefig("../results/reconstruction/bs_surface")

    mrf.plotI(ky=0, plotBand=True, plotBandInit=True, cmapName='YlOrBr', bandColor='cyan', initColor='lime')
    plt.tight_layout()
    plt.savefig("../results/reconstruction/band_ky_0")

    mrf.plotI(kx=0, plotBand=True, plotBandInit=True, cmapName='YlOrBr', bandColor='cyan', initColor='lime')
    plt.tight_layout()
    plt.savefig("../results/reconstruction/band_kx_0")

    # plot loss
    mrf.plotLoss()
    plt.savefig('../results/reconstruction/loss')

    # Save results
    path_save = '../results/band_data/'
    mrf.saveBand(f"{path_save}mrf_rec_{band_index}.h5", index=band_index)

    # think about maybe serializing the mrf for later use

# parameter
num_bands = 15
n_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# call the wrapper function
# kwargs: band_index, offset, k_scale, eta, n_epochs
for epoch in n_epochs:
  for band_index in range(num_bands):
      print(f"reconstructing the {band_index}th band, epoch={epoch}...")
      reconstruct(band_index=band_index, offset=0.6, k_scale=1.1, eta=0.1, n_epochs=epoch)
      plot_bs(num_bands, epoch)