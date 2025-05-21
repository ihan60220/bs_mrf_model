import pickle
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

# Create MRF model
print("creating mrf model...")
mrf = MrfRec(E=E, kx=kx, ky=ky, I=I, eta=.12)
mrf.I_normalized = True

# Initialize mrf model with band structure approximation from DFT
print("initializing E_0 to DFT calculations...")
path_dft = '../data/theory/WSe2_PBEsol_bands.mat'


def reconstruct(**kwargs):
    """
    Reconstructs a single band with hyperparameters\n

    parameters:\n
    **kwargs:\n
    offset: float\n
    k_scale: flaot\n
    eta: float\n
    n_epochs: int\n
    """

    # there is a total of 80 different bands
    offset = kwargs['offset']
    k_scale = kwargs['k_scale']
    eta = kwargs['eta']
    n_epochs = kwargs['n_epochs']

    kx_dft, ky_dft, E_dft = mrf.loadBandsMat(path_dft)

    # possible modify source to train multiple bands at once
    mrf.initializeBand(kx=kx_dft, ky=ky_dft, Eb=E_dft[2*band_index+1,...], offset=offset, kScale=k_scale, flipKAxes=True)

    # save the E0 as either an h5 or pickled file
    with open(f'../results/band_data/init_band_n={band_index}_epoch={n_epochs}.pkl', 'wb') as f:
      pickle.dump(mrf.E0, f) # serialize the list

    # Run optimization to perform reconstruction
    print("training model...")

    mrf.eta = eta
    # due to limited memory on laptop, reset graph every iteration
    mrf.iter_para(n_epochs, graph_reset=True) # if plotting objective vs loss, then set updatelogP=True

    # Save results
    mrf.saveBand(f"../results/band_data/bs_recon_n={band_index}_epoch={n_epochs}.h5", index=band_index)

# parameter
num_bands = 14 # there are around 80 different spin-split energy bands, but after the 14th, the energy levels are well beyond -20eV
n_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# use the odd-indexed bands, even-indexed bands giving errors

for epoch in n_epochs:
  for band_index in range(num_bands):
      print(f"reconstructing the {band_index}th band, epoch={epoch}, bi={2*band_index+1}...")
      reconstruct(band_index=2*band_index+1, offset=0.6, k_scale=1.1, eta=0.1, n_epochs=epoch)

  plot_bs(num_bands, epoch)