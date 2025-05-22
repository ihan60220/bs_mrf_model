import numpy as np
from fuller.mrfRec import MrfRec
from fuller.utils import loadHDF

# Load preprocessed data
print("loading preprocessed data...")
data = loadHDF('../results/preprocessing/WSe2_preprocessed.h5')
E = data['E']
kx = data['kx']
ky = data['ky']
I = data['V']

# take the kx=0 cut
I_2d = I[128, :, :]

print(kx, ky)

print(I_2d.shape, E.shape)

# save as npz
np.savez("./2d_slice", intensities=I_2d.T, energies=E)