from sklearn.cluster import DBSCAN
import numpy as np
from fuller.utils import loadHDF
import matplotlib.pyplot as plt

# use intensity-based clustering on ARPES data

# feature extraction based on density based clustering

# load ARPES band mapping

# Load preprocessed data
print("loading preprocessed data...")
data = loadHDF('../results/preprocessing/WSe2_preprocessed.h5')
E = data['E']
kx = data['kx']
ky = data['ky']
I = data['V']

print(I.shape)

# first take a 2D cross section of the band mapping
intensity_2d = I[128,:,:] # E = 0 cross section

# flip axes so that it is (kx, E) instead of (E, kx)
intensity_2d = np.swapaxes(intensity_2d, 0, 1)

print("kx = ", kx[128])

print(kx, ky)
print(intensity_2d.shape)

# plot the E = 0 cross section
print(intensity_2d.shape)

plt.imshow(intensity_2d)
plt.show()

# use the intensity values to cluster pixels
print(E.size, kx.size, ky.size, I.size)