
from fuller.mrfRec import MrfRec

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator


# import dft band data
path_dft = './data/theory/WSe2_HSE06_bands.mat'

kx, ky, E = MrfRec.loadBandsMat(path_dft)

band_index = 30  # there is a total of 80 different bands
offset = 0.4    # default was -0.1
k_scale = 1.0


# plotting

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

print(kx)
print(ky)

ax.scatter(kx, ky, E)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()