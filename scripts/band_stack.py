from fuller.generator import bandstack
from fuller.utils import loadHDF


# Load preprocessed data
print("loading preprocessed data...")
data = loadHDF('../data/pes/3_smooth.h5')
E = data['E']
kx = data['kx']
ky = data['ky']
I = data['V']

bs = bandstack()