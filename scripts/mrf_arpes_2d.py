"""
mrf_arpes_2d.py

Extracts band dispersions from 2D ARPES intensity maps I(k, ω)
using a Markov Random Field (chain) and dynamic programming.

Includes realistic synthetic test data and highlighted reconstruction overlay.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import argparse
import os


def load_data(intensity_path, energy_path=None):
    """
    Load ARPES intensity and energy grid from file(s).
    Supports:
      - .npz with arrays 'intensities' and 'energies'
      - .npy for intensities (requires energy_path)
      - .csv or .txt for intensities (requires energy_path)
    """
    ext = os.path.splitext(intensity_path)[1].lower()
    if ext == '.npz':
        data = np.load(intensity_path)
        intensities = data['intensities']
        energies = data['energies']
    else:
        if ext == '.npy':
            intensities = np.load(intensity_path)
        elif ext in ['.csv', '.txt']:
            intensities = np.loadtxt(intensity_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported intensity file type: {ext}")
        if energy_path is None:
            raise ValueError("energy_path must be provided when not loading .npz")
        e_ext = os.path.splitext(energy_path)[1].lower()
        if e_ext == '.npy':
            energies = np.load(energy_path)
        elif e_ext in ['.csv', '.txt']:
            energies = np.loadtxt(energy_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported energy file type: {e_ext}")
    return intensities, energies


def preprocess_arpes(intensities, symmetry_axes=None, sigma_smooth=1.0, clip_limits=(0.01, 0.99)):
    preproc = intensities.copy()
    if symmetry_axes:
        for axis in symmetry_axes:
            left = preproc[:axis]
            right = preproc[axis+1:][::-1]
            n = min(len(left), len(right))
            avg = 0.5 * (left[-n:] + right[:n])
            preproc[axis-n:axis] = avg
            preproc[axis+1:axis+1+n] = avg[::-1]
    lo, hi = np.quantile(preproc, clip_limits)
    preproc = np.clip(preproc, lo, hi)
    preproc = (preproc - lo) / (hi - lo)
    preproc = gaussian_filter(preproc, sigma=(sigma_smooth, sigma_smooth))
    return preproc


def extract_band_chain_mrf(intensities, energies, lambda_smooth):
    N_k, N_e = intensities.shape
    scores = intensities.copy()
    dp = np.zeros((N_k, N_e))
    backpointer = np.zeros((N_k, N_e), dtype=int)
    dp[0] = scores[0]
    for i in range(1, N_k):
        for m in range(N_e):
            penalty = lambda_smooth * (energies[m] - energies)**2
            prev_scores = dp[i-1] - penalty
            j_best = np.argmax(prev_scores)
            dp[i, m] = scores[i, m] + prev_scores[j_best]
            backpointer[i, m] = j_best
    path = np.zeros(N_k, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for i in range(N_k-2, -1, -1):
        path[i] = backpointer[i+1, path[i+1]]
    return energies[path]


def extract_multiple_bands(intensities, energies, lambda_smooth, num_bands, symmetry_axes=None):
    residual = preprocess_arpes(intensities, symmetry_axes=symmetry_axes)
    bands = []
    for _ in range(num_bands):
        band = extract_band_chain_mrf(residual, energies, lambda_smooth)
        bands.append(band)
        idx = np.argmin(np.abs(energies[None, :] - band[:, None]), axis=1)
        for i_k, j_e in enumerate(idx):
            j0 = max(0, j_e-1)
            j1 = min(residual.shape[1], j_e+2)
            residual[i_k, j0:j1] = 0.0
    return bands


def plot_bands(intensities, energies, bands, output_path):
    k = np.arange(intensities.shape[0])
    plt.figure(figsize=(8,6))
    plt.imshow(intensities.T, extent=[k[0], k[-1], energies[0], energies[-1]],
               aspect='auto', origin='lower', cmap='gray')
    # Highlight extracted bands in red
    for i, band in enumerate(bands):
        plt.plot(k, band, color='red', linewidth=3, label='Extracted Band' if i==0 else None)
    plt.xlabel('Momentum index (k)')
    plt.ylabel('Energy (ω)')
    plt.title('ARPES Intensity and Extracted Band')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Intensity')
    if bands:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def test_random(output_path):
    # More realistic synthetic data: three bands + background + noise
    N_k, N_e = 200, 400
    k = np.linspace(-1, 1, N_k)
    E = np.linspace(-6, 3, N_e)
    K, E_grid = np.meshgrid(k, E, indexing='ij')
    # Define three band dispersions
    band1 = -1.5 * (K**2) + 0.2                          # parabolic
    band2 = 0.8 * K - 1.0                               # linear
    band3 = 0.5 * np.sin(np.pi * K) - 2.0               # sinusoidal
    # Simulate peaks with Lorentzian broadening
    gamma = 0.1  # linewidth
    lorentz = lambda x: (gamma/2)**2 / ((x)**2 + (gamma/2)**2)
    intensity = lorentz(E_grid - band1) + 0.8 * lorentz(E_grid - band2)
    intensity += 0.6 * lorentz(E_grid - band3)
    # Add energy-dependent background
    background = 0.2 + 0.05 * (E_grid - E_grid.min())
    intensity += background
    # Smooth to simulate resolution
    intensity = gaussian_filter(intensity, sigma=(2, 2))
    # Add random noise
    intensity += 0.05 * np.random.rand(N_k, N_e)

    bands = extract_multiple_bands(intensity, E, lambda_smooth=10.0, num_bands=3)
    plot_bands(intensity, E, bands, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRF-based band extraction from 2D ARPES data")
    parser.add_argument('--intensities', help="Path to intensities (.npz, .npy, .csv)")
    parser.add_argument('--energies', help="Path to energies (.npy, .csv). Not needed with .npz")
    parser.add_argument('--output', default='bands.png', help="Output plot file")
    parser.add_argument('--lambda_smooth', type=float, default=5.0, help="Smoothness weight λ")
    parser.add_argument('--sigma_smooth', type=float, default=1.0, help="Gaussian smoothing σ")
    parser.add_argument('--num_bands', type=int, default=1, help="Number of bands to extract")
    parser.add_argument('--symmetry_axes', nargs='*', type=int, help="Symmetry axes indices")
    parser.add_argument('--test', action='store_true', help="Run on random synthetic data")
    args = parser.parse_args()

    if args.test:
        test_random(args.output)
    else:
        intensities, energies = load_data(args.intensities, args.energies)
        bands = extract_multiple_bands(intensities, energies,
                                       args.lambda_smooth, args.num_bands,
                                       symmetry_axes=args.symmetry_axes)
        plot_bands(intensities, energies, bands, args.output)
