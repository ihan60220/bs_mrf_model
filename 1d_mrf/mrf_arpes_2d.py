"""
mrf_arpes_2d.py

Extracts band dispersions from 2D ARPES intensity maps I(k, ω)
and provides tools to extract 2D cuts along arbitrary k-space paths
(e.g., high-symmetry lines in WSe2) from full 3D ARPES data I(kx, ky, ω).

Includes realistic synthetic test data with adjustable noise and highlighted reconstruction overlay.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import argparse
import os


def load_data(intensity_path, energy_path=None):
    """
    Load 2D ARPES intensity map I(k, ω) and corresponding energy grid.
    Supports:
      - .npz with arrays 'intensities' (Nk x Ne) and 'energies' (Ne)
      - .npy, .csv, .txt for intensities (requires energy_path)
    Returns:
      intensities: np.ndarray shape (Nk, Ne)
      energies:     np.ndarray shape (Ne,)
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
            raise ValueError("energy_path must be provided when loading separate intensities")
        e_ext = os.path.splitext(energy_path)[1].lower()
        if e_ext == '.npy':
            energies = np.load(energy_path)
        elif e_ext in ['.csv', '.txt']:
            energies = np.loadtxt(energy_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported energy file type: {e_ext}")
    return intensities, energies


def load_3d_data(intensity_path, kx_path, ky_path, energy_path=None):
    """
    Load 3D ARPES data I(kx, ky, ω) plus momentum and energy axes.
    Supports .npz bundles with keys:
      - 'intensities': shape (Nkx, Nky, Ne)
      - 'kx':          shape (Nkx,)
      - 'ky':          shape (Nky,)
      - 'energies':    shape (Ne,)
    Or separate files for intensities (.npy/.csv/.txt) + kx, ky, energies.
    Returns:
      intensities3d: np.ndarray (Nkx, Nky, Ne)
      kx_vals:       np.ndarray (Nkx,)
      ky_vals:       np.ndarray (Nky,)
      energies:      np.ndarray (Ne,)
    """
    ext = os.path.splitext(intensity_path)[1].lower()
    if ext == '.npz':
        data = np.load(intensity_path)
        intensities3d = data['intensities']
        kx_vals = data['kx']
        ky_vals = data['ky']
        energies = data['energies']
    else:
        # Load intensities 3D array
        if ext == '.npy':
            intensities3d = np.load(intensity_path)
        elif ext in ['.csv', '.txt']:
            intensities3d = np.loadtxt(intensity_path, delimiter=',').reshape((-1,))
        else:
            raise ValueError(f"Unsupported intensity file type: {ext}")
        # load axes
        kx_vals = np.load(kx_path) if kx_path.endswith('.npy') else np.loadtxt(kx_path, delimiter=',')
        ky_vals = np.load(ky_path) if ky_path.endswith('.npy') else np.loadtxt(ky_path, delimiter=',')
        if energy_path is None:
            raise ValueError("energy_path must be provided when loading separate intensities")
        energies = np.load(energy_path) if energy_path.endswith('.npy') else np.loadtxt(energy_path, delimiter=',')
    return intensities3d, kx_vals, ky_vals, energies


def extract_2d_cut(intensities3d, kx_vals, ky_vals, energies, high_sym_points, n_points=300):
    """
    Extract a 2D cut I(s, ω) along a k-space path defined by high-symmetry points.

    Parameters:
    - intensities3d: np.ndarray shape (Nkx, Nky, Ne)
    - kx_vals, ky_vals: 1D arrays of momentum axes
    - energies:          1D energy array
    - high_sym_points:   list of (kx, ky) waypoints in same units as kx_vals, ky_vals
    - n_points:          total number of samples along path

    Returns:
    - cut_intensity: np.ndarray shape (n_points, Ne)
    - path_dist:     np.ndarray shape (n_points,) cumulative distance along path
    """
    # Build piecewise-linear path
    pts = np.array(high_sym_points)
    seg_vecs = pts[1:] - pts[:-1]
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    cumlen = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_len = cumlen[-1]
    # Sample distances
    path_dist = np.linspace(0, total_len, n_points)
    coords = np.zeros((n_points, 2))
    for i, d in enumerate(path_dist):
        # find segment index
        idx = np.searchsorted(cumlen, d, side='right') - 1
        idx = min(idx, len(seg_lengths)-1)
        frac = (d - cumlen[idx]) / seg_lengths[idx] if seg_lengths[idx] > 0 else 0
        coords[i] = pts[idx] + frac * seg_vecs[idx]
    # Interpolation
    interp = RegularGridInterpolator((kx_vals, ky_vals, energies), intensities3d,
                                     bounds_error=False, fill_value=0.0)
    # Query at each (kx,ky,ω)
    cut_intensity = np.zeros((n_points, len(energies)))
    for i, (kx_s, ky_s) in enumerate(coords):
        samp = np.column_stack([np.full_like(energies, kx_s),
                                 np.full_like(energies, ky_s),
                                 energies])
        cut_intensity[i] = interp(samp)
    return cut_intensity, path_dist

# (rest of file remains unchanged...)


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
    dp = np.zeros((N_k, N_e))
    backpointer = np.zeros((N_k, N_e), dtype=int)
    dp[0] = intensities[0]

    for i in range(1, N_k):
        for m in range(N_e):
            penalty = lambda_smooth * (energies[m] - energies)**2
            prev_scores = dp[i-1] - penalty
            j_best = np.argmax(prev_scores)
            dp[i, m] = intensities[i, m] + prev_scores[j_best]
            backpointer[i, m] = j_best

    path = np.zeros(N_k, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for i in range(N_k-2, -1, -1):
        path[i] = backpointer[i+1, path[i+1]]
    return energies[path]



def extract_multiple_bands(intensities, energies, lambda_smooth, num_bands, symmetry_axes=None, removal_width=5):
    """
    Iteratively extract multiple bands, plotting each as it's found.

    Parameters:
    - intensities:    2D ARPES map (Nk x Ne)
    - energies:       1D energy grid
    - lambda_smooth:  smoothness prior weight
    - num_bands:      number of bands to extract
    - symmetry_axes:  optional symmetrization axes
    - removal_width:  half-width (in bins) around each detected band to remove

    Returns:
    - bands: list of 1D arrays of length Nk
    """
    residual = preprocess_arpes(intensities, symmetry_axes=symmetry_axes)
    original = intensities
    bands = []
    N_k, N_e = residual.shape
    k_indices = np.arange(N_k)

    for idx in range(num_bands):
        band = extract_band_chain_mrf(residual, energies, lambda_smooth)
        bands.append(band)

        """
        # Plot current band
        plt.figure(figsize=(6,4))
        plt.imshow(residual.T, extent=[k_indices[0], k_indices[-1], energies[0], energies[-1]],
                   aspect='auto', origin='lower', cmap='gray')
        plt.plot(k_indices, band, color='cyan', linewidth=2, label=f'Band {idx+1}')
        plt.xlabel('Momentum index (k)')
        plt.ylabel('Energy (ω)')
        plt.title(f'Reconstructed Band {idx+1}')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        plt.show()
        """

        # Remove band region from residual
        nearest = np.argmin(np.abs(energies[None, :] - band[:, None]), axis=1)
        for i_k, j_e in enumerate(nearest):
            j0 = max(0, j_e - removal_width)
            j1 = min(N_e, j_e + removal_width + 1)
            residual[i_k, j0:j1] = 0.0

    return bands



def plot_bands(intensities, energies, bands, output_path):
    k = np.arange(intensities.shape[0])
    plt.figure(figsize=(8,6))
    plt.imshow(intensities.T, extent=[k[0], k[-1], energies[0], energies[-1]],
               aspect='auto', origin='lower', cmap='gray')
    for i, band in enumerate(bands):
        plt.plot(k, band, color='red', linewidth=3,
                 label='Extracted Band' if i==0 else None)
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


def test_random(output_path, noise_level=0.05, random_seed=None):
    # More realistic synthetic data: three bands + background + adjustable noise
    if random_seed is not None:
        np.random.seed(random_seed)
    N_k, N_e = 200, 400
    k = np.linspace(-1, 1, N_k)
    E = np.linspace(-6, 3, N_e)
    K, E_grid = np.meshgrid(k, E, indexing='ij')
    band1 = -1.5 * (K**2) + 0.2
    band2 = 0.8 * K - 1.0
    band3 = 0.5 * np.sin(np.pi * K) - 2.0
    gamma = 0.1
    lorentz = lambda x: (gamma/2)**2 / ((x)**2 + (gamma/2)**2)
    intensity = lorentz(E_grid - band1) + 0.8 * lorentz(E_grid - band2)
    intensity += 0.6 * lorentz(E_grid - band3)
    background = 0.2 + 0.05 * (E_grid - E_grid.min())
    intensity += background
    intensity = gaussian_filter(intensity, sigma=(2, 2))
    intensity += noise_level * np.random.rand(N_k, N_e)

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
    parser.add_argument('--test', action='store_true', help="Run on synthetic data")
    parser.add_argument('--noise_level', type=float, default=0.05, help="Noise amplitude for synthetic data")
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.test:
        test_random(args.output, noise_level=args.noise_level, random_seed=args.random_seed)
    else:
        intensities, energies = load_data(args.intensities, args.energies)
        bands = extract_multiple_bands(intensities, energies,
                                       args.lambda_smooth, args.num_bands,
                                       symmetry_axes=args.symmetry_axes)
        plot_bands(intensities, energies, bands, args.output)
