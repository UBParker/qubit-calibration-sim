"""Robust statistics for the outlier and residual-structure gates.

Masking happens around QM's fitters, never inside them: a node flags points here, hands QM's fitter
the kept points, and records what it masked.

- Robust scale: the median absolute deviation times 1.4826, consistent for Gaussian data
  (Rousseeuw & Croux 1993, https://doi.org/10.1080/01621459.1993.10476408).
- Isolated points against a running median: the Hampel identifier idea (Hampel 1974,
  https://doi.org/10.1080/01621459.1974.10482962).
- Serial correlation of residuals: Durbin & Watson 1950 (https://doi.org/10.1093/biomet/37.3-4.409).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import chi2

MAD_TO_SIGMA = 1.4826


def robust_sigma(x: np.ndarray) -> float:
    """MAD scale estimate: 1.4826 median(|x - median(x)|)."""
    x = np.asarray(x, dtype=float)
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def iq_line_outliers(z: np.ndarray, k: float = 6.0, n_iter: int = 3) -> np.ndarray:
    """Bool mask of averaged I/Q points that leave the g-e line.

    Dispersive readout of g/e mixtures puts every averaged point on the segment between the |g> and
    |e> responses, so the perpendicular component is ADC noise only (shot noise lies along the line).
    keep = all; repeat n_iter: centre = mean(z[keep]); axis = first right-singular vector of
    (z[keep] - centre); perp = Im((z - centre) conj(axis)); keep = |perp - median(perp[keep])| <= k
    robust_sigma(perp[keep]).
    """
    z = np.asarray(z, dtype=complex)
    keep = np.ones(z.size, dtype=bool)
    for _ in range(n_iter):
        centre = z[keep].mean()
        c = z[keep] - centre
        _, _, vh = np.linalg.svd(np.stack([c.real, c.imag]).T, full_matrices=False)
        axis = complex(vh[0, 0], vh[0, 1])
        perp = ((z - centre) * np.conj(axis)).imag
        scale = robust_sigma(perp[keep])
        if scale <= 0:
            break
        keep = np.abs(perp - np.median(perp[keep])) <= k * scale
    return ~keep


def residual_outliers(residual: np.ndarray, k: float = 7.0) -> np.ndarray:
    """|r - median(r)| > k robust_sigma(r). k = 7, not 5: shot noise is heteroscedastic along the line
    (a clean 05 run reached 5.0)."""
    r = np.asarray(residual, dtype=float)
    scale = robust_sigma(r)
    if scale <= 0:
        return np.zeros(r.size, dtype=bool)
    return np.abs(r - np.median(r)) > k * scale


# Median of |z - m| (m the 3-point complex running median) for complex Gaussian noise of unit rms per
# quadrature, measured on 2e6 samples: 0.845.
MEDIAN3_DEVIATION = 0.845


def isolated_spikes(z: np.ndarray, k: float = 9.0, frac: float = 0.5) -> np.ndarray:
    """Hampel-style test for narrow-line sweeps (Hampel 1974).

    m_k = complex(median of Re, median of Im) over points k-1..k+1 (end points mirror their
    neighbour); d_k = |z_k - m_k|; scale s = sqrt(2) median(d) / MEDIAN3_DEVIATION over the whole
    sweep, so s estimates sqrt(2) times the per-quadrature noise rms; b = complex median of z.
    Flag d_k > k s and |m_k - b| < frac |z_k - b|. A real feature at least 2 steps wide keeps its
    neighbours above half its excursion and is never flagged. Two adjacent glitches are not caught
    (documented limitation).

    The scale is one number from d, not a 21-point running median of |diff(z)| as first planned.
    On the flanks of a line |diff(z)| is large while d stays at the noise (the 3-point median of a
    monotone run is its middle point): the diff-based scale two FWHM from a FWHM-3-step line was
    2.5 times its value far from it and hid a 30 sigma spike there. A running median of d instead
    collapsed near clean 03a lines (clean normalised deviation up to 79 in 31 bring-up seeds). The
    whole-sweep median of d gave a clean maximum of 4.6 over 02a and 03a in seeds 0 to 30.
    """
    z = np.asarray(z, dtype=complex)
    n = z.size
    if n < 3:
        return np.zeros(n, dtype=bool)
    re = median_filter(z.real, size=3, mode="mirror")
    im = median_filter(z.imag, size=3, mode="mirror")
    m = re + 1j * im
    d = np.abs(z - m)
    s = math.sqrt(2) * float(np.median(d)) / MEDIAN3_DEVIATION
    b = complex(np.median(z.real), np.median(z.imag))
    return (d > k * s) & (np.abs(m - b) < frac * np.abs(z - b))


def durbin_watson(residual: np.ndarray) -> float:
    """Σ diff(r)² / Σ (r - mean r)². About 2 for white residuals, toward 0 for slow structure.
    Durbin & Watson 1950."""
    r = np.asarray(residual, dtype=float)
    denom = float(np.sum((r - r.mean()) ** 2))
    return float(np.sum(np.diff(r) ** 2) / denom) if denom > 0 else float("nan")


def residual_peak(residual: np.ndarray, fitted_amplitude: float) -> tuple[float, float, int]:
    """3-point boxcar of the residual; noise s = robust_sigma(diff(residual)) / sqrt(2); returns
    (max(boxcar) / (s / sqrt(3)), max(boxcar) / fitted_amplitude, index of the maximum)."""
    r = np.asarray(residual, dtype=float)
    box = np.convolve(r, np.ones(3) / 3, mode="same")
    s = robust_sigma(np.diff(r)) / math.sqrt(2)
    i = int(np.argmax(box))
    z = float(box[i] / (s / math.sqrt(3))) if s > 0 else float("inf")
    fraction = float(box[i] / fitted_amplitude) if fitted_amplitude else float("inf")
    return z, fraction, i


def shot_outliers(I: np.ndarray, Q: np.ndarray, k: float = 12.0) -> np.ndarray:
    """Shots further than k sigma from their preparation's complex median, sigma = mean of
    robust_sigma(I), robust_sigma(Q). Decay during readout smears |e> shots toward |g> by about
    5 sigma; 12 sigma keeps them."""
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    centre = complex(np.median(I), np.median(Q))
    sigma = 0.5 * (robust_sigma(I) + robust_sigma(Q))
    return np.abs(I + 1j * Q - centre) > k * sigma


def block_homogeneity_pvalue(wrong: np.ndarray, n_blocks: int = 20) -> float:
    """Chi-square test (n_blocks - 1 dof) that the wrong-assignment fraction is the same in consecutive
    equal blocks of shots (acquisition order). 1.0 when there are no wrong assignments."""
    w = np.asarray(wrong, dtype=bool)
    total = int(w.sum())
    if total == 0 or total == w.size:
        return 1.0
    blocks = np.array_split(w, n_blocks)
    sizes = np.array([b.size for b in blocks], dtype=float)
    observed = np.array([b.sum() for b in blocks], dtype=float)
    p = total / w.size
    expected = sizes * p
    stat = float(np.sum((observed - expected) ** 2 / (expected * (1 - p))))
    return float(chi2.sf(stat, n_blocks - 1))
