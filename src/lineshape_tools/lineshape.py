"""Implements functionality to evaluate optical lineshapes."""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numba import njit

from lineshape_tools.constants import kelvin2eV, omega2eV

logger = logging.getLogger(__name__)


def get_Stot(dq: np.ndarray, omega: np.ndarray) -> float:
    """Calculate the Huang-Rhys factor Stot.

    Args:
        dq (np.ndarray): mass-weighted displacement vector in normal mode basis in (amu^{1/2} Ang).
        omega (np.ndarray): normal mode phonon frequencies (in eV/amu/Ang^2).
    """
    return np.sum(0.5 * omega * dq**2 / omega2eV)


@njit(cache=True, fastmath=True, error_model="numpy")
def gaussian(x: np.ndarray, s: float) -> np.ndarray:
    """Evaluate the Gaussian function with smearing s."""
    return np.exp(-(x**2) / 2 / s**2) / np.sqrt(2 * np.pi) / s


@njit(cache=True, fastmath=True, error_model="numpy")
def lorentzian(x: np.ndarray, g: float) -> np.ndarray:
    """Evaluate the Lorentzian function with broadening g."""
    return g / np.pi / (x**2 + g**2)


class Broadening(NamedTuple):
    gamma_zpl: float
    sigma_zpl: float
    method_psb: np.ndarray
    value_psb: np.ndarray

    @classmethod
    def create(
        cls,
        omega: np.ndarray,
        gamma_zpl: float = 0.001,
        sigma_zpl: float = 0.0,
        sigma_psb: tuple[float, float] = (0.005, 0.001),
        gamma_lvm: float = 0.001,
        ipr_cut: float = 10.0,
        ipr: np.ndarray | None = None,
    ) -> Broadening:
        """Create an instance of Broadening.

        Args:
            omega (np.ndarray): normal mode phonon frequencies (in eV/amu/Ang^2).
            gamma_zpl (float): Lorentzian broadening in the ZPL to capture homogeneous broadening.
            sigma_zpl (float): Gaussian broadening in the ZPL to capture inhomogeneous broadening.
            sigma_psb (float, float): Gaussian broadening used to broaden the partial Huang-Rhys
                factors. The broadening factor is linearly interpolated from sigma_psb[0] at zero
                frequency to sigma_psb[1] at the highest (non-LVM) frequency.
            gamma_lvm (float): Lorentzian broadening applied to local vibrational modes identified
                by their inverse participation ratio.
            ipr_cut (float): Local vibrational modes are identified when their ipr < ipr_cut.
            ipr (np.ndarray): array of inverse participation ratios for each mode.
        """
        w_k = omega2eV * omega

        method_psb = np.zeros(omega.shape[0], dtype=np.uint8)

        if ipr is not None:
            mask = ipr < ipr_cut
            w_max = w_k[~mask].max()
        else:
            mask = None
            w_max = w_k.max()

        x = w_k / w_max
        value_psb = sigma_psb[0] * (1 - x) + sigma_psb[1] * x

        if ipr is not None:
            method_psb[mask] = 1
            value_psb[mask] = gamma_lvm

        return cls(
            gamma_zpl=gamma_zpl, sigma_zpl=sigma_zpl, method_psb=method_psb, value_psb=value_psb
        )


@njit(cache=True, fastmath=True, error_model="numpy")
def _do_compute_phonon_spec_func_zeroT(
    w: np.ndarray,
    dq: np.ndarray,
    omega: np.ndarray,
    broadening: Broadening,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the phonon spectral function and related quantities at zero T."""
    dw = w[1] - w[0]

    w_k = omega2eV * omega
    S_k = 0.5 * omega * dq**2 / omega2eV

    S_w, dos = np.zeros_like(w), np.zeros_like(w)
    for i in range(S_k.shape[0]):
        broad_f = lorentzian if broadening.method_psb[i] == 1 else gaussian
        S_w += S_k[i] * broad_f(w - w_k[i], broadening.value_psb[i])
        dos += broad_f(w - w_k[i], broadening.value_psb[i])
    dos /= S_k.shape[0]

    S_0 = S_w.sum() * dw
    if not np.isclose(S_k.sum(), S_0, rtol=1e-3):
        print(
            "Warning! Inconsistency in computed HR factors (",
            S_k.sum(),
            S_0,
            "). You may want to decrease sigma.",
        )
    S_t = np.fft.rfft(S_w) * dw
    t = 2 * np.pi * np.fft.rfftfreq(S_w.shape[0]) / dw

    G_t = np.exp(S_t - S_0 - broadening.gamma_zpl * t - 0.5 * broadening.sigma_zpl**2 * t**2)
    A_w = np.fft.irfft(G_t) / dw
    return dos, S_w, A_w


@njit(cache=True, fastmath=True, error_model="numpy")
def _do_compute_phonon_spec_func(
    w: np.ndarray,
    dq: np.ndarray,
    omega: np.ndarray,
    broadening: Broadening,
    T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the phonon spectral function and related quantities at arbitrary T."""
    dw = w[1] - w[0]
    kT = kelvin2eV * T

    w_k = omega2eV * omega
    S_k = 0.5 * omega * dq**2 / omega2eV

    n_k = np.zeros_like(S_k)
    n_k[w_k > 0] = 1 / (np.exp(w_k[w_k > 0] / kT) - 1)

    S_w, C_w, dos = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    for i in range(S_k.shape[0]):
        broad_f = lorentzian if broadening.method_psb[i] == 1 else gaussian
        S_w += S_k[i] * broad_f(w - w_k[i], broadening.value_psb[i])
        C_w += n_k[i] * S_k[i] * broad_f(w - w_k[i], broadening.value_psb[i])
        dos += broad_f(w - w_k[i], broadening.value_psb[i])
    dos /= S_k.shape[0]

    S_0, C_0 = (S_w.sum() * dw), (C_w.sum() * dw)
    if not np.isclose(S_k.sum(), S_0, rtol=1e-3):
        print(
            "Warning! Inconsistency in computed HR factors (",
            S_k.sum(),
            S_0,
            "). You may want to decrease sigma.",
        )

    S_t = np.fft.fft(S_w) * dw
    C_t = np.fft.fft(C_w) * dw
    t = 2 * np.pi * np.fft.fftfreq(S_w.shape[0]) / dw

    G_t = np.exp(
        S_t
        - S_0
        + C_t
        + C_t.conj()
        - 2 * C_0
        - broadening.gamma_zpl * np.abs(t)
        - 0.5 * broadening.sigma_zpl**2 * t**2
    )
    A_w = np.fft.ifft(G_t) / dw
    return dos, S_w, A_w.real


def get_phonon_spec_func(
    dq: np.ndarray,
    omega: np.ndarray,
    broadening: Broadening | None = None,
    resolution: float = 1e-3,
    w_max: float | None = None,
    pad: float = 0.1,
    T: float = 0.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the phonon spectral function and related quantities.

    If broadening is not provided, kwargs will be passed to Broadening.create
    (see :func:`Broadening.create`).

    When T == 0. (the default), an optimized evaluation of the spectral function is utilized.

    Args:
        dq (np.ndarray): mass-weighted displacement vector in normal mode basis in (amu^{1/2} Ang).
        omega (np.ndarray): normal mode phonon frequencies (in eV/amu/Ang^2).
        broadening (Broadening): instance of :class:`Broadening`.
        resolution (float): desired energy resolution (in eV) of the calculated spectral functions.
        w_max (np.ndarray): maximum frequency in grid.
        pad (float): energy (in eV) below ZPL to pad the spectral function.
        T (float): Temperature in kelvin.
        **kwargs: keyword arguments passed to :func:`Broadening.create` when broadening is not
            provided as input.

    Returns:
        w (np.ndarray): frequencies at which the functions were evaluated at (ħω)
        dos (np.ndarray): phonon density of states ρ(ħω)
        S_w (np.ndarray): Huang-Rhys spectral density S(ħω)
        A_w (np.ndarray): spectral function A(ħω)
    """
    broadening = broadening or Broadening.create(omega=omega, **kwargs)

    if resolution > broadening.gamma_zpl and resolution > broadening.sigma_zpl:
        resolution = max(broadening.gamma_zpl, broadening.sigma_zpl)
        logger.info(f"energy resolution is larger than ZPL broadening, decreasing to {resolution}")

    w_max = w_max or (pad + max(2 * get_Stot(dq, omega), 3) * omega.max() * omega2eV)
    N = int(w_max / resolution) + 1
    # ensure an even number of grid points for T = 0 evaluation
    if T <= 0.0:
        N += N % 2
    w = np.linspace(0.0, w_max, N)
    if N > 50_000:
        logger.warning(f"large number of grid points {N=}")
    logger.debug(f"made frequency grid with {w_max=} {N=}")

    if T > 0:
        dos, S_w, A_w = _do_compute_phonon_spec_func(w, dq, omega, broadening, T)
    else:
        dos, S_w, A_w = _do_compute_phonon_spec_func_zeroT(w, dq, omega, broadening)

    # padding at the end of w was for wrap-around in fft, need to roll arrays
    n_shift = np.sum(w >= w[-1] - pad)

    dos = np.roll(dos, n_shift)
    S_w = np.roll(S_w, n_shift)
    A_w = np.roll(A_w, n_shift)

    new_w = np.roll(w, n_shift)
    new_w[:n_shift] -= w[-1]

    return new_w, dos, S_w, A_w


def convert_A_to_L(
    w: np.ndarray,
    A: np.ndarray,
    dE: float,
    emission: bool = True,
    norm: str = "area",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert phonon spectral function into a (normalized) optical intensity.

    Args:
        w (np.ndarray): frequencies (in eV) where A is evaluated.
        A (np.ndarray): spectral function.
        dE (float): energy of the zero-phonon line.
        emission (bool): determines if the intensity corresponds to emission or absorption.
        norm (str): normalization of luminescence (area or max).

    Returns:
        new_w (np.ndarray): expanded range of frequencies where intensity is evaluated.
        L (np.ndarray): optical intensity L(ħω) on expanded frequencies new_w.
    """
    new_w = (dE - w) if emission else (dE + w)

    # remove negative frequencies, they should be superfluous anyway
    tmp_A = A[new_w >= 0.0]
    new_w = new_w[new_w >= 0.0]

    L = new_w**3 * tmp_A if emission else new_w * tmp_A
    if norm[0].lower() == "m":
        L /= L.max()
    else:
        L /= L.sum() * (w[1] - w[0])
    return new_w, L
