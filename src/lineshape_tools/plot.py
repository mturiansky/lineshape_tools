"""Various plotting utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lineshape_tools.constants import omega2eV
from lineshape_tools.lineshape import convert_A_to_L, get_phonon_spec_func
from lineshape_tools.phonon import get_ipr, get_phonons

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def _make_subplots(
    spec_funcs: list[tuple],
    dE: float,
    emission: bool,
    omega_max: float,
    omega_mult: float,
    figsize: tuple[float, float],
    skip_lim_adjust: bool,
    ax=None,
):
    """Helper function to make subplot-type plot."""
    if ax is not None:
        fig = ax[0].get_figure()
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    for spec_func in spec_funcs:
        w, dos, S, tw, L = spec_func

        if dos is not None:
            ax[0].plot(w, dos)

        if S is not None:
            ax[1].plot(w, S)

        if L is not None:
            ax[2].plot(tw, L)

    if not skip_lim_adjust:
        for a in ax:
            a.set_ylim((0, a.get_ylim()[1]))

        for a in ax[:2]:
            a.set_xlim((0.0, omega_max + 0.005))

        if emission:
            ax[2].set_xlim((dE - omega_mult * omega_max - 0.02, dE + 0.02))
        else:
            ax[2].set_xlim((dE - 0.02, dE + omega_mult * omega_max + 0.02))

    ax[0].set_xlabel(r"$\hbar\omega$ [eV]")
    ax[0].set_ylabel(r"DOS $\rho(\hbar\omega)$ [eV$^{-1}$]")
    ax[1].set_xlabel(r"$\hbar\omega$ [eV]")
    ax[1].set_ylabel(r"Spectral Density $S(\hbar\omega)$ [eV$^{-1}$]")
    ax[2].set_xlabel(r"Energy [eV]")
    if emission:
        ax[2].set_ylabel(r"Luminescence $L(\hbar\omega)$ [eV$^{-1}$]")
    else:
        ax[2].set_ylabel(r"Absorption $L(\hbar\omega)$ [eV$^{-1}$]")
    return fig, ax


def _make_inset(
    spec_funcs: list[tuple],
    dE: float,
    emission: bool,
    omega_max: float,
    omega_mult: float,
    figsize: tuple[float, float],
    skip_lim_adjust: bool,
    ax=None,
):
    """Helper function to make inset-type plot."""
    if ax is not None:
        fig = ax[0].get_figure()
    else:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)
        ax = [ax1, fig.add_axes((0.17, 0.55, 0.4, 0.35))]

    for spec_func in spec_funcs:
        w, _, S, tw, L = spec_func

        if L is not None:
            ax[0].plot(tw, L)

        if S is not None:
            ax[1].plot(w, S)

    if not skip_lim_adjust:
        for a in ax:
            a.set_ylim((0, a.get_ylim()[1]))

        ax[1].set_xlim((0.0, omega_max + 0.005))

        if emission:
            ax[0].set_xlim((dE - omega_mult * omega_max - 0.02, dE + 0.02))
        else:
            ax[0].set_xlim((dE - 0.02, dE + omega_mult * omega_max + 0.02))

    ax[0].set_xlabel(r"Energy [eV]")
    if emission:
        ax[0].set_ylabel(r"Luminescence $L(\hbar\omega)$ [eV$^{-1}$]")
    else:
        ax[0].set_ylabel(r"Absorption $L(\hbar\omega)$ [eV$^{-1}$]")
    ax[1].set_xlabel(r"$\hbar\omega$ [eV]")
    ax[1].set_ylabel(r"$S(\hbar\omega)$ [eV$^{-1}$]")
    return fig, ax


def _make_single(
    func_type: str,
    spec_funcs: list[tuple],
    dE: float,
    emission: bool,
    omega_max: float,
    omega_mult: float,
    figsize: tuple[float, float],
    skip_lim_adjust: bool,
    ax=None,
):
    """Helper function to make a single-function plot."""
    if ax is not None:
        fig = ax.get_figure()
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for spec_func in spec_funcs:
        w, dos, S, tw, L = spec_func

        if func_type[0].lower() == "d":
            ax.plot(w, dos)
        elif func_type[0].lower() == "s":
            ax.plot(w, S)
        else:
            ax.plot(tw, L)

    if not skip_lim_adjust:
        ax.set_ylim((0, ax.get_ylim()[1]))

        if func_type[0].lower() in ("d", "s"):
            ax.set_xlim((0.0, omega_max + 0.005))
        else:
            if emission:
                ax.set_xlim((dE - omega_mult * omega_max - 0.02, dE + 0.02))
            else:
                ax.set_xlim((dE - 0.02, dE + omega_mult * omega_max + 0.02))

    if func_type[0].lower() in ("d", "s"):
        ax.set_xlabel(r"$\hbar\omega$ [eV]")
    else:
        ax.set_xlabel(r"Energy [eV]")

    if func_type[0].lower() == "d":
        ax.set_ylabel(r"DOS $\rho(\hbar\omega)$ [eV$^{-1}$]")
    elif func_type[0].lower() == "s":
        ax.set_ylabel(r"Spectral Density $S(\hbar\omega)$ [eV$^{-1}$]")
    else:
        if emission:
            ax.set_ylabel(r"Luminescence $L(\hbar\omega)$ [eV$^{-1}$]")
        else:
            ax.set_ylabel(r"Absorption $L(\hbar\omega)$ [eV$^{-1}$]")
    return fig, ax


def plot_spec_funcs(
    dynmats: tuple | np.ndarray | Path | str | list[tuple | np.ndarray | Path | str],
    dq: np.ndarray | None,
    dE: float,
    gamma_zpl: float = 0.001,
    sigma_zpl: float = 0.0,
    sigma_psb: tuple[float, float] = (0.005, 0.001),
    gamma_psb: tuple[float, float] | None = None,
    emission: bool = True,
    omega_mult: float = 5.0,
    omega_max: float = 0.0,
    norm: str = "area",
    T: float = 0,
    plot_type: str = "subplot",
    figsize: tuple[float, float] = (8.0, 2.5),
    skip_lim_adjust: bool = False,
    ax=None,
):
    """Make a plot of the spectral functions and luminescence/absorption intensity.

    Args:
        dynmats (tuple | np.ndarray | Path | str): path to a dynamical matrix in .npz format or a
            np.ndarray corresponding to a dynamical matrix (shape 3N x 3N). If a tuple is given,
            assume that the spectral functions have already been obtained. The tuple should contain
            elements (w, dos, S, w_L, L) where w is the freq grid of dos/S and w_L is the freq grid
            of L. A list of dynmats can be provided instead.
        dq (np.ndarray): mass-weighted displacement vector in (amu^{1/2} Ang). Can be None if all
            dynmats that are provided are of type tuple (see above).
        dE (float): energy of the zero-phonon line.
        gamma_zpl (float): Lorentzian broadening in the ZPL to capture homogeneous broadening.
        sigma_zpl (float): Gaussian broadening in the ZPL to capture inhomogeneous broadening.
        sigma_psb (float, float): Gaussian broadening used to broaden the partial Huang-Rhys
            factors. The broadening factor is linearly interpolated from sigma_psb[0] at zero
            frequency to sigma_psb[1] at the highest (non-LVM) frequency.
        gamma_psb (float, float): Turns on Lorentzian broadening of local vibrational modes
            identified by their inverse participation ratio. gamma_psb[0] is ipr_cut and
            gamma_psb[1] is gamma_lvm. See :class:`Broadening`.
        emission (bool): determines if the intensity corresponds to emission or absorption.
        omega_mult (float): how many factors of the maximum phonon frequency from the ZPL will be
            plotted in the luminescence/absorption intensity.
        omega_max (float): maximum phonon frequency can be provided if known, used in determining
            the plot ranges.
        norm (str): normalization of luminescence (area or max).
        T (float): Temperature in kelvin.
        plot_type (str): type of plot to generate. "subplot" will generate a subplot for the dos, S
            and L, respectively. "inset" will plot L with S as an inset. To make a plot of a single
            type of function, specify "dos", "S", or "L".
        figsize (tuple): figsize passed to `matplotlib.pyplot.subplots`.
        skip_lim_adjust (bool): do not adjust xlim and ylim when making plots.
        ax (plt.Axes): matplotlib axes object (an array of them) if already made.
    """
    if not isinstance(dynmats, list):
        dynmats = [dynmats]

    spec_funcs = []
    for i, dynmat in enumerate(dynmats):
        if isinstance(dynmat, tuple):
            spec_funcs.append(dynmat)
            continue

        logger.info(f"dynmat {i} - diagonalizing")
        omega, U = get_phonons(dynmat)
        dq_k = U.T @ dq

        if gamma_psb is not None:
            logger.info(f"dynmat {i} - computing inverse participation ratios")
            ipr, ipr_cut, gamma_lvm = get_ipr(U), gamma_psb[0], gamma_psb[1]
        else:
            ipr, ipr_cut, gamma_lvm = None, None, None

        if omega2eV * omega.max() > omega_max:
            omega_max = omega2eV * omega.max()

        logger.info(f"dynmat{i} - evaluating spectral functions")
        w, dos, S, A = get_phonon_spec_func(
            dq_k,
            omega,
            sigma_psb=sigma_psb,
            gamma_zpl=gamma_zpl,
            sigma_zpl=sigma_zpl,
            ipr=ipr,
            ipr_cut=ipr_cut,
            gamma_lvm=gamma_lvm,
            T=T,
        )
        tw, L = convert_A_to_L(w, A, dE, emission=emission, norm=norm)

        spec_funcs.append((w, dos, S, tw, L))

    if omega_max <= 0.0 and not skip_lim_adjust:
        logger.warning("the maximum omega could not be determined, falling back to 0.1")
        omega_max = 0.1

    if plot_type[0].lower() == "i":
        return _make_inset(
            spec_funcs, dE, emission, omega_max, omega_mult, figsize, skip_lim_adjust, ax=ax
        )
    elif plot_type.lower()[:2] == "su":
        return _make_subplots(
            spec_funcs, dE, emission, omega_max, omega_mult, figsize, skip_lim_adjust, ax=ax
        )
    elif plot_type[0].lower() in ("d", "s", "l"):
        return _make_single(
            plot_type,
            spec_funcs,
            dE,
            emission,
            omega_max,
            omega_mult,
            figsize,
            skip_lim_adjust,
            ax=ax,
        )
    else:
        raise ValueError(f"unknown plot_type {plot_type}")
