"""Implements functionality for handling phonons."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lineshape_tools.constants import omega2eV

if TYPE_CHECKING:
    from ase.atoms import Atoms


def get_ipr(U: np.ndarray) -> np.ndarray:
    """Evaluate the inverse participation ratio.

    Args:
        U (np.ndarray): matrix with phonon eigenvectors as columns (see np.linalg.eigh).
    """
    return 1 / np.sum(
        np.sum(U.reshape((U.shape[0] // 3, 3, U.shape[1])) ** 2, axis=1) ** 2, axis=0
    )


def get_disp_vect(atoms0: Atoms, atoms1: Atoms) -> np.ndarray:
    """Compute smallest displacement vector between two sets of atoms.

    Args:
        atoms0 (Atoms): first set of atoms
        atoms1 (Atoms): second set of atoms

    Returns:
        np.ndarray: 3N dimensional displacement vector in Å
    """
    dx = atoms0.get_scaled_positions() - atoms1.get_scaled_positions()
    dx -= np.round(dx)
    dx = dx @ atoms0.cell
    return dx.flatten()


def get_dq_vect(atoms0: Atoms, atoms1: Atoms) -> np.ndarray:
    """Compute smallest mass-weighted displacement vector between two sets of atoms.

    Args:
        atoms0 (Atoms): first set of atoms
        atoms1 (Atoms): second set of atoms

    Returns:
        np.ndarray: 3N dimensional displacement vector in amu^{1/2} Å
    """
    return np.repeat(np.sqrt(atoms0.get_masses()), 3) * get_disp_vect(atoms0, atoms1)


def get_phonons(
    dynmat: np.ndarray | Path | str, acoustic_tol: float = 5e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the phonon frequencies and eigenvectors from a dynamical matrix.

    Args:
        dynmat (np.ndarray | Path | str): path to a dynamical matrix in .npz format or a np.ndarray
            corresponding to a dynamical matrix (shape 3N x 3N).
        acoustic_tol (float): tolerance (in eV) to determine acoustic phonon modes.

    Returns:
        omega (np.ndarray): phonon frequencies (in eV/amu/Ang^2).
        U (np.ndarray): matrix with phonon eigenvectors as columns (see np.linalg.eigh).
    """
    if isinstance(dynmat, Path) or isinstance(dynmat, str):
        H = np.load(dynmat)["H"]
    else:
        H = dynmat

    omega2, U = np.linalg.eigh(H)
    omega2[omega2 < (acoustic_tol / omega2eV) ** 2] = 0.0
    return np.sqrt(omega2), U
