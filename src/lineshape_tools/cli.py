"""Contains functionality for the command line interface."""

from __future__ import annotations

import logging
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import ase.io
import numpy as np
from cyclopts import Parameter
from tqdm import tqdm

from lineshape_tools.constants import omega2eV
from lineshape_tools.lineshape import convert_A_to_L, gaussian, get_phonon_spec_func, get_Stot
from lineshape_tools.phonon import get_disp_vect, get_ipr, get_phonons
from lineshape_tools.plot import plot_spec_funcs

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from mace.calculators import MACECalculator

logger = logging.getLogger(__name__)


def collect(
    files: Annotated[list[Path], Parameter(negative="")],
    output_file: Path = Path("./database.extxyz"),
    strategy: str = "none",
    read_index: str = ":",
    max_force: float = 2.0,
    min_force: float = -np.inf,
    dx_tol: float = 0.1,
    rtol: float = 1e-5,
    config_weight: float = 1.0,
    force_weighting: bool = False,
    allow_constraint: bool = False,
) -> None:
    """Collect and process data for fine-tuning.

    Collect files into an extxyz database that can be used for fine-tuning. An optional filtering
    strategy can be applied. This is potentially useful if relaxation data is being included in the
    dataset, as it can be noisey from having multiple closely spaced geometries close to the
    equilibrium geomtry. By default, configurations with too large forces will be thrown away to
    avoid potential anharmonic contributions to the PES.

    Args:
        files (list): list of paths to files that are parseable by ase.io. The files should contain
            atomic geometries, total energies, and forces at a minimum (for example, vasprun.xml).
        output_file (Path): optional path where data is written (output to stdout by default)
        strategy (str): optional specification of strategy to be used for filtering. Available
            options are 'none', 'qr', or 'dx'.
        read_index (str): pythonic index passed to ase.io.read to determine which structures are
            read from the input files (the same value is used for each file). The default ":" reads
            all of the structures, while ":3" would read the first three for example.
        max_force (float): remove structures where the maximum force acting on any atom is above
            the specified value (in eV/Å)
        min_force (float): remove structures where the maximum force acting on any atom is below
            the specified value (in eV/Å)
        dx_tol (float): tolerance (in Å) for how far atoms must move to accept configuration in
            'dx' filtering strategy
        rtol (float): tolerance ratio for determining rank of displacement vectors in 'qr' strategy
        config_weight (float): set the configuration weight for training
        force_weighting (bool): store a config_weight that's inversely proportional to the max
            force that any atom feels in the configuration [min(0.02 / max_fpa, 1)]. Overwrites
            the value specified by config_weight.
        allow_constraint (bool): allow constraints on atoms (e.g. selective dynamics) to modify the
            forces. This is typically not desirable.
    """
    if strategy.lower() not in ("none", "qr", "dx"):
        raise ValueError("invalid strategy choice")

    if output_file.exists():
        logger.info(f"{output_file} already exists, appending to it")

    total_read = 0
    all_atoms = []
    for fname in tqdm(files, desc="[*] reading files", disable=len(files) < 2):
        read_atoms = ase.io.read(fname, read_index)
        total_read += len(read_atoms)
        for atoms in tqdm(read_atoms, desc="[*] processing atoms", disable=len(read_atoms) < 10):
            if len(atoms.constraints) > 0 and not allow_constraint:
                logger.debug("atoms are constrained, removing prior to extracting forces")
                atoms.set_constraint(None)

            forces = atoms.get_forces()
            if min_force < np.linalg.norm(forces, axis=1).max() < max_force:
                atoms.info["REF_energy"] = atoms.get_potential_energy()
                atoms.new_array("REF_forces", forces)
                atoms.calc = None
                atoms.info["config_weight"] = config_weight
                if force_weighting:
                    atoms.info["config_weight"] = min(
                        0.02 / np.linalg.norm(forces, axis=1).max(), 1
                    )
                    if (cw := atoms.info["config_weight"]) < 0.02:
                        logger.warning(f"small config weight found ({cw})")
                all_atoms.append(atoms)

    logger.info(
        f"read {total_read} configurations and discarded {total_read - len(all_atoms)} based"
        " on forces criteria"
    )

    filtered_atoms = []
    if strategy.lower() == "qr":
        from scipy.linalg import qr

        logger.warning("qr strategy assumes the last read structure is the equilibrium geometry")
        if np.linalg.norm(all_atoms[-1].arrays["REF_forces"], axis=1).max() > 0.02:
            warnings.warn("large forces found in last structure", stacklevel=0)

        sqrt_mass = np.repeat(np.sqrt(all_atoms[-1].get_masses()), 3)

        dq = np.zeros((sqrt_mass.shape[0], len(all_atoms) - 1), dtype=np.float64)
        for atoms_i, atoms in enumerate(tqdm(all_atoms[:-1], desc="[*] computing dq")):
            dx = get_disp_vect(atoms, all_atoms[-1])
            dq[:, atoms_i] = sqrt_mass * dx

        logger.info("computing pivoted QR factorization")
        _, R, P = qr(dq, mode="economic", pivoting=True)

        rdiag = np.diag(R)
        rank = np.sum(np.abs(rdiag) > rtol * rdiag.max())

        filtered_atoms = [all_atoms[-1]] + [all_atoms[atoms_i] for atoms_i in np.sort(P[:rank])]
    elif strategy.lower() == "dx":
        filtered_atoms.append(all_atoms[0])
        for atoms in tqdm(all_atoms[1:], desc="[*] dx filtering"):
            dx = get_disp_vect(filtered_atoms[-1], atoms)
            if np.linalg.norm(dx) > dx_tol:
                filtered_atoms.append(atoms)
    else:
        filtered_atoms = all_atoms

    logger.info(f"filtered {len(all_atoms) - len(filtered_atoms)} configurations from dataset")

    with open(output_file, "a") as f:
        for atoms in tqdm(filtered_atoms, desc="[*] writing"):
            ase.io.write(f, atoms, format="extxyz")


def get_force_opt_modes(
    n: int,
    omega2: np.ndarray,
    U: np.ndarray,
    sqrt_mass: np.ndarray,
    F: float = 0.5,
    tol: float = 1e-6,
    seed: int = 897689932,
    start_with_min_spread: bool = False,
    save_plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin phonons by energy and find a vector in the subspace that minimizes the spread in forces.

    Args:
        n (int): number of modes to select
        omega2 (np.ndarray): frequencies squared of the modes (directly from np.linalg.eigh)
        U (np.ndarray): matrix where eigenvectors of modes are cols (directly from np.linalg.eigh)
        sqrt_mass (np.ndarray): sqrt of the vector of atomic masses
        F (float): target forces to optimize amplitudes for
        tol (float): convergence tolerance for scipy minimize call
        seed (int): seed value for random number generator
        start_with_min_spread (bool): determines if mode with smallest force spread is used as the
            starting point for optimization. Uses a random vector otherwise.
        save_plot (bool): save a plot for analyzing resulting modes

    Returns:
        modes (np.ndarray): generated modes as columns of the matrix
        mode_dqs (np.ndarray): optimized displacement amplitudes following above criteria
    """
    inv_sqrt_mass = 1 / sqrt_mass

    # acoustic phonon filtering
    omega2[omega2 < (0.0005 / omega2eV) ** 2] = 0.0

    opt_path, opt_info = np.einsum_path("i,ij,j,j->i", sqrt_mass, U, omega2, np.ones(U.shape[0]))
    logger.debug(opt_info)

    def loss(x, ind=None):
        if ind is not None:
            tx = np.zeros(U.shape[0])
            tx[ind] = x
        else:
            tx = x

        Fx = np.einsum("i,ij,j,j->i", sqrt_mass, U, omega2, tx, optimize=opt_path)
        Fx = np.linalg.norm(Fx.reshape((-1, 3)), axis=1)
        return np.mean((Fx - Fx.mean()) ** 2)

    def constr_f(x):
        return np.sum(x**2)

    def constr_J(x):
        return 2 * x

    def constr_H(x, v):
        return np.diag(2 * v[0] * np.ones(x.shape[0]))

    def log_callback(intermediate_result):
        ir = intermediate_result
        logger.debug(f"{ir.nit:04d} | {ir.fun:.06e} | {ir.optimality:.06e}")

    from scipy.optimize import NonlinearConstraint, minimize

    norm_constr = NonlinearConstraint(constr_f, 1, 1, jac=constr_J, hess=constr_H)

    rng = np.random.default_rng(seed)

    modes = np.zeros((U.shape[0], n), dtype=np.float64)
    mode_dqs = np.zeros(n, dtype=np.float64)
    freqs = np.empty(n)

    subspace_inds = np.array_split(np.arange(3, U.shape[0]), n)
    for imode, ind in enumerate(tqdm(subspace_inds, desc="[*] optimizing modes")):
        subs_spread = np.empty(ind.shape[0])
        for i, ind_i in enumerate(ind):
            Fx = np.linalg.norm((sqrt_mass * omega2[ind_i] * U[:, ind_i]).reshape((-1, 3)), axis=1)
            subs_spread[i] = np.mean((Fx - Fx.mean()) ** 2)

        if start_with_min_spread:
            imin = np.argmin(subs_spread)
            logger.info(
                f"starting from mode {ind[imin]} "
                f"with frequency {1000 * omega2eV * np.sqrt(omega2[ind[imin]]):.02f} meV "
                f"and spread {subs_spread[imin]:.06e}"
            )
            x0 = np.zeros(ind.shape[0])
            x0[imin] = 1.0
        else:
            x0 = rng.random(ind.shape[0]) - 0.5
            x0 /= np.linalg.norm(x0)

        logger.debug(" nit |     loss     |  optimality")
        res = minimize(
            lambda x: loss(x, ind=ind),  # noqa: B023
            x0,
            tol=tol,
            method="trust-constr",
            constraints=[norm_constr],
            callback=log_callback,
        )

        if not res.success:
            logger.warning(f"optimization failed - {res.message}")

        logger.debug(f"final spread {res.fun}, subspace spread {subs_spread}")
        if not np.all(res.fun < subs_spread):
            logger.warning("optimization failed to find a vector with smaller spread")

        modes[ind, imode] = res.x
        freqs[imode] = omega2eV * np.sqrt(modes[:, imode] @ np.diag(omega2) @ modes[:, imode])
        logger.debug(f"found mode with frequency {1000 * freqs[imode]:.02f} meV")

        Fx = np.linalg.norm(
            np.einsum("i,ij,j,j->i", sqrt_mass, U, omega2, modes[:, imode]).reshape((-1, 3)),
            axis=1,
        )
        mode_dqs[imode] = F / Fx.max()

        dx = np.linalg.norm(
            (inv_sqrt_mass * mode_dqs[imode] * (U @ modes[:, imode])).reshape((-1, 3)), axis=1
        )
        if dx.max() > 0.05:
            new_dq = (0.05 / dx.max()) * mode_dqs[imode]
            logger.warning(
                f"max displacement too large ({dx.max()} > 0.05), "
                f"resetting dq {mode_dqs[imode]} -> {new_dq}"
            )
            mode_dqs[imode] = new_dq
        elif dx.max() < 0.005:
            new_dq = (0.005 / dx.max()) * mode_dqs[imode]
            logger.warning(
                f"max displacement too small ({dx.max()} < 0.005), "
                f"resetting dq {mode_dqs[imode]} -> {new_dq}"
            )
            mode_dqs[imode] = new_dq

    if save_plot:
        w = np.linspace(0, 0.1, 1000)
        dos = np.zeros_like(w)
        for i in range(U.shape[0]):
            dos += gaussian(w - omega2eV * np.sqrt(omega2[i]), 0.001)
        dos /= U.shape[0]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(w, dos, color="k")
        ax.fill_between(w, dos, color="k", alpha=0.2, lw=0)
        for imode in range(n):
            pdos = np.zeros_like(w)
            for i in range(U.shape[0]):
                pdos += modes[i, imode] ** 2 * gaussian(w - omega2eV * np.sqrt(omega2[i]), 0.001)
            p = ax.plot(w, pdos / n, lw=1)
            ax.axvline(x=freqs[imode], color=p[0].get_color(), alpha=0.2)
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Density of States")
        plt.savefig("pdos.png", dpi=600, bbox_inches="tight")

    return U @ modes, mode_dqs


def get_random_phonons(
    n: int,
    omega2: np.ndarray,
    U: np.ndarray,
    sqrt_mass: np.ndarray,
    F: float = 0.5,
    seed: int = 897689932,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin phonons by energy and select one randomly from each bin.

    The amplitude of each phonon is chosen to produce a max force per atom as close to F as
    possible. The max displacement on a given atom is kept within a reasonable range (0.005, 0.05).

    Args:
        n (int): number of modes to select
        omega2 (np.ndarray): frequencies squared of the modes (directly from np.linalg.eigh)
        U (np.ndarray): matrix where eigenvectors of modes are cols (directly from np.linalg.eigh)
        sqrt_mass (np.ndarray): sqrt of the vector of atomic masses
        F (float): target forces to optimize amplitudes for
        seed (int): seed value for random number generator

    Returns:
        modes (np.ndarray): generated modes as columns of the matrix
        mode_dqs (np.ndarray): optimized displacement amplitudes following above criteria
    """
    inv_sqrt_mass = 1 / sqrt_mass

    # acoustic phonon filtering
    omega2[omega2 < (0.0005 / omega2eV) ** 2] = 0.0

    modes = np.zeros((U.shape[0], n), dtype=np.float64)
    mode_dqs = np.empty(n, dtype=np.float64)

    rng = np.random.default_rng(seed)

    # starting from 3 to skip acoustic phonons
    subspace_inds = np.array_split(np.arange(3, U.shape[0]), n)
    for i, inds in enumerate(tqdm(subspace_inds, desc="[*] selecting random modes")):
        imode = inds[rng.integers(inds.shape[0])]
        modes[:, i] = U[:, imode]

        Fx = np.linalg.norm((sqrt_mass * omega2[imode] * U[:, imode]).reshape((-1, 3)), axis=1)
        mode_dqs[i] = F / Fx.max()

        dx = np.linalg.norm((mode_dqs[i] * inv_sqrt_mass * U[:, imode]).reshape((-1, 3)), axis=1)
        if dx.max() > 0.05:
            new_dq = (0.05 / dx.max()) * mode_dqs[i]
            logger.warning(
                f"max displacement too large ({dx.max()} > 0.05), "
                f"resetting dq {mode_dqs[i]} -> {new_dq}"
            )
            mode_dqs[i] = new_dq
        elif dx.max() < 0.005:
            new_dq = (0.005 / dx.max()) * mode_dqs[i]
            logger.warning(
                f"max displacement too small ({dx.max()} < 0.005), "
                f"resetting dq {mode_dqs[i]} -> {new_dq}"
            )
            mode_dqs[i] = new_dq
    return modes, mode_dqs


def gen_confs(
    struct_path: Path,
    num_conf: int,
    strategy: str = "rand",
    output_dir: Path = Path("./confs"),
    accepting_mode: Path | None = None,
    dynmat_file: Path | None = None,
    orthogonalize: bool = False,
    default_max_dx: float = 0.015,
    start_with_min_spread: bool = False,
    opt_tol: float = 1e-6,
    seed: int = 897689932,
) -> None:
    """Generate additional configurations to enhance fine-tuning dataset.

    Args:
        struct_path (Path): path to file containing structure that will be displaced
        num_conf (int): total number of additional configurations to generate
        strategy (str): strategy used to generate the additional configurations. Available options
            are 'rand', 'phon_rand', and 'phon_opt'.
        output_dir (Path): output directory where the new configurations will be written to
        accepting_mode (Path): path to file containing the structure that defines the accepting
            mode. For example, if struct_path refers to the ground-state equilibrium geometry, then
            accepting_mode should refer to the excited-state equilibrium geometry and vice versa.
        dynmat_file (Path): path to the .npz file containing the dynamical matrix presumably
            calculated using the "compute-dynmat" function.
        orthogonalize (bool): perform Gram-Schmidt orthogonalization at the last step
        default_max_dx (float): default value for max displacement of a given atom
        start_with_min_spread (bool): determines if mode with smallest force spread is used as the
            starting point for optimization in phon_opt strategy. Uses a random vector otherwise.
        opt_tol (float): convergence tolerance for the call to scipy minimize in the phon_opt strat
        seed (int): seed value for random number generator
    """
    if strategy.lower() not in ("rand", "phon_rand", "phon_opt"):
        raise ValueError("invalid strategy choice")

    if output_dir.exists():
        raise ValueError("output directory already exists")

    struct: Atoms = ase.io.read(struct_path)  # type: ignore[assignment]
    sqrt_mass = np.repeat(np.sqrt(struct.get_masses()), 3)

    # makes written poscars ugly and adds unnecessary data
    if struct.has("momenta"):
        del struct.arrays["momenta"]

    imode = 0
    modes = np.zeros((3 * len(struct), num_conf), dtype=np.float64)
    mode_dqs = np.zeros(num_conf, dtype=np.float64)

    if accepting_mode is not None:
        am_struct: Atoms = ase.io.read(accepting_mode)  # type: ignore[assignment]
        dx = get_disp_vect(struct, am_struct)
        dq = sqrt_mass * dx
        logger.info(f"accepting mode dQ={np.linalg.norm(dq)} amu^0.5 Å")

        modes[:, 0] = dq / np.linalg.norm(dq)
        mode_dqs[0] = (
            default_max_dx
            / np.linalg.norm((modes[:, 0] / sqrt_mass).reshape((-1, 3)), axis=1).max()
        )

        imode += 1

    if strategy.lower()[:4] == "phon":
        logger.info("working in phonon basis")

        if dynmat_file is None:
            raise ValueError("dynamical matrix file is needed to compute phonons")

        logger.info(f"reading dynamical matrix from {dynmat_file}")
        data = np.load(dynmat_file)
        H = data["H"]

        if not np.allclose(sqrt_mass, data["sqrt_mass"], rtol=1e-4):
            logger.debug(sqrt_mass, data["sqrt_mass"])
            raise ValueError("sqrt_mass from struct is not compatible with dynamical matrix file")

        # diagonalize dynamical matrix
        omega2, U = np.linalg.eigh(H)

        if strategy.lower() == "phon_opt":
            logger.info("generating force-optimized phonons")
            modes[:, imode:], mode_dqs[imode:] = get_force_opt_modes(
                num_conf - imode,
                omega2,
                U,
                sqrt_mass,
                tol=opt_tol,
                start_with_min_spread=start_with_min_spread,
                seed=seed,
            )
        else:
            logger.info("selecting phonons randomly")
            modes[:, imode:], mode_dqs[imode:] = get_random_phonons(
                num_conf - imode,
                omega2,
                U,
                sqrt_mass,
                seed=seed,
            )
    else:
        logger.info("generating random modes")

        from scipy.linalg import qr

        rng = np.random.default_rng(seed)
        modes[:, imode:], _, _ = qr(
            rng.random((modes.shape[0], num_conf - imode)) - 0.5, mode="economic", pivoting=True
        )
        for i in range(imode, num_conf):
            mode_dqs[i] = (
                default_max_dx
                / np.linalg.norm((modes[:, i] / sqrt_mass).reshape((-1, 3)), axis=1).max()
            )

    if orthogonalize:
        from scipy.linalg import qr

        logger.info("performing final Gram-Schmidt orthogonalization of all modes")
        modes, _ = qr(modes, mode="economic")

        logger.info("recomputing displacement amplitude after orthogonalization")
        for i in range(num_conf):
            mode_dqs[i] = (
                default_max_dx
                / np.linalg.norm((modes[:, i] / sqrt_mass).reshape((-1, 3)), axis=1).max()
            )

    # write to output directory
    for imode in tqdm(range(num_conf), desc="[*] writing structures"):
        dx = mode_dqs[imode] * modes[:, imode] / sqrt_mass
        logger.debug(
            f"mode {imode} - max displacement  "
            f"{np.linalg.norm(dx.reshape((-1, 3)), axis=1).max():.06f} Å  "
            f"{np.abs(dx).max():.06f} Å"
        )

        atoms = struct.copy()
        atoms.positions += dx.reshape((-1, 3))

        fname = output_dir / f"{imode}/POSCAR"
        fname.parent.mkdir(parents=True)
        ase.io.write(fname, atoms, format="vasp", direct=True)

    # store invoking command
    with open(output_dir / "cmd.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")


def reestimate_e0s_linear_system(
    calculator: MACECalculator,
    database_atoms: list[Atoms],
    elements: list | None = None,
    initial_e0s: dict | None = None,
) -> dict:
    """Estimate atomic reference energies (E0s) by solving a linear system.

    Notes:
        Slightly adapted from code by Noam Bernstein based on private communications
        with Ilyes Batatia and Joe Hart.

        This functionality will eventually be removed once merged into MACE.

    Args:
        calculator (MACECalculator): Calculator object for the MACE model.
        database_atoms (list): List of ase Atoms objects with energy and atomic_numbers.
        elements (list): List of element atomic numbers to consider, default to set present in
            database_atoms.
        initial_e0s (dict): Dictionary mapping element atomic numbers to E0 values, default to
            values returned by foundation_model for isolated atom configs>

    Returns:
        Dictionary with re-estimated E0 values for each element
    """
    # filter configs without energy
    database_atoms = [
        atoms for atoms in database_atoms if atoms.info.get("REF_energy") is not None
    ]

    if len(database_atoms) == 0:
        raise ValueError("database does not contain REF_energy tag")

    if not elements:
        elements = np.unique([Z for atoms in database_atoms for Z in atoms.numbers]).tolist()

    if not initial_e0s:
        initial_e0s = {Z: 0.0 for Z in elements}
        try:
            for Z in elements:
                for i in range(len(calculator.models)):
                    z_ind = calculator.z_table.z_to_index(Z)
                    initial_e0s[Z] += float(
                        calculator.models[i].atomic_energies_fn.atomic_energies[z_ind]
                    )
        except Exception as e:
            logger.warning(f"unexpected exception in getting initial E0s: {e}")
            logger.warning("falling back to explicit isolated atom calculations")

            from ase.atoms import Atoms

            for Z in elements:
                calculator.calculate(
                    atoms=Atoms(numbers=[Z], cell=[20] * 3, pbc=[True] * 3), properties=["energy"]
                )
                initial_e0s[Z] = calculator.results.get("energy")
        logger.info(f"using initial E0s: {initial_e0s}")

    # A matrix: each row contains atom counts for each element
    # b vector: each entry is the prediction error for a configuration
    A = np.zeros((len(database_atoms), len(elements)))
    b = np.zeros(len(database_atoms))

    logger.info(
        f"solving linear system with {len(database_atoms)} equations and {len(elements)} unknowns"
    )

    # - A[i,j] is the count of element j in configuration i
    # - b[i] is the error (true - predicted) for configuration i
    # - x[j] will be the energy correction for element j
    for i, atoms in enumerate(tqdm(database_atoms, desc="[*] foundation model predictions")):
        calculator.calculate(atoms=atoms.copy(), properties=["energy"])
        b[i] = atoms.info["REF_energy"] - calculator.results.get("energy")

        # atom counts for each element
        for j, element in enumerate(elements):
            A[i, j] = np.sum(atoms.get_atomic_numbers() == element)

    # solve with least squares
    try:
        corrections, _, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as e:
        logger.warning(f"error using lstsq to solve the linear system: {e}")
        logger.warning("falling back to foundation model E0s")
        return initial_e0s.copy()

    if np.linalg.norm(corrections) > 1e6:
        logger.critical(
            f"abnormally large corrections found, rank determination may have failed: {s}"
        )

    new_e0s = {}
    for i, element in enumerate(elements):
        new_e0s[element] = initial_e0s[element] + corrections[i]
        logger.debug(
            f"element {element}: foundation E0 = {initial_e0s[element]:.4f}, "
            f"correction = {corrections[i]:.4f}, new E0 = {new_e0s[element]:.4f}"
        )

    # statistics about the fit
    b_after = b - A @ corrections
    mse_before, mse_after = np.mean(b**2), np.mean(b_after**2)
    improvement = (1 - mse_after / mse_before) * 100

    logger.debug(f"mean squared error before correction: {mse_before:.4f} eV²")
    logger.debug(f"mean squared error after correction: {mse_after:.4f} eV²")
    logger.debug(f"improvement: {improvement:.1f}%")

    if rank < len(elements):
        logger.warning(f"system is rank deficient (rank {rank}/{len(elements)})")
        logger.warning(
            "some elements may be linearly dependent or not well represented in the dataset."
        )

    return new_e0s


def gen_ft_config(
    out: Path | str = "./config.default",
    estimate_e0s: bool = False,
    device: str = "cuda",
    name: str = "fine-tuned",
    mace_model: str = "medium-omat-0",
    database: Path | str = "./database.extxyz",
    head: str = "default",
) -> None:
    """Generate a configuration file for mace_run_train.

    Args:
        out (Path): path where the mace_run_train configuration file is written to.
        estimate_e0s (bool): estimate the E0s for training of the foundation model.
        device (str): device string passed to MACE to determine where calculation is performed.
        name (str): name of the model.
        mace_model (str): pre-trained MACE model that is used, can be a local path
        database (Path): path to the training dataset file (likely generated with :func:`collect`)
        head (str): which head from the model to use for prediction
    """
    e0s: str | dict

    if Path(out).exists():
        raise ValueError(f"Output path {out} already exists!")

    if not estimate_e0s:
        logger.warning(f'Please update the "E0s" tag in {out} prior to running mace_run_train.')
        e0s = "TO_BE_REPLACED"
    else:
        from mace.calculators import mace_mp

        calc = mace_mp(
            model=mace_model,
            dispersion=False,
            default_dtype="float64",
            device=device,
            enable_cueq=(device == "cuda" and find_spec("cuequivariance") is not None),
            head=head,
        )

        logger.info("running E0 estimation")
        db_atoms: list[Atoms] = ase.io.read(database, ":")  # type: ignore[assignment]
        e0s = reestimate_e0s_linear_system(calc, db_atoms)

    with open(out, "w") as f:
        f.write(
            "\n".join(
                [
                    f"name: {name}",
                    f"foundation_model: {mace_model}",
                    f"train_file: {database}",
                    f"valid_file: {database}",
                    f'E0s: "{e0s}"',
                    "multiheads_finetuning: false",
                    "energy_weight: 1",
                    "forces_weight: 10",
                    "stress_weight: 0",
                    "ema: true",
                    "ema_decay: 0.999",
                    "lr: 0.001",
                    "max_num_epochs: 500",
                    "default_dtype: float64",
                    "batch_size: 1",
                    "valid_batch_size: 1",
                ]
            )
            + "\n"
        )


def parse_force_constants_file(fname: Path | str) -> np.ndarray:
    """Parse a force constants file from phonopy."""
    with open(fname) as f:
        H = np.zeros([int(x) for x in next(f).split()] + [3, 3], dtype=np.float64)

        for _ in range(H.shape[0] * H.shape[1]):
            i, j = [int(x) - 1 for x in next(f).split()]

            H[i, j, :, :] = np.fromstring(
                " ".join([next(f).strip() for _ in range(3)]), sep=" "
            ).reshape((3, 3))

    H = H.swapaxes(1, 2).reshape((3 * H.shape[0], 3 * H.shape[0]))
    return H


def convert_from_phonopy(
    fname: Path | str, atoms_file: Path | str, save_file: Path | str = "dynmat.npz"
) -> None:
    """Convert phonopy FORCE_CONSTANTS to dynmat.npz.

    FORCE_CONSTANTS is written by phonopy when specifying the "--writefc" tag.

    Args:
        fname (Path): path to FORCE_CONSTANTS file.
        atoms_file (Path): path to file containing the equilibrium structure that was used to
            evaluate the force constants. (Only needed to extract sqrt masses.)
        save_file (Path): path where dynamical matrix is saved to (should end in .npz)
    """
    logger.info(f"reading force constants from {fname}")
    H = parse_force_constants_file(fname)

    atoms: Atoms = ase.io.read(atoms_file)  # type: ignore[assignment]
    sqrt_mass = np.repeat(np.sqrt(atoms.get_masses()), 3)
    inv_sqrt_mass = 1 / sqrt_mass

    H = np.einsum("i,ij,j->ij", inv_sqrt_mass, H, inv_sqrt_mass)

    logger.info(f"saving dynamical matrix to {save_file}")
    np.savez_compressed(save_file, H=H, sqrt_mass=sqrt_mass, cmd=" ".join(sys.argv))


def compute_dynmat(
    input_struct: Path,
    save_file: Path = Path("./dynmat.npz"),
    mace_model: str = "medium-omat-0",
    device: str = "cuda",
    head: str = "default",
    relax_struct: bool = True,
    analytical_hessian: bool = True,
    relax_algo: str = "LBFGSLineSearch",
    fmax: float = 0.001,
) -> None:
    """Calculate the dynamical matrix using MACE.

    Args:
        input_struct (Path): structure about which to compute the dynamical matrix
        save_file (Path): path where dynamical matrix is saved to (should end in .npz)
        mace_model (str): pre-trained MACE model that is used, can be a local path
        device (str): device string passed to MACE to determine where calculation is performed
        head (str): which head from the model to use for prediction
        relax_struct (bool): determines if an atomic relaxation is performed prior to computing the
            Hessian matrix. This is recommended if the model does not predict the same equilibrium
            structure as your explicit DFT calculation, which is generally the case unless good
            fine tuning has been performed.
        analytical_hessian (bool): determines if the Hessian is computed analytically or
            numerically using finite differences
        relax_algo (str): name of algorithm from ase.optimize that is used for atomic relaxation.
        fmax (float): force convergence criteria for atomic relaxation in eV/Ä
    """
    from mace.calculators import mace_mp

    atoms: Atoms = ase.io.read(input_struct)  # type: ignore[assignment] # noqa: F823
    atoms.calc = mace_mp(
        model=mace_model,
        dispersion=False,
        default_dtype="float64",
        device=device,
        enable_cueq=(device == "cuda" and find_spec("cuequivariance") is not None),
        head=head,
    )

    if relax_struct:
        import ase.optimize as ase_optim

        optim = getattr(ase_optim, relax_algo)
        optim(atoms).run(fmax=fmax)

    if np.linalg.norm(atoms.get_forces(), axis=1).max() > 0.02:
        warnings.warn("large forces found", stacklevel=0)

    if analytical_hessian:
        H = atoms.calc.get_hessian(atoms).reshape((3 * len(atoms), 3 * len(atoms)))
    else:
        from ase.atoms import Atoms
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms

        phonopy_atoms = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions(),
        )
        phonopy = Phonopy(phonopy_atoms, supercell_matrix=np.eye(3), log_level=2)
        phonopy.generate_displacements(distance=0.02)

        forces = []
        for phonopy_atoms in tqdm(
            phonopy.supercells_with_displacements, desc="[*] computing forces"
        ):
            atoms_dx = Atoms(
                symbols=phonopy_atoms.symbols,
                scaled_positions=phonopy_atoms.scaled_positions,
                cell=phonopy_atoms.cell,
                pbc=True,
            )
            atoms.calc.calculate(atoms=atoms_dx, properties=["forces"])
            forces.append(atoms.calc.results["forces"])

        phonopy.forces = np.array(forces)
        phonopy.produce_force_constants()
        phonopy.symmetrize_force_constants()

        if phonopy.force_constants is None:
            raise RuntimeError("phonopy failed to produce force constants")

        H = phonopy.force_constants.swapaxes(1, 2).reshape((3 * len(atoms), 3 * len(atoms)))

    if not np.allclose(H, H.T):
        warnings.warn("Hessian matrix is not symmetric", stacklevel=0)

    sqrt_mass = np.repeat(np.sqrt(atoms.get_masses()), 3)
    inv_sqrt_mass = 1 / sqrt_mass
    H = np.einsum("i,ij,j->ij", inv_sqrt_mass, H, inv_sqrt_mass)
    np.savez_compressed(save_file, H=H, sqrt_mass=sqrt_mass, cmd=" ".join(sys.argv))


def compute_lineshape(
    ground: Path,
    excited: Path,
    dynmat_file: Path,
    emission: Annotated[bool, Parameter(name="--luminescence", negative="--absorption")] = True,
    dE: float | None = None,
    gamma_zpl: float = 0.001,
    sigma_zpl: float = 0.0,
    sigma_psb: tuple[float, float] = (0.005, 0.001),
    gamma_psb: tuple[float, float] | None = None,
    omega_mult: float = 5.0,
    norm: str = "area",
    T: Annotated[float, Parameter(name=["--T", "-T"])] = 0.0,
    plot: str | None = None,
) -> None:
    """Calculate the spectral density/function and lineshape for a given dynamical matrix.

    Args:
        ground (Path): path to structure containing the ground state equilibrium geometry.
        excited (Path): path to structure containing the excited state equilibrium geometry.
        dynmat_file (Path): path to dynamical matrix file produced by :func:`compute_dynmat` or by
            phonopy and converted with :func:`convert_from_phonopy`.
        emission (bool): write luminescence (True) or absorption (False) spectrum.
        dE (float): zero-phonon line energy in eV, inferred from ground/excited if not provided.
        gamma_zpl (float): Lorentzian broadening in the ZPL to capture homogeneous broadening.
        sigma_zpl (float): Gaussian broadening in the ZPL to capture inhomogeneous broadening.
        sigma_psb (float, float): Gaussian broadening used to broaden the partial Huang-Rhys
            factors. The broadening factor is linearly interpolated from sigma_psb[0] at zero
            frequency to sigma_psb[1] at the highest (non-LVM) frequency.
        gamma_psb (float, float): Turns on Lorentzian broadening of local vibrational modes
            identified by their inverse participation ratio. gamma_psb[0] is ipr_cut and
            gamma_psb[1] is gamma_lvm. See :class:`Broadening`.
        omega_mult (float): number of factors of maximum phonon frequency from ZPL to plot.
        norm (str): normalization of luminescence (area or max).
        T (float): Temperature in kelvin.
        plot (str): if provide, specifies the type of plot to be generated and save in the current
            working directory. Can be "subplot", "inset", "dos", "S", or "L". See
            :func:`plot_spec_funcs` for more info.
    """
    initial, final = (excited, ground) if emission else (ground, excited)

    ini_atoms: Atoms = ase.io.read(initial)  # type: ignore[assignment]
    fin_atoms: Atoms = ase.io.read(final)  # type:ignore[assignment]

    if dE is None:
        try:
            dE = np.abs(fin_atoms.get_potential_energy() - ini_atoms.get_potential_energy())
        except RuntimeError:
            logger.critical("input files do not contain total energy, so dE cannot be determined")
            raise
        logger.info(f"energy difference not provided, found dE = {dE} from input structures")

    sqrt_mass = np.repeat(np.sqrt(fin_atoms.get_masses()), 3)

    dx = get_disp_vect(fin_atoms, ini_atoms)
    dq = sqrt_mass * dx
    logger.info(f"dQ={np.linalg.norm(dq):.06f} amu^{{1/2}} Å")

    logger.info("reading dynmat matrix")
    data = np.load(dynmat_file)
    H = data["H"]

    # probably not needed, but doesn't hurt
    if not np.allclose(sqrt_mass, data["sqrt_mass"]):
        warnings.warn(
            "sqrt_mass in dynmat file is not compatible with ground/excited conf",
            stacklevel=0,
        )

    logger.info("diagonalizing")
    omega, U = get_phonons(H)
    dq_k = U.T @ dq

    logger.info(f"total Huang-Rhys factor Stot={get_Stot(dq_k, omega):.04f}")

    if gamma_psb is not None:
        logger.info("computing inverse participation ratios")
        ipr, ipr_cut, gamma_lvm = get_ipr(U), gamma_psb[0], gamma_psb[1]
    else:
        ipr, ipr_cut, gamma_lvm = None, None, None

    logger.info("computing spectral functions")
    w, dos, S, A = get_phonon_spec_func(
        dq_k,
        omega,
        sigma_psb=sigma_psb,
        gamma_zpl=gamma_zpl,
        sigma_zpl=sigma_zpl,
        gamma_lvm=gamma_lvm,
        ipr=ipr,
        ipr_cut=ipr_cut,
        T=T,
    )
    tw, L = convert_A_to_L(w, A, dE, emission=emission, norm=norm)

    logger.info("saving results to .txt files")
    np.savetxt("spec_funcs.txt", np.array((w, S, A)).T)
    np.savetxt("lineshape.txt", np.array((tw, L)).T)

    if plot is not None:
        import matplotlib.pyplot as plt

        plot_spec_funcs(
            (w, dos, S, tw, L),
            None,
            dE,
            emission=emission,
            omega_mult=omega_mult,
            omega_max=(omega2eV * omega.max()),
            plot_type=plot,
        )
        plt.savefig("lineshape.png", dpi=600, bbox_inches="tight")


def analyze_dynmat(dynmat: Path, structure: Path) -> None:
    """Produce analysis plots of the dynamical matrix."""
    H = np.load(dynmat)["H"]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    p = ax.imshow(np.log(np.abs(H)), vmin=-6)
    plt.colorbar(p)
    plt.savefig("H.png", dpi=600, bbox_inches="tight")

    atoms: Atoms = ase.io.read(structure)  # type: ignore[assignment]
    d = atoms.get_all_distances(mic=True)

    fig, ax = plt.subplots(figsize=(4, 3))
    for i in range(3):
        for j in range(3):
            ax.plot(d.flatten(), np.abs(H[i::3, j::3].flatten()), ".", ms=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\vert {\bf R}_I - {\bf R}_J \vert$ [${\rm \AA}$]")
    ax.set_ylabel(r"$\vert \Phi_{I\alpha,J\beta} \vert$ [eV/amu$^{1/2}$ ${\rm \AA}$]")
    plt.savefig("radial_H.png", dpi=600, bbox_inches="tight")
