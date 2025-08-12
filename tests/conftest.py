import sys

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from lineshape_tools.constants import omega2eV


@pytest.fixture
def test_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def cheap_calc(test_dir):
    cell = [[0.0, 1.77, 1.77], [1.77, 0.0, 1.77], [1.77, 1.77, 0.0]]
    gnd = Atoms("C2", scaled_positions=[[0.0] * 3, [0.25] * 3], cell=cell, pbc=[True] * 3)
    exc = gnd.copy()
    exc.positions[0, 0] += 0.1

    gnd.calc = SinglePointCalculator(gnd, energy=0.0, forces=[[0.0] * 3] * 2)
    exc.calc = SinglePointCalculator(exc, energy=2.0, forces=[[0.0] * 3] * 2)

    ase.io.write("cheap.gnd.extxyz", gnd, format="extxyz")
    ase.io.write("cheap.exc.extxyz", exc, format="extxyz")

    return test_dir


class mace_mp:
    results = {}

    def __init__(self, *args, **kwargs):
        pass

    def calculate(self, atoms, *args, **kwargs):
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}

    def get_potential_energy(self, *args, **kwargs):
        return 0.0

    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3))

    def get_hessian(self, atoms):
        natoms = len(atoms)
        avg_mass = np.mean(atoms.get_masses())
        omega = np.linspace(0.0, 0.1, 3 * natoms) * np.sqrt(avg_mass) / omega2eV
        return np.diag(omega**2).reshape((3 * natoms, 3, natoms))


module = type(sys)("mace")
module.submodule = type(sys)("calculators")
module.submodule.mace_mp = mace_mp
sys.modules["mace"] = module
sys.modules["mace.calculators"] = module.submodule
