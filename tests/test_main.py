from pathlib import Path

import pytest

from lineshape_tools.__main__ import app

test_files = Path(__file__).parent / "files"


@pytest.mark.parametrize(
    "command",
    [
        None,
        "collect",
        "compute-dynmat",
        "compute-lineshape",
        "convert-from-phonopy",
        "gen-confs",
        "gen-ft-config",
    ],
)
def test_help(capsys, command):
    if command is not None:
        app.meta([command, "--help"])
    else:
        app.meta([])

    captured = capsys.readouterr()

    assert "Usage: " in captured.out

    if command is not None:
        assert command in captured.out


def test_fullrun(cheap_calc):
    app.meta(["collect", "cheap.gnd.extxyz"])
    assert (cheap_calc / "database.extxyz").exists()

    app.meta(["compute-dynmat", "cheap.gnd.extxyz"])
    assert (cheap_calc / "dynmat.npz").exists()

    app.meta(
        [
            "compute-lineshape",
            "cheap.gnd.extxyz",
            "cheap.exc.extxyz",
            "dynmat.npz",
            "--plot",
            "subplots",
        ]
    )
    assert (cheap_calc / "lineshape.png").exists()
    assert (cheap_calc / "lineshape.txt").exists()
    assert (cheap_calc / "spec_funcs.txt").exists()


@pytest.mark.slow
@pytest.mark.parametrize("defect", ["C_N-GaN", "NV-diamond", "dimer-hBN"])
def test_examples_fullrun(test_dir, defect):
    app.meta(["collect", str(test_files / f"{defect}.gnd.extxyz.gz")])
    assert (test_dir / "database.extxyz").exists()

    app.meta(["compute-dynmat", str(test_files / f"{defect}.gnd.extxyz.gz")])
    assert (test_dir / "dynmat.npz").exists()

    app.meta(
        [
            "compute-lineshape",
            str(test_files / f"{defect}.gnd.extxyz.gz"),
            str(test_files / f"{defect}.exc.extxyz.gz"),
            "dynmat.npz",
            "--plot",
            "subplots",
        ]
        + (["--de", "1.06"] if defect == "C_N-GaN" else [])
    )
    assert (test_dir / "lineshape.png").exists()
    assert (test_dir / "lineshape.txt").exists()
    assert (test_dir / "spec_funcs.txt").exists()


@pytest.mark.parametrize("strategy", ["none", "qr", "dx"])
def test_collect(test_dir, strategy):
    app.meta(["collect", str(test_files / "C_N-GaN.exc.extxyz.gz"), "--strategy", f"{strategy}"])
    assert (test_dir / "database.extxyz").exists()


@pytest.mark.parametrize("strategy", ["rand", "phon_rand", "phon_opt"])
def test_gen_confs(cheap_calc, strategy):
    app.meta(["compute-dynmat", "cheap.gnd.extxyz"])
    app.meta(
        [
            "gen-confs",
            "cheap.gnd.extxyz",
            "2",
            "--strategy",
            f"{strategy}",
            "--dynmat-file",
            "dynmat.npz",
            "--opt-tol",
            "1e-1",
        ]
    )
    assert list((cheap_calc / "confs").glob("*/POSCAR"))
    assert (cheap_calc / "confs/cmd.txt").exists()


def test_gen_confs_am(cheap_calc):
    app.meta(["compute-dynmat", "cheap.gnd.extxyz"])
    app.meta(
        [
            "gen-confs",
            "cheap.gnd.extxyz",
            "2",
            "--accepting-mode",
            "cheap.exc.extxyz",
        ]
    )
    assert list((cheap_calc / "confs").glob("*/POSCAR"))
    assert (cheap_calc / "confs/cmd.txt").exists()


def test_gen_confs_orthogonalize(cheap_calc):
    app.meta(["compute-dynmat", "cheap.gnd.extxyz"])
    app.meta(
        [
            "gen-confs",
            "cheap.gnd.extxyz",
            "2",
            "--orthogonalize",
        ]
    )
    assert list((cheap_calc / "confs").glob("*/POSCAR"))
    assert (cheap_calc / "confs/cmd.txt").exists()


def test_gen_ft_config(cheap_calc):
    app.meta(["gen-ft-config"])
    assert (cheap_calc / "config.default").exists()

    app.meta(["collect", "cheap.gnd.extxyz"])
    app.meta(["gen-ft-config", "--out", "estimated_e0s.config.default", "--estimate-e0s"])
    assert (cheap_calc / "estimated_e0s.config.default").exists()


def test_convert_from_phonopy(cheap_calc):
    app.meta(["convert-from-phonopy", str(test_files / "FORCE_CONSTANTS"), "cheap.gnd.extxyz"])
    assert (cheap_calc / "dynmat.npz").exists()


def test_compute_dynmat_numerical_hessian(cheap_calc):
    app.meta(["compute-dynmat", "cheap.gnd.extxyz", "--no-analytical-hessian"])
    assert (cheap_calc / "dynmat.npz").exists()


@pytest.mark.parametrize("plot_type", ["subplots", "inset", "dos", "S", "L"])
def test_plots(cheap_calc, plot_type):
    app.meta(["compute-dynmat", "cheap.gnd.extxyz"])
    app.meta(
        [
            "compute-lineshape",
            "cheap.gnd.extxyz",
            "cheap.exc.extxyz",
            "dynmat.npz",
            "--plot",
            f"{plot_type}",
        ]
    )
    assert (cheap_calc / "lineshape.png").exists()
    assert (cheap_calc / "lineshape.txt").exists()
    assert (cheap_calc / "spec_funcs.txt").exists()


def test_finiteT_plot(cheap_calc):
    app.meta(["compute-dynmat", "cheap.gnd.extxyz"])
    app.meta(
        [
            "compute-lineshape",
            "cheap.gnd.extxyz",
            "cheap.exc.extxyz",
            "dynmat.npz",
            "--T",
            "300",
            "--plot",
            "L",
        ]
    )
    assert (cheap_calc / "lineshape.png").exists()
    assert (cheap_calc / "lineshape.txt").exists()
    assert (cheap_calc / "spec_funcs.txt").exists()


def test_lvm_plot(cheap_calc):
    app.meta(["compute-dynmat", str(test_files / "C_N-GaN.gnd.extxyz.gz")])
    app.meta(
        [
            "compute-lineshape",
            str(test_files / "C_N-GaN.gnd.extxyz.gz"),
            str(test_files / "C_N-GaN.exc.extxyz.gz"),
            "dynmat.npz",
            "--gamma-psb",
            "0.99",
            "0.001",
            "--plot",
            "L",
        ]
    )
    assert (cheap_calc / "lineshape.png").exists()
    assert (cheap_calc / "lineshape.txt").exists()
    assert (cheap_calc / "spec_funcs.txt").exists()


def test_analyze_dynmat(cheap_calc):
    app.meta(["compute-dynmat", "cheap.gnd.extxyz"])
    app.meta(["analyze-dynmat", "dynmat.npz", "cheap.gnd.extxyz"])
    assert (cheap_calc / "H.png").exists()
    assert (cheap_calc / "radial_H.png").exists()
