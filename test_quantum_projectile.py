from pathlib import Path

import numpy as np

import quantum_projectile as qp
from quantum_projectile import (
    Params,
    classical_center,
    density,
    energy_expectation,
    free_psi,
    gravity_phase,
    momentum_expectation,
    numerical_normalization,
    psi,
    run_all,
    sigma_t,
    trapezoid,
)


def test_initial_state_matches_assignment():
    p = Params()
    x = np.array([-0.4, 0.2, 1.1])
    y = np.array([0.1, -0.2, 0.3])
    z = np.array([2.0, 3.0, 3.7])
    expected = (
        (2 * np.pi * p.sigma0**2) ** (-3 / 4)
        * np.exp(-(x**2 + y**2 + (z - p.z0) ** 2) / (4 * p.sigma0**2))
        * np.exp(1j * p.p0 * x / p.hbar)
    )
    assert np.allclose(psi(x, y, z, 0.0, p), expected)


def test_exact_accelerating_frame_transformation():
    p = Params()
    x, y, z, t = 0.31, -0.27, 2.4, 0.63
    expected = gravity_phase(z, t, p) * free_psi(
        x, y, z + 0.5 * p.g * t**2, t, p
    )
    assert np.allclose(psi(x, y, z, t, p), expected)


def test_solution_satisfies_schrodinger_equation_and_phase_signs():
    """A wrong z phase, cubic phase, or coordinate shift fails this residual."""

    p = Params()
    x, y, z, t = 0.37, -0.21, 2.35, 0.58
    spatial_step = 2.0e-4
    time_step = 1.0e-5

    def value(x_value, y_value, z_value, t_value):
        return psi(x_value, y_value, z_value, t_value, p)

    dpsi_dt = (
        value(x, y, z, t + time_step) - value(x, y, z, t - time_step)
    ) / (2 * time_step)
    centre = value(x, y, z, t)
    laplacian = 0j
    for offset in ((spatial_step, 0, 0), (0, spatial_step, 0), (0, 0, spatial_step)):
        dx, dy, dz = offset
        laplacian += (
            value(x + dx, y + dy, z + dz, t)
            - 2 * centre
            + value(x - dx, y - dy, z - dz, t)
        ) / spatial_step**2

    lhs = 1j * p.hbar * dpsi_dt
    rhs = -(p.hbar**2 / (2 * p.m)) * laplacian + p.m * p.g * z * centre
    assert np.allclose(lhs, rhs, rtol=2e-6, atol=2e-7)


def test_density_equals_abs_psi_squared():
    p = Params()
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(2, 30))
    z = p.z0 + rng.normal(size=30)
    t = 0.71
    assert np.allclose(
        np.abs(psi(x, y, z, t, p)) ** 2,
        density(x, y, z, t, p),
        rtol=1e-11,
        atol=1e-12,
    )


def test_normalization_numerically():
    p = Params()
    for t in (0.0, 0.5, 1.5):
        assert abs(numerical_normalization(t, p, n=61, nsigma=5.5) - 1.0) < 3e-5


def test_ehrenfest_trajectory_and_momentum():
    p = Params()
    t = np.linspace(0, 2.0, 11)
    x, y, z = classical_center(t, p)
    px, py, pz = momentum_expectation(t, p)
    assert np.allclose(x, p.v0 * t)
    assert np.allclose(y, 0)
    assert np.allclose(z, p.z0 - 0.5 * p.g * t**2)
    assert np.allclose(px, p.p0)
    assert np.allclose(py, 0)
    assert np.allclose(pz, -p.m * p.g * t)


def test_phase_gradient_at_packet_center_matches_ehrenfest_momentum():
    p = Params()
    t = 0.73
    x, y, z = (float(value) for value in classical_center(t, p))
    step = 1.0e-5
    centre = psi(x, y, z, t, p)

    derivatives = (
        (psi(x + step, y, z, t, p) - psi(x - step, y, z, t, p)) / (2 * step),
        (psi(x, y + step, z, t, p) - psi(x, y - step, z, t, p)) / (2 * step),
        (psi(x, y, z + step, t, p) - psi(x, y, z - step, t, p)) / (2 * step),
    )
    local_momentum = np.array(
        [p.hbar * np.imag(derivative / centre) for derivative in derivatives]
    )
    expected = np.array([p.p0, 0.0, -p.m * p.g * t])
    assert np.allclose(local_momentum, expected, rtol=2e-9, atol=2e-9)


def test_spreading_starts_at_sigma0():
    p = Params()
    assert np.isclose(sigma_t(0.0, p), p.sigma0)


def test_energy_formula():
    p = Params()
    expected = (
        p.p0**2 / (2 * p.m)
        + 3 * p.hbar**2 / (8 * p.m * p.sigma0**2)
        + p.m * p.g * p.z0
    )
    assert np.isclose(energy_expectation(p), expected)


def test_energy_formula_against_numerical_quadrature():
    """Compute <T> from gradients and <V> from a separate 1D quadrature."""

    p = Params()
    coordinate = np.linspace(-6 * p.sigma0, 6 * p.sigma0, 4001)
    gaussian = (2 * np.pi * p.sigma0**2) ** (-1 / 4) * np.exp(
        -coordinate**2 / (4 * p.sigma0**2)
    )
    phi_x = gaussian * np.exp(1j * p.p0 * coordinate / p.hbar)
    derivative_x = np.gradient(phi_x, coordinate, edge_order=2)
    derivative_zero_momentum = np.gradient(gaussian, coordinate, edge_order=2)
    kinetic = p.hbar**2 / (2 * p.m) * (
        trapezoid(np.abs(derivative_x) ** 2, coordinate)
        + 2 * trapezoid(np.abs(derivative_zero_momentum) ** 2, coordinate)
    )
    z = coordinate + p.z0
    potential = p.m * p.g * trapezoid(z * np.abs(gaussian) ** 2, coordinate)
    assert np.isclose(kinetic + potential, energy_expectation(p), rtol=2e-5)


def test_all_required_outputs_smoke(tmp_path):
    paths = run_all(Params(), tmp_path, tmax=0.2, frames=4, fps=4, smoke=True)
    assert {path.name for path in paths} == {
        "density_xz.gif",
        "real_psi_xz.gif",
        "trajectory_spreading.png",
        "classical_limit.png",
    }
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths)


def test_main_accepts_explicit_arguments(monkeypatch, tmp_path):
    """PowerShell-style arguments can also be supplied directly from Python/Jupyter."""

    captured = {}

    def fake_run_all(p, outdir, tmax, frames, fps, smoke):
        captured.update(
            p=p,
            outdir=outdir,
            tmax=tmax,
            frames=frames,
            fps=fps,
            smoke=smoke,
        )
        return ["sentinel"]

    monkeypatch.setattr(qp, "run_all", fake_run_all)
    result = qp.main(
        [
            "--outdir",
            str(tmp_path),
            "--m",
            "7.0",
            "--g",
            "1.5",
            "--sigma0",
            "0.8",
            "--frames",
            "12",
            "--smoke",
        ]
    )

    assert result == ["sentinel"]
    assert captured["p"].m == 7.0
    assert captured["p"].g == 1.5
    assert captured["p"].sigma0 == 0.8
    assert captured["frames"] == 12
    assert captured["smoke"] is True


def test_main_ignores_jupyter_kernel_file_argument(monkeypatch):
    """The ipykernel '-f kernel.json' argument must not reach argparse."""

    captured = {}

    def fake_run_all(p, outdir, tmax, frames, fps, smoke):
        captured.update(p=p, outdir=outdir, smoke=smoke)
        return ["jupyter-ok"]

    monkeypatch.setattr(qp, "run_all", fake_run_all)
    monkeypatch.setattr(
        qp.sys,
        "argv",
        [
            "ipykernel_launcher.py",
            "-f",
            r"C:\\Users\\student\\AppData\\Roaming\\jupyter\\runtime\\kernel.json",
            "--m",
            "4.0",
            "--smoke",
        ],
    )
    monkeypatch.setitem(qp.sys.modules, "ipykernel", object())

    result = qp.main()

    assert result == ["jupyter-ok"]
    assert captured["p"].m == 4.0
    assert captured["smoke"] is True
