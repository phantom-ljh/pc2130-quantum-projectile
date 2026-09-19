"""Exact Gaussian wavepacket evolution in the linear potential V(z) = m g z."""

from dataclasses import dataclass, replace
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

try:
    trapezoid = np.trapezoid
except AttributeError:  # NumPy 1.26
    trapezoid = np.trapz


@dataclass(frozen=True)
class Params:
    """Physical parameters in any one consistent system of units."""

    hbar: float = 1.0
    m: float = 2.0
    g: float = 1.0
    sigma0: float = 0.60
    z0: float = 3.0
    v0: float = 2.0

    def __post_init__(self):
        if self.hbar <= 0 or self.m <= 0 or self.sigma0 <= 0:
            raise ValueError("hbar, m, and sigma0 must be positive")

    @property
    def p0(self):
        return self.m * self.v0


def tau(t, p):
    """Dimensionless free-spreading time."""

    return p.hbar * np.asarray(t) / (2 * p.m * p.sigma0**2)


def sigma_t(t, p):
    """One-coordinate standard deviation of the probability density."""

    return p.sigma0 * np.sqrt(1 + tau(t, p) ** 2)


def classical_center(t, p):
    """Ehrenfest position expectation values (x, y, z)."""

    t = np.asarray(t)
    return p.v0 * t, np.zeros_like(t, dtype=float), p.z0 - 0.5 * p.g * t**2


def momentum_expectation(t, p):
    """Ehrenfest momentum expectation values (px, py, pz)."""

    t = np.asarray(t)
    return (
        np.full_like(t, p.p0, dtype=float),
        np.zeros_like(t, dtype=float),
        -p.m * p.g * t,
    )


def free_psi(x, y, z, t, p):
    """Free-particle evolution of the assigned initial Gaussian."""

    x, y, z, t = map(np.asarray, (x, y, z, t))
    q = 1 + 1j * tau(t, p)
    prefactor = (2 * np.pi * p.sigma0**2) ** (-3 / 4) * q ** (-3 / 2)
    displacement_squared = (x - p.v0 * t) ** 2 + y**2 + (z - p.z0) ** 2
    envelope = np.exp(-displacement_squared / (4 * p.sigma0**2 * q))
    translation_phase = np.exp(
        1j * (p.p0 * x - p.p0**2 * t / (2 * p.m)) / p.hbar
    )
    return prefactor * envelope * translation_phase


def gravity_phase(z, t, p):
    """Phase in the exact accelerating-frame transformation."""

    z, t = map(np.asarray, (z, t))
    action = p.m * p.g * t * z + p.m * p.g**2 * t**3 / 6
    return np.exp(-1j * action / p.hbar)


def psi(x, y, z, t, p):
    """Exact solution for H = -hbar^2 nabla^2/(2m) + m g z."""

    shifted_z = np.asarray(z) + 0.5 * p.g * np.asarray(t) ** 2
    return gravity_phase(z, t, p) * free_psi(x, y, shifted_z, t, p)


def density(x, y, z, t, p):
    """Closed-form |psi|^2, useful for plots and normalization checks."""

    s = sigma_t(t, p)
    xc, _, zc = classical_center(t, p)
    radius_squared = (
        (np.asarray(x) - xc) ** 2
        + np.asarray(y) ** 2
        + (np.asarray(z) - zc) ** 2
    )
    return (2 * np.pi * s**2) ** (-1.5) * np.exp(
        -radius_squared / (2 * s**2)
    )


def energy_expectation(p):
    """Conserved expectation value of kinetic plus potential energy."""

    translational = p.p0**2 / (2 * p.m)
    localization = 3 * p.hbar**2 / (8 * p.m * p.sigma0**2)
    gravitational = p.m * p.g * p.z0
    return translational + localization + gravitational


def numerical_normalization(t, p, n=101, nsigma=6.0):
    """Integrate |psi|^2 over a finite box centered on the packet."""

    s = float(sigma_t(t, p))
    xc, _, zc = [float(value) for value in classical_center(t, p)]
    x = np.linspace(xc - nsigma * s, xc + nsigma * s, n)
    y = np.linspace(-nsigma * s, nsigma * s, n)
    z = np.linspace(zc - nsigma * s, zc + nsigma * s, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    rho = density(X, Y, Z, t, p)
    integral_z = trapezoid(rho, z, axis=2)
    integral_yz = trapezoid(integral_z, y, axis=1)
    return float(trapezoid(integral_yz, x, axis=0))


def make_xz_grid(p, tmax, nx=300, nz=240):
    s = float(sigma_t(tmax, p))
    x0, _, z0 = classical_center(0, p)
    x1, _, z1 = classical_center(tmax, p)
    margin = 4.5 * s
    x = np.linspace(
        min(float(x0), float(x1)) - margin,
        max(float(x0), float(x1)) + margin,
        nx,
    )
    z = np.linspace(
        min(float(z0), float(z1)) - margin,
        max(float(z0), float(z1)) + margin,
        nz,
    )
    X, Z = np.meshgrid(x, z, indexing="xy")
    return x, z, X, Z


def _prepare_output(outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return outpath


def animate_density_xz(
    p, outpath="density_xz.gif", tmax=2.4, frames=100, fps=25, nx=300, nz=240
):
    outpath = _prepare_output(outpath)
    x, z, X, Z = make_xz_grid(p, tmax, nx=nx, nz=nz)
    times = np.linspace(0, tmax, frames)
    rho0 = density(X, 0, Z, 0, p)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(
        rho0,
        origin="lower",
        extent=[x.min(), x.max(), z.min(), z.max()],
        aspect="auto",
        vmin=0,
        vmax=float(rho0.max()),
    )
    fig.colorbar(im, ax=ax, label=r"$|\psi(x,0,z,t)|^2$")
    (dot,) = ax.plot([], [], marker="o", linestyle="None", label="packet centre")
    (trail,) = ax.plot([], [], linestyle="--", lw=1.2, label="classical trajectory")
    ax.set(xlabel="x", ylabel="z")
    ax.legend(loc="upper right")

    def update(i):
        t = times[i]
        im.set_data(density(X, 0, Z, t, p))
        xc, _, zc = classical_center(times[: i + 1], p)
        dot.set_data([xc[-1]], [zc[-1]])
        trail.set_data(xc, zc)
        ax.set_title(f"Probability density, t={t:.3f}")
        return im, dot, trail

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    animation.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return outpath


def animate_real_psi_xz(
    p, outpath="real_psi_xz.gif", tmax=2.4, frames=100, fps=25, nx=300, nz=240
):
    outpath = _prepare_output(outpath)
    x, z, X, Z = make_xz_grid(p, tmax, nx=nx, nz=nz)
    times = np.linspace(0, tmax, frames)
    initial_real_part = np.real(psi(X, 0, Z, 0, p))
    amplitude = float(np.max(np.abs(initial_real_part)))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(
        initial_real_part,
        origin="lower",
        extent=[x.min(), x.max(), z.min(), z.max()],
        aspect="auto",
        vmin=-amplitude,
        vmax=amplitude,
        cmap="RdBu_r",
    )
    fig.colorbar(im, ax=ax, label=r"$\mathrm{Re}\,\psi$")
    (dot,) = ax.plot([], [], marker="o", linestyle="None")
    ax.set(xlabel="x", ylabel="z")

    def update(i):
        t = times[i]
        im.set_data(np.real(psi(X, 0, Z, t, p)))
        xc, _, zc = classical_center(t, p)
        dot.set_data([float(xc)], [float(zc)])
        ax.set_title(f"Re(psi), t={t:.3f}")
        return im, dot

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    animation.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return outpath


def save_trajectory_spreading_figure(
    p, outpath="trajectory_spreading.png", tmax=2.4, dpi=180
):
    outpath = _prepare_output(outpath)
    t = np.linspace(0, tmax, 400)
    x, _, z = classical_center(t, p)
    s = sigma_t(t, p)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, z, label="packet centre")
    ax.fill_between(x, z - s, z + s, alpha=0.2, label=r"$\pm\sigma(t)$")
    ax.set(
        xlabel=r"$\langle x\rangle$",
        ylabel="z",
        title="Classical centre with quantum spreading",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def save_classical_limit_figure(
    p, outpath="classical_limit.png", tmax=2.4, dpi=180
):
    outpath = _prepare_output(outpath)
    t = np.linspace(0, tmax, 400)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for factor in (1, 5, 20):
        varied = replace(p, m=p.m * factor)
        ax.plot(t, sigma_t(t, varied) / varied.sigma0, label=f"m={factor:g} m0")
    ax.set(
        xlabel="t",
        ylabel=r"$\sigma(t)/\sigma_0$",
        title="Classical limit: larger mass suppresses spreading",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def run_all(p, outdir="outputs", tmax=2.4, frames=100, fps=25, smoke=False):
    """Generate every required animation and figure; return their paths."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    nx, nz, dpi = (80, 64, 90) if smoke else (300, 240, 180)
    if smoke:
        frames = min(frames, 6)

    print("Parameters:", p)
    print("<E> =", energy_expectation(p))
    print("Norm t=0 =", numerical_normalization(0, p, n=61))
    print("Norm t=tmax =", numerical_normalization(tmax, p, n=61))
    paths = [
        animate_density_xz(
            p, out / "density_xz.gif", tmax, frames, fps, nx=nx, nz=nz
        ),
        animate_real_psi_xz(
            p, out / "real_psi_xz.gif", tmax, frames, fps, nx=nx, nz=nz
        ),
        save_trajectory_spreading_figure(
            p, out / "trajectory_spreading.png", tmax, dpi=dpi
        ),
        save_classical_limit_figure(p, out / "classical_limit.png", tmax, dpi=dpi),
    ]
    for path in paths:
        print("created", path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--tmax", type=float, default=2.4)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--smoke", action="store_true", help="quick low-resolution run")
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--sigma0", type=float, default=0.6)
    parser.add_argument("--z0", type=float, default=3.0)
    parser.add_argument("--v0", type=float, default=2.0)
    parser.add_argument("--hbar", type=float, default=1.0)
    args = parser.parse_args()
    params = Params(args.hbar, args.m, args.g, args.sigma0, args.z0, args.v0)
    run_all(params, args.outdir, args.tmax, args.frames, args.fps, args.smoke)


if __name__ == "__main__":
    main()
