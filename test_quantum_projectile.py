import numpy as np

from quantum_projectile import (
    Params, psi, density, sigma_t, classical_center,
    momentum_expectation, energy_expectation, numerical_normalization,
)

def test_initial_state_matches_assignment():
    p = Params()
    x = np.array([-0.4, 0.2, 1.1])
    y = np.array([0.1, -0.2, 0.3])
    z = np.array([2.0, 3.0, 3.7])
    expected = (
        (2*np.pi*p.sigma0**2)**(-3/4)
        * np.exp(-(x**2+y**2+(z-p.z0)**2)/(4*p.sigma0**2))
        * np.exp(1j*p.p0*x/p.hbar)
    )
    assert np.allclose(psi(x,y,z,0.0,p), expected)

def test_density_equals_abs_psi_squared():
    p = Params()
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(2,30))
    z = p.z0 + rng.normal(size=30)
    t = 0.71
    assert np.allclose(np.abs(psi(x,y,z,t,p))**2, density(x,y,z,t,p), rtol=1e-11, atol=1e-12)

def test_normalization_numerically():
    p = Params()
    for t in (0.0, 0.5, 1.5):
        assert abs(numerical_normalization(t,p,n=61,nsigma=5.5)-1.0) < 3e-5

def test_ehrenfest_trajectory_and_momentum():
    p = Params()
    t = np.linspace(0,2.0,11)
    x,y,z = classical_center(t,p)
    px,py,pz = momentum_expectation(t,p)
    assert np.allclose(x,p.v0*t)
    assert np.allclose(y,0)
    assert np.allclose(z,p.z0-0.5*p.g*t**2)
    assert np.allclose(px,p.p0)
    assert np.allclose(py,0)
    assert np.allclose(pz,-p.m*p.g*t)

def test_spreading_starts_at_sigma0():
    p = Params()
    assert np.isclose(sigma_t(0.0,p),p.sigma0)

def test_energy_formula():
    p = Params()
    expected = p.p0**2/(2*p.m)+3*p.hbar**2/(8*p.m*p.sigma0**2)+p.m*p.g*p.z0
    assert np.isclose(energy_expectation(p),expected)
