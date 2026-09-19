# ASSIGNMENT_MAP.md

This file maps the official PC2130 "Quantum projectile" questions to the analytical work, code, figures, and viva preparation.

## (a) Schrödinger equation

Use
    V(z) = m g z
because F_z = -dV/dz = -m g.

Then
    i hbar d_t psi =
    [-(hbar^2/2m) nabla^2 + m g z] psi.

No computation is required here; the code uses exactly this Hamiltonian.

## (b) Energy eigenvalue equation

Separate variables in
    [-(hbar^2/2m) nabla^2 + m g z] u_E = E u_E.

x and y are free directions, so use plane waves with transverse momenta k_x,k_y.
The z equation is the linear-potential Airy equation. Define

    E_perp = hbar^2 (k_x^2+k_y^2)/(2m)
    epsilon_z = E - E_perp
    alpha = (2 m^2 g / hbar^2)^(1/3)
    xi = alpha [z - epsilon_z/(m g)].

Then
    d^2 Z/dxi^2 - xi Z = 0.

For the full line, the acceptable generalized eigenfunction is proportional to Ai(xi);
Bi(xi) is excluded by its divergence as z -> +infinity. The spectrum is continuous.
A fully normalized continuum convention should be stated carefully in the report.

## (c) Initial Gaussian

Initial state:
    psi0 = (2 pi sigma0^2)^(-3/4)
           exp[-(x^2+y^2+(z-z0)^2)/(4 sigma0^2)]
           exp[i p0 x/hbar].

Normalization follows because |psi0|^2 factorizes into three normalized Gaussians.

Energy:
    <E> = p0^2/(2m)
          + 3 hbar^2/(8 m sigma0^2)
          + m g z0.

Code/tests:
- test_initial_state_matches_assignment
- test_normalization_numerically
- test_energy_formula
- test_energy_formula_against_numerical_quadrature

## (d) Time evolution

The exact accelerating-frame result is

    psi(x,y,z,t)
      = exp[-i(m g t z + m g^2 t^3/6)/hbar]
        psi_free(x,y,z + g t^2/2,t).

For the assigned Gaussian, the code evaluates this analytically through:
- free_psi()
- gravity_phase()
- psi()

Key width:
    sigma(t)=sigma0 sqrt[1+(hbar t/(2m sigma0^2))^2].

Tests:
- test_exact_accelerating_frame_transformation
- test_solution_satisfies_schrodinger_equation_and_phase_signs
- test_density_equals_abs_psi_squared

## (e) Ehrenfest theorem

The code/tests verify

    <x> = v0 t
    <y> = 0
    <z> = z0 - g t^2/2

and

    <px> = p0
    <py> = 0
    <pz> = -m g t.

These obey
    d<x>/dt = <px>/m
and
    d<pz>/dt = -m g = <-dV/dz>.

Tests:
- test_ehrenfest_trajectory_and_momentum
- test_phase_gradient_at_packet_center_matches_ehrenfest_momentum

## (f) Animations and physical interpretation

Required generated outputs:
- density_xz.gif: y=0 slice of |psi|^2
- real_psi_xz.gif: y=0 slice of Re psi
- trajectory_spreading.png: classical centre plus quantum width
- classical_limit.png: relative spreading for increasing mass

Interpretation:
- The centre follows the classical projectile trajectory exactly.
- The packet spreads because sigma(t) grows.
- Uniform gravity changes the centre motion and wavefunction phase but does not remove free Gaussian spreading.
- A single Gaussian under a linear potential has no interference fringes in |psi|^2.
- Increasing mass suppresses relative spreading, showing the classical-limit trend.
- Re psi oscillations represent phase structure and must not be interpreted as negative probability.

## Submission checklist

Report (single PDF, maximum target length: 4 pages as stated in the course introduction):
- answer all (a)-(f)
- show clear derivation steps and final results
- include relevant figures
- discuss physical meaning
- include an AI-use attribution written by the student

Code:
- submit the Python code separately to Canvas
- before submission run: python -m pytest -q
- generate final outputs with: python quantum_projectile.py --outdir outputs

Viva:
- be able to explain free_psi(), gravity_phase(), psi(), density(), and sigma_t()
- be able to change m, g, sigma0, z0, and v0
- be able to explain what changes physically after each parameter change
- be able to regenerate the animations
