# AGENTS.md — Codex instructions

You are working on a PC2130 Quantum Mechanics I individual assignment.
AI use is permitted as a learning aid, but the student must understand and explain all work.

## Source-of-truth physics

Hamiltonian:
    H = -(hbar^2/2m) nabla^2 + m g z

Initial state:
    psi0 = (2*pi*sigma0^2)^(-3/4)
           exp[-(x^2+y^2+(z-z0)^2)/(4 sigma0^2)]
           exp[i p0 x / hbar],
    p0 = m v0 > 0.

Exact gravity transformation:
    psi(x,y,z,t)
      = exp[-i(m g t z + m g^2 t^3/6)/hbar]
        * psi_free(x,y,z + g t^2/2,t).

Probability density:
    sigma(t) = sigma0*sqrt(1 + (hbar*t/(2*m*sigma0^2))^2)
    center = (v0*t, 0, z0 - g*t^2/2).

Energy expectation:
    <E> = p0^2/(2m) + 3*hbar^2/(8*m*sigma0^2) + m*g*z0.

Ehrenfest:
    <p> = (p0, 0, -m*g*t).

## Coding requirements

- Keep the exact analytical formula as the main implementation.
- Do not replace it with an opaque numerical PDE solver.
- Keep functions small and readable.
- Maintain tests for the initial state, normalization, |psi|^2, Ehrenfest centre/momentum, and energy formula.
- Generate at least:
  1) y=0 animation of |psi|^2,
  2) y=0 animation of Re(psi),
  3) static trajectory/spreading figure,
  4) classical-limit figure.
- Use editable parameters.
- Do not hide physical constants in magic numbers.
- Keep comments focused on physics interpretation.
- Run pytest before finishing.
