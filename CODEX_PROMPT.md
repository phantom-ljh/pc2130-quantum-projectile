# Codex task: finish and verify the PC2130 Quantum Projectile code

Read AGENTS.md, README.md, quantum_projectile.py, and test_quantum_projectile.py.

Make the repository submission-ready for the computational and animation part of the assignment while keeping the code simple enough for a student to explain in a viva.

1. Verify the exact analytical solution for V(z)=m*g*z, including every gravity-phase sign.
2. Verify t=0 reproduces exactly the supplied 3D Gaussian.
3. Verify analytically and numerically that |psi|^2 is normalized.
4. Verify <E> = p0^2/(2m) + 3*hbar^2/(8m*sigma0^2) + m*g*z0.
5. Verify Ehrenfest:
   <x>=v0*t, <y>=0, <z>=z0-g*t^2/2,
   <px>=p0, <py>=0, <pz>=-m*g*t.
6. Run tests and fix errors rather than weakening tests.
7. Generate a short version of every animation/figure and confirm output files exist.
8. Optionally add a clean 3D probability-density isosurface using an optional dependency only.
9. Add VIVA_NOTES.md explaining wavefunction vs density animation, spreading, classical limit, and why a single Gaussian in a linear potential has no density interference fringes.
10. Keep report prose out of the codebase.

Finish by running python -m pytest -q and a short animation smoke test, then report files changed and caveats.
