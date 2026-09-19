# PC2130 Quantum Projectile

Computational and animation code for the PC2130 assignment **Quantum
projectile**.

> AI was used as a learning/coding aid. The submitted report should be written
> in the student's own words, and every derivation, code block, parameter
> choice, figure, and physical interpretation should be explainable during the
> viva.

## Model

A particle of mass \(m\) moves in three dimensions under uniform gravity in
the \(-z\) direction,

\[
V(z)=mgz,\qquad
H=-\frac{\hbar^2}{2m}\nabla^2+mgz.
\]

The initial state is

\[
\psi(x,y,z,0)=
(2\pi\sigma_0^2)^{-3/4}
\exp\left[-\frac{x^2+y^2+(z-z_0)^2}{4\sigma_0^2}\right]
e^{ip_0x/\hbar},
\qquad p_0=mv_0.
\]

The exact time evolution used by the code is

\[
\psi(x,y,z,t)=
e^{-\frac{i}{\hbar}\left(mgtz+\frac16mg^2t^3\right)}
\psi_{\rm free}\left(x,y,z+\frac12gt^2,t\right).
\]

Hence

\[
\sigma(t)=\sigma_0\sqrt{1+
\left(\frac{\hbar t}{2m\sigma_0^2}\right)^2},
\]

and the packet centre obeys

\[
\langle x\rangle=v_0t,\qquad
\langle y\rangle=0,\qquad
\langle z\rangle=z_0-\frac12gt^2.
\]

The energy expectation is

\[
\langle E\rangle=
\frac{p_0^2}{2m}
+\frac{3\hbar^2}{8m\sigma_0^2}
+mgz_0.
\]

The sign derivation and viva explanations are in [VIVA_NOTES.md](VIVA_NOTES.md).

## Run and verify

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python quantum_projectile.py --outdir outputs
```

For a quick, low-resolution generation of every required output:

```bash
python quantum_projectile.py --outdir smoke_outputs --smoke
```

Both commands generate:

- `density_xz.gif`: the \(y=0\) probability density;
- `real_psi_xz.gif`: the \(y=0\) real part of the wavefunction;
- `trajectory_spreading.png`: centre trajectory and one-sigma width;
- `classical_limit.png`: suppression of relative spreading as mass increases.

The default numerical parameters are dimensionless and chosen to make
spreading and phase motion visually clear. The formulas remain valid in SI
units when all parameters use a consistent unit system. Every parameter is
editable from the command line; run `python quantum_projectile.py --help` for
the options.

## Viva targets

Be able to explain why:

1. \(V=mgz\) gives force \(-mg\hat{\mathbf z}\).
2. The wavepacket centre follows the classical parabola exactly.
3. Uniform gravity does not stop Gaussian spreading.
4. A single Gaussian in a linear potential does not create density
   interference fringes.
5. Increasing the mass suppresses relative spreading.
6. \(\operatorname{Re}\psi\) can oscillate strongly while \(|\psi|^2\) remains
   a smooth Gaussian.
