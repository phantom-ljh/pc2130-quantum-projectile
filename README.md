# PC2130 Quantum Projectile

Computational and animation code for the PC2130 assignment **Quantum projectile**.

> AI was used as a learning/coding aid. The submitted report should be written in the student's own words, and every derivation, code block, parameter choice, figure, and physical interpretation should be explainable during the viva.

## Model

A particle of mass (m) moves in three dimensions under uniform gravity in the (-z) direction,

[
V(z)=mgz,qquad
H=-rac{hbar^2}{2m}
abla^2+mgz.
]

The initial state is

[
psi(x,y,z,0)=
(2pisigma_0^2)^{-3/4}
exp!left[-rac{x^2+y^2+(z-z_0)^2}{4sigma_0^2}ight]
e^{ip_0x/hbar},
qquad p_0=mv_0.
]

The exact time evolution used by the code is

[
psi(x,y,z,t)=
e^{-rac{i}{hbar}left(mgtz+rac16mg^2t^3ight)}
psi_{m free}!left(x,y,z+rac12gt^2,tight).
]

Hence

[
sigma(t)=sigma_0sqrt{1+left(rac{hbar t}{2msigma_0^2}ight)^2},
]

and the packet centre obeys

[
langle xangle=v_0t,quad
langle yangle=0,quad
langle zangle=z_0-rac12gt^2.
]

The energy expectation is

[
langle Eangle=
rac{p_0^2}{2m}
+rac{3hbar^2}{8msigma_0^2}
+mgz_0.
]

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python quantum_projectile.py --outdir outputs
```

Expected outputs:

- `outputs/density_xz.gif`
- `outputs/real_psi_xz.gif`
- `outputs/trajectory_spreading.png`
- `outputs/classical_limit.png`

The default numerical parameters are dimensionless and chosen to make spreading and phase motion visually clear. The analytical formulas remain valid in SI units if a consistent unit system is used.

## Viva targets

Be able to explain why:

1. (V=mgz) gives force (-mghat z).
2. The wavepacket centre follows the classical parabola exactly.
3. Uniform gravity does not stop Gaussian spreading.
4. A single Gaussian in a linear potential does not create density interference fringes.
5. Increasing the mass suppresses relative spreading.
6. (mathrm{Re},psi) can oscillate strongly while (|psi|^2) remains a smooth Gaussian.
