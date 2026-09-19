# Viva notes: quantum projectile

## 1. Exact solution and the gravity-phase signs

Write the free-particle solution as \(\phi(x,y,s,t)\) and define

\[
s=z+\frac12gt^2,\qquad
\psi=e^{-iA/\hbar}\phi(x,y,s,t),\qquad
A=mgtz+\frac16mg^2t^3.
\]

The plus sign in \(s\) makes a free packet centred at \(s=z_0\) appear at
\(z=z_0-gt^2/2\). Differentiating gives

\[
i\hbar\partial_t\psi=e^{-iA/\hbar}
\left[(mgz+\tfrac12mg^2t^2)\phi+i\hbar gt\,\partial_s\phi
+i\hbar\partial_t\phi\right].
\]

The transformed kinetic term is

\[
-\frac{\hbar^2}{2m}\partial_z^2\psi=e^{-iA/\hbar}
\left[\frac12mg^2t^2\phi+i\hbar gt\,\partial_s\phi
-\frac{\hbar^2}{2m}\partial_s^2\phi\right].
\]

After adding \(mgz\psi\), both sides agree because \(\phi\) satisfies the
free Schrödinger equation. This checks the minus sign in the exponential and
the positive \(mgtz\) and \(mg^2t^3/6\) terms inside \(A\).

## 2. Initial state and normalization

At \(t=0\), the shift and gravity phase vanish, while the spreading factor
\(q=1+i\hbar t/(2m\sigma_0^2)\) becomes one. The code therefore returns the
assigned Gaussian and the plane wave \(e^{ip_0x/\hbar}\) exactly.

The density is a product of three normalized normal distributions:

\[
|\psi|^2=\frac{1}{(2\pi\sigma(t)^2)^{3/2}}
\exp\left[-\frac{(x-v_0t)^2+y^2+(z-z_0+gt^2/2)^2}
{2\sigma(t)^2}\right].
\]

Each one-dimensional integral is one, so the three-dimensional integral is
one for every time. `numerical_normalization` checks the same statement on a
large finite box.

## 3. Energy expectation

At \(t=0\), the mean kinetic energy contains the translational part and the
momentum variance of all three coordinates:

\[
\langle T\rangle=\frac{p_0^2}{2m}
+3\frac{\hbar^2}{8m\sigma_0^2}.
\]

Because the initial Gaussian has \(\langle z\rangle=z_0\),
\(\langle V\rangle=mgz_0\). Hence

\[
\langle E\rangle=\frac{p_0^2}{2m}
+\frac{3\hbar^2}{8m\sigma_0^2}+mgz_0.
\]

The Hamiltonian is time independent, so this expectation is conserved. The
tests also evaluate the kinetic gradient and potential integral numerically.

## 4. Ehrenfest motion

Since \(-\partial V/\partial z=-mg\), Ehrenfest's theorem gives

\[
\langle p_x\rangle=p_0,\quad \langle p_y\rangle=0,\quad
\langle p_z\rangle=-mgt,
\]

and therefore

\[
\langle x\rangle=v_0t,\quad \langle y\rangle=0,\quad
\langle z\rangle=z_0-\frac12gt^2.
\]

The packet centre follows the classical trajectory exactly, but its width
still grows as

\[
\sigma(t)=\sigma_0\sqrt{1+
\left(\frac{\hbar t}{2m\sigma_0^2}\right)^2}.
\]

Uniform gravity accelerates the centre; it does not remove free spreading.

## 5. What the animations show

- The density animation shows measurement probability. It stays a smooth,
  positive Gaussian while falling and spreading.
- The real-part animation shows phase as alternating positive and negative
  colour. These oscillations are not alternating probabilities.
- A single Gaussian in a linear potential has no density interference
  fringes: there are not two distinguishable amplitudes being recombined,
  and the gravity factor is a pure phase that cancels in \(|\psi|^2\).
- Increasing \(m\) decreases \(\hbar t/(2m\sigma_0^2)\), so relative
  spreading becomes smaller. This is the classical-limit trend shown in the
  final figure.
