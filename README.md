# Bi-level optimization for Sliced Wasserstein with Generalized Geodesics (SWGG)

Let us define

$$
    F(\theta) = \left\langle C_{xy}, \pi^\star(\theta) \right\rangle
$$

where

$$
    \pi^\star(\theta) = \arg \min_\pi \left\langle C_{xy}^\theta, \pi \right\rangle  \, .
$$

Similar to Berthet et al, we can define

$$
    F_\varepsilon(\theta) = E_{\omega \thicksim p_{\theta, \varepsilon}} \left[ \left\langle C_{xy}, \pi^\star(\omega) \right\rangle \right] \, .
$$

In order to minimize $F_\varepsilon(\theta)$ through gradient descent, we would like to compute

$$
    \nabla_\theta F_\varepsilon
        = \nabla_\theta E_{\omega \thicksim p_{\theta, \varepsilon}} \left[ F(\omega) \right]  \, .
$$

The derivative of the expectation can be computed using Stein's Lemma that gives, in the case of a Gaussian distribution:

$$
    \nabla_\theta F_\varepsilon = E_{z \thicksim p_{0, 1}} \left[ F(\theta + \varepsilon z) \cdot \frac{z}{\varepsilon} \right]
$$

which can be estimated using Monte-Carlo sampling.
