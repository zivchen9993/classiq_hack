"""
ising.py -- exact Walsh-Hadamard expansion of the binary energy landscape,
and its truncation to a 2-local Ising Hamiltonian.

Every function on {+1,-1}^M has an exact parity-basis expansion
(O'Donnell, Analysis of Boolean Functions, Prop. 1.8):

    E(z) = sum_S Ehat_S * prod_{i in S} z_i ,
    Ehat_S = 2^-M sum_z E(z) prod_{i in S} z_i

Replacing z_i -> Z_i gives the exact diagonal cost Hamiltonian.  Keeping
only |S| <= 2 gives the hardware-friendly form

    H_C^(2) = c I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

The transform itself is O(M 2^M) and runs classically in milliseconds.
It is NOT the quantum contribution -- it is how we BUILD the Hamiltonian
that QAOA then searches.  Preparing the amplitude vector would require
evaluating all 2^M configurations first, at which point no search remains.
"""

import numpy as np


def fwht(a):
    """In-place fast Walsh-Hadamard transform (unnormalised)."""
    a = np.array(a, dtype=float, copy=True)
    n = a.size
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i + h].copy()
            y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y
            a[i + h:i + 2 * h] = x - y
        h *= 2
    return a


def walsh_coefficients(E, M):
    """Ehat[S] for every subset S, indexed by its bitmask."""
    return fwht(E) / (1 << M)


def popcount(x):
    return bin(x).count("1")


def extract_ising(Ehat, M):
    """Pull constant, field and coupling terms out of the Walsh spectrum."""
    c = float(Ehat[0])
    h = np.array([Ehat[1 << i] for i in range(M)])
    J = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            J[i, j] = Ehat[(1 << i) | (1 << j)]
    return c, h, J


def truncation_quality(Ehat, M):
    """
    kappa = fraction of decision-relevant Walsh energy captured by
    degree <= 2 terms.  High kappa justifies the 2-local truncation;
    low kappa means three-sector interference interactions matter.
    The constant term is excluded -- it carries no decision information.
    """
    idx = np.arange(1, 1 << M)
    deg = np.array([popcount(int(k)) for k in idx])
    p = Ehat[idx] ** 2
    return float(p[deg <= 2].sum() / p.sum()), deg, p


def two_local_energies(c, h, J, M):
    """Energy of every basis state under H_C^(2). Returns array of size 2^M."""
    n = 1 << M
    k = np.arange(n)
    bits = ((k[:, None] >> np.arange(M)[None, :]) & 1)
    Z = 1 - 2 * bits                                    # (n, M) spins
    E2 = np.full(n, c) + Z @ h
    iu, ju = np.triu_indices(M, k=1)
    E2 += (Z[:, iu] * Z[:, ju]) @ J[iu, ju]
    return E2
