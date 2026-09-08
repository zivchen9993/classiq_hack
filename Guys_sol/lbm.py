"""
lbm.py -- D2Q5 lattice Boltzmann advection-diffusion for the user-demand
density field rho(x, y, t).

This is the CLASSICAL reference implementation.  It is deliberately written
as the two canonical LBM operators, because those are exactly the two
operators a quantum LBM implements:

    stream   : f_i(x + c_i, t+1) <- f_i(x, t)      -> qubit shift operator
    collide  : f_i <- f_i - (1/tau)(f_i - f_i^eq)  -> ancilla + rotation

Use this to (a) produce the demand prediction end-to-end today, and
(b) validate any QLBM circuit against an exact classical reference.
"""

import numpy as np

# D2Q5 lattice
C = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
W = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
CS2 = 1.0 / 3.0


class DemandLBM:
    """Advection-diffusion of a scalar demand density on an N x N lattice."""

    def __init__(self, rho0, u_field, tau=1.0):
        """
        rho0    : (N, N) initial demand density
        u_field : (N, N, 2) advection velocity in lattice units (|u| << 1)
        tau     : BGK relaxation time; diffusivity D = cs^2 (tau - 1/2)
        """
        self.N = rho0.shape[0]
        self.u = u_field
        self.tau = tau
        self.f = self.equilibrium(rho0, u_field)

    @staticmethod
    def equilibrium(rho, u):
        cu = np.einsum("ij,xyj->xyi", C, u)          # (N,N,5)
        return W[None, None, :] * rho[..., None] * (1.0 + cu / CS2)

    @property
    def rho(self):
        return self.f.sum(axis=-1)

    def stream(self):
        """Shift each population along its lattice velocity (periodic)."""
        for i, c in enumerate(C):
            self.f[..., i] = np.roll(self.f[..., i], shift=(c[0], c[1]),
                                     axis=(0, 1))

    def collide(self):
        """BGK relaxation towards local equilibrium."""
        feq = self.equilibrium(self.rho, self.u)
        self.f += (feq - self.f) / self.tau

    def step(self, n=1):
        for _ in range(n):
            self.collide()
            self.stream()
        return self.rho


# ----------------------------------------------------------------------
def gaussian_blob(N, cx, cy, sigma, amp=1.0):
    y, x = np.mgrid[0:N, 0:N]
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def morning_commute(N=16, seed=0):
    """
    Synthetic stand-in for a real demand snapshot.  Two residential
    concentrations that drift towards a business district.

    SWAP FOR REAL DATA: the Telecom Italia Milan grid is already a
    100x100 lattice of 235 m cells at 10-minute resolution -- load an
    NxN sub-grid straight into rho0 and fit u_field from consecutive
    frames (optical flow or a least-squares continuity fit).
    """
    rho0 = (gaussian_blob(N, N * 0.25, N * 0.30, N * 0.10, 1.0)
            + gaussian_blob(N, N * 0.20, N * 0.75, N * 0.09, 0.8)
            + 0.05)
    rho0 /= rho0.sum()

    # drift towards the business district in the north-east
    u = np.zeros((N, N, 2))
    u[..., 0] = 0.15
    u[..., 1] = 0.08
    return rho0, u
