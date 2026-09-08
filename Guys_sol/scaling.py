"""
scaling.py -- does the quantum route still hold up as the network grows?

For M = 9, 12, 15 we enumerate the full landscape (up to 32768 configs),
extract the Ising Hamiltonian, and compare:

    greedy coordinate ascent   (the classical heuristic the brief cites)
    QAOA on H_C^(2)
    exact optimum

Greedy is run from many restarts so the comparison is honest -- a
single-start greedy would be a strawman.
"""

import time

import numpy as np

from netsim import Scenario
from objective import Objective, index_to_spins, enumerate_energies
from ising import walsh_coefficients, extract_ising, truncation_quality, \
    two_local_energies
from qaoa import run_qaoa, sample_best
from lbm import DemandLBM, morning_commute

SC_KW = dict(n_users=48, n_time=20, n_fade=10, speed_mps=15.0, area_m=900.0)


def greedy_multistart(obj, M, n_start=10, seed=0):
    """Best-of-n_start coordinate ascent -- a genuinely strong baseline."""
    rng = np.random.default_rng(seed)
    best_z, best_v = None, -np.inf
    n_eval = 0
    for s in range(n_start):
        z = np.ones(M) if s == 0 else rng.choice([-1, 1], M)
        v = obj.utility(z)
        n_eval += 1
        improved = True
        while improved:
            improved = False
            for i in range(M):
                z[i] *= -1
                nv = obj.utility(z)
                n_eval += 1
                if nv > v + 1e-12:
                    v, improved = nv, True
                else:
                    z[i] *= -1
        if v > best_v:
            best_z, best_v = z.copy(), v
    return best_z, best_v, n_eval


def main():
    rho_now, u = morning_commute(16, seed=0)
    rho = DemandLBM(rho_now.copy(), u, tau=1.0).step(20)

    print(f"{'M':>4}{'configs':>10}{'kappa':>8}{'greedy gap':>12}"
          f"{'QAOA gap':>10}{'QAOA evals':>12}{'greedy evals':>14}"
          f"{'exh evals':>11}")
    print("-" * 81)

    for n_sites in (3, 4, 5):
        sc = Scenario(density=rho, rng=101, n_sites=n_sites, **SC_KW)
        M = sc.M
        obj = Objective(sc, n_calib=48)

        t0 = time.time()
        E = enumerate_energies(obj, M)
        t_exh = time.time() - t0
        E_opt = E.min()

        Ehat = walsh_coefficients(E, M)
        c, h, J = extract_ising(Ehat, M)
        kappa, _, _ = truncation_quality(Ehat, M)
        E2 = two_local_energies(c, h, J, M)

        zg, vg, n_g = greedy_multistart(obj, M, n_start=10, seed=3)
        gap_g = (-vg) - E_opt

        r = run_qaoa(E2, M, p=3, seed=5, n_restart=6)
        kb, seen = sample_best(r["probs"], E, n_shots=256)
        gap_q = E[kb] - E_opt
        # honest accounting: QAOA needs the landscape only through h,J.
        # Sampled candidates are the only full-simulator evaluations.
        n_q = int(seen.size)

        print(f"{M:>4}{1<<M:>10}{kappa:>8.3f}{gap_g:>12.5f}"
              f"{gap_q:>10.5f}{n_q:>12}{n_g:>14}{1<<M:>11}"
              f"   (exhaustive {t_exh:.0f}s)")


if __name__ == "__main__":
    main()
