"""
run_all.py -- the full pipeline, end to end.

    demand rho(t)  --LBM-->  rho_hat(t+dt)
                   -->  RAN simulation  -->  4 KPIs  -->  F_safe(z)
                   -->  Walsh-Hadamard  -->  h_i, J_ij
                   -->  QAOA            -->  z*

Produces:
  1. exact 2^9 landscape + true optimum      (validation)
  2. Walsh spectrum, kappa, Ising couplings  (structure)
  3. QAOA vs exhaustive vs greedy vs random  (benchmark)
  4. proactive vs reactive vs static tilts   (the headline result)
"""

import json
import time

import numpy as np

from netsim import Scenario
from objective import Objective, index_to_spins, enumerate_energies, KPIS
from ising import (walsh_coefficients, extract_ising, truncation_quality,
                   two_local_energies)
from qaoa import run_qaoa, sample_best
from lbm import DemandLBM, morning_commute

M = 9
GRID = 16
AREA = 900.0
HORIZON = 20          # LBM steps ahead == one re-planning interval

SC_KW = dict(n_users=48, n_time=20, n_fade=12, speed_mps=15.0, area_m=AREA)


def make_scenario(density, seed):
    return Scenario(density=density, rng=seed, **SC_KW)


def greedy(obj, M):
    """Coordinate-ascent from all-down: flip whichever sector helps most."""
    z = np.ones(M)
    best = obj.utility(z)
    improved = True
    while improved:
        improved = False
        for i in range(M):
            z[i] *= -1
            v = obj.utility(z)
            if v > best + 1e-12:
                best, improved = v, True
            else:
                z[i] *= -1
    return z, best


def report_row(name, z, obj_eval):
    F, d = obj_eval.utility(z, detail=True)
    r = d["raw"]
    return dict(method=name, F=F,
                cov=r["cov"], thr=r["thr"], drop=r["drop"], ho=r["ho"],
                z="".join("+" if v > 0 else "-" for v in z))


def main():
    t_start = time.time()
    out = {}

    # ------------------------------------------------------------------
    # 1. Demand: now, predicted future, and true future
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STAGE 1  demand field evolution (lattice Boltzmann)")
    rho_now, u = morning_commute(GRID, seed=0)

    sim = DemandLBM(rho_now.copy(), u, tau=1.0)
    rho_pred = sim.step(HORIZON)                       # what the model predicts

    # "truth" = same dynamics plus unmodelled local activity
    rng = np.random.default_rng(11)
    bump = np.zeros_like(rho_now)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    bump += np.exp(-((xx - GRID * 0.7) ** 2 +
                     (yy - GRID * 0.25) ** 2) / (2 * (GRID * 0.08) ** 2))
    rho_true = rho_pred + 0.25 * bump / bump.sum() * rho_pred.sum()
    rho_true /= rho_true.sum()

    shift = np.array(np.unravel_index(rho_pred.argmax(), rho_pred.shape)) - \
        np.array(np.unravel_index(rho_now.argmax(), rho_now.shape))
    pred_err = float(np.abs(rho_pred - rho_true).sum())
    print(f"  mass conservation      : {rho_pred.sum():.6f}")
    print(f"  demand peak shift      : {shift} cells over {HORIZON} steps")
    print(f"  prediction L1 error    : {pred_err:.4f}")
    out["lbm"] = dict(mass=float(rho_pred.sum()), pred_l1=pred_err)

    # ------------------------------------------------------------------
    # 2. Landscape on the PREDICTED future demand
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STAGE 2  network simulation + exhaustive landscape")
    sc_plan = make_scenario(rho_pred, seed=101)
    obj_plan = Objective(sc_plan)

    t0 = time.time()
    E = enumerate_energies(obj_plan, M)                # 512 configurations
    print(f"  evaluated {E.size} configs in {time.time()-t0:.1f}s")
    k_opt = int(E.argmin())
    z_opt = index_to_spins(k_opt, M)
    print(f"  exact optimum          : E = {E.min():.4f}  "
          f"F = {-E.min():.4f}")
    print(f"  landscape spread       : E in [{E.min():.4f}, {E.max():.4f}]")

    # ------------------------------------------------------------------
    # 3. Walsh -> Ising
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STAGE 3  Walsh-Hadamard -> Ising Hamiltonian")
    Ehat = walsh_coefficients(E, M)
    c, h, J = extract_ising(Ehat, M)
    kappa, deg, p = truncation_quality(Ehat, M)
    E2 = two_local_energies(c, h, J, M)

    rmse = float(np.sqrt(np.mean((E - E2) ** 2)))
    rank = float(np.corrcoef(np.argsort(np.argsort(E)),
                             np.argsort(np.argsort(E2)))[0, 1])
    k2 = int(E2.argmin())
    gap2 = float(E[k2] - E.min())

    print(f"  kappa (deg<=2 energy)  : {kappa:.4f}")
    print(f"  2-local RMSE           : {rmse:.5f}   "
          f"(landscape sd {np.std(E):.5f})")
    print(f"  rank correlation       : {rank:.4f}")
    print(f"  2-local optimum gap    : {gap2:.5f}")
    print(f"  |h| range              : {np.abs(h).min():.4f} .. "
          f"{np.abs(h).max():.4f}")
    print(f"  |J| max                : {np.abs(J).max():.4f}")
    strongest = np.dstack(np.unravel_index(
        np.argsort(np.abs(J).ravel())[::-1][:3], J.shape))[0]
    print(f"  strongest couplings    : "
          + ", ".join(f"({a},{b}) J={J[a,b]:+.4f}" for a, b in strongest))
    out["walsh"] = dict(kappa=kappa, rmse=rmse, rank=rank, gap2=gap2,
                        h=h.tolist(), J=J.tolist(), c=c)

    # ------------------------------------------------------------------
    # 4. QAOA
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STAGE 4  QAOA on H_C^(2)")
    qres = {}
    for p_depth in (1, 2, 3):
        r = run_qaoa(E2, M, p=p_depth, seed=5)
        kb, seen = sample_best(r["probs"], E, n_shots=256)
        ar = float((E2.mean() - r["exp_energy"]) /
                   (E2.mean() - E2.min()))
        gap = float(E[kb] - E.min())
        print(f"  p={p_depth}  approx-ratio {ar:.3f}   "
              f"P(gs)={r['probs'][E2.argmin()]:.4f}   "
              f"sampled-best gap {gap:.5f}   "
              f"{'OPTIMAL' if gap < 1e-9 else ''}")
        qres[p_depth] = dict(ar=ar, gap=gap, k=kb, n_unique=int(seen.size))
    out["qaoa"] = {str(k): dict(ar=v["ar"], gap=v["gap"]) for k, v in qres.items()}
    z_qaoa = index_to_spins(qres[3]["k"], M)

    # ------------------------------------------------------------------
    # 5. Baselines on the planning scenario
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STAGE 5  benchmark against classical baselines")
    z_greedy, _ = greedy(obj_plan, M)
    rng2 = np.random.default_rng(4)
    z_rand = rng2.choice([-1, 1], M)
    z_static = np.ones(M)

    rows = [report_row("static (all down)", z_static, obj_plan),
            report_row("random", z_rand, obj_plan),
            report_row("greedy coord-ascent", z_greedy, obj_plan),
            report_row("QAOA p=3", z_qaoa, obj_plan),
            report_row("exhaustive optimum", z_opt, obj_plan)]
    print_table(rows)
    out["bench_plan"] = rows

    # ------------------------------------------------------------------
    # 6. THE HEADLINE: proactive vs reactive
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STAGE 6  proactive vs reactive, scored on the TRUE future demand")

    # reactive: optimise against demand as it is NOW
    sc_now = make_scenario(rho_now, seed=202)
    obj_now = Objective(sc_now)
    E_now = enumerate_energies(obj_now, M)
    z_reactive = index_to_spins(int(E_now.argmin()), M)

    # proactive: optimise against the LBM-predicted future demand
    z_proactive = z_opt

    # score everything on the true future demand, fresh fading seed
    sc_test = make_scenario(rho_true, seed=303)
    obj_test = Objective(sc_test)
    E_test = enumerate_energies(obj_test, M)
    z_oracle = index_to_spins(int(E_test.argmin()), M)

    rows2 = [report_row("static tilts", z_static, obj_test),
             report_row("reactive (demand now)", z_reactive, obj_test),
             report_row("proactive (LBM predicted)", z_proactive, obj_test),
             report_row("oracle (true future)", z_oracle, obj_test)]
    print_table(rows2)

    f_re = rows2[1]["F"]
    f_pro = rows2[2]["F"]
    f_or = rows2[3]["F"]
    f_st = rows2[0]["F"]
    recov = (f_pro - f_re) / max(f_or - f_re, 1e-9)
    print(f"\n  proactive - reactive   : {f_pro - f_re:+.4f} utility")
    print(f"  proactive - static     : {f_pro - f_st:+.4f} utility")
    print(f"  share of the remaining oracle gap closed : {100*recov:.1f}%")
    out["headline"] = dict(static=f_st, reactive=f_re,
                           proactive=f_pro, oracle=f_or, recovered=recov)

    print("=" * 72)
    print(f"total runtime {time.time()-t_start:.1f}s")

    with open("results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez("landscape.npz", E=E, Ehat=Ehat, h=h, J=J, E2=E2,
             rho_now=rho_now, rho_pred=rho_pred, rho_true=rho_true,
             deg=deg, p=p)
    print("wrote results.json and landscape.npz")


def print_table(rows):
    hdr = f"  {'method':<26}{'F_safe':>9}{'cov':>8}{'thr':>9}" \
          f"{'drop':>8}{'HO':>8}   config"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['method']:<26}{r['F']:>9.4f}{r['cov']:>8.3f}"
              f"{r['thr']:>9.2f}{r['drop']:>8.3f}{r['ho']:>8.3f}   {r['z']}")


if __name__ == "__main__":
    main()
