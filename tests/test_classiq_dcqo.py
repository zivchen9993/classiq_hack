"""Dependency-free validation of the Classiq DCQO Pauli construction."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classiq_dcqo import (bias_pauli_terms, cost_pauli_terms,
                          dcqo_pauli_operators, merge_pauli_terms,
                          local_trotter_reference,
                          pauli_commutator, scheduled_step_terms,
                          xy_driver_pauli_terms)  # noqa: E402
from qubo_builder import TiltQUBO  # noqa: E402
from dcqo import _dense_operators  # noqa: E402


PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}


def dense(terms, n):
    out = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for coefficient, sparse in terms:
        lookup = dict(sparse)
        term = np.array([[1.0 + 0.0j]])
        for q in range(n):
            term = np.kron(term, PAULI[lookup.get(q, "I")])
        out += coefficient * term
    return out


def toy_qubo():
    # Two blocks x two tilts, with unary and cross-block pairwise physics.
    Q = np.array([[-1.1, 4.0, 0.2, -0.3],
                  [0.0, -0.4, 0.5, 0.1],
                  [0.0, 0.0, -0.9, 4.0],
                  [0.0, 0.0, 0.0, -0.2]])
    return TiltQUBO(2, 2, Q, offset=4.0, penalty=2.0,
                    w_cov=1.0, w_int=1.0, w_ho=1.0,
                    cov_norm=np.array([[-0.9, -1.6], [-1.1, -1.8]]),
                    int_norm={}, ho_norm={},
                    extra_pairwise={(0, 1): np.array([[0.2, -0.3], [0.5, 0.1]])})


def test_pauli_commutator_matches_dense_matrix():
    q = toy_qubo()
    driver = xy_driver_pauli_terms(q, driver_g=0.73, topology="chain")
    cost = cost_pauli_terms(q)
    got = dense(pauli_commutator(driver, cost), q.n_vars)
    Hd, Hf = dense(driver, q.n_vars), dense(cost, q.n_vars)
    want = 1j * (Hd @ Hf - Hf @ Hd)
    assert np.max(np.abs(got - want)) < 1e-12
    assert np.max(np.abs(got - got.conj().T)) < 1e-12


def test_penalty_is_removed_from_constraint_preserving_cost():
    q = toy_qubo()
    terms = cost_pauli_terms(q)
    # Changing the penalty changes neither the physical terms nor the CD terms.
    q2 = toy_qubo()
    q2.penalty = 0.0
    q2.Q[0, 0] += 2.0
    q2.Q[1, 1] += 2.0
    q2.Q[2, 2] += 2.0
    q2.Q[3, 3] += 2.0
    q2.Q[0, 1] -= 4.0
    q2.Q[2, 3] -= 4.0
    assert terms == cost_pauli_terms(q2)


def test_cd_contains_higher_body_terms_for_touched_couplings():
    q = toy_qubo()
    operators = dcqo_pauli_operators(q, topology="chain", cd_probes=2, seed=9)
    weights = {len(paulis) for _, paulis in operators["cd"]}
    assert 3 in weights
    assert all(abs(coefficient) > 1e-12 for coefficient, _ in operators["cd"])


def test_generated_operators_match_reduced_numpy_reference():
    q = toy_qubo()
    bias = np.array([[0.03, -0.03], [0.08, -0.08]])
    operators = dcqo_pauli_operators(q, unary_bias=bias, topology="chain",
                                     cd_probes=2, seed=9)
    indices = []
    for cfg in ((0, 0), (0, 1), (1, 0), (1, 1)):
        bits = np.zeros(q.n_vars, dtype=int)
        bits[cfg[0]] = 1
        bits[2 + cfg[1]] = 1
        indices.append(int("".join(map(str, bits)), 2))

    H_initial = dense(operators["initial"], q.n_vars)[np.ix_(indices, indices)]
    H_final = dense(operators["final"], q.n_vars)[np.ix_(indices, indices)]
    H_cd = dense(operators["cd"], q.n_vars)[np.ix_(indices, indices)]
    want_initial, want_final, want_cd, _ = _dense_operators(operators["ops"])

    def centered(matrix):
        return matrix - np.trace(matrix) / len(matrix) * np.eye(len(matrix))

    assert np.max(np.abs(centered(H_initial) - centered(want_initial))) < 1e-12
    assert np.max(np.abs(centered(H_final) - centered(want_final))) < 1e-12
    assert np.max(np.abs(H_cd - want_cd)) < 1e-12


def test_bias_maps_one_hot_projectors_to_z_up_to_constant():
    bias = np.array([[0.3, -0.2], [0.7, 0.1]])
    H = dense(bias_pauli_terms(bias), 4)
    for cfg in ((0, 0), (0, 1), (1, 0), (1, 1)):
        bits = np.zeros(4, dtype=int)
        bits[cfg[0]] = 1
        bits[2 + cfg[1]] = 1
        idx = int("".join(map(str, bits)), 2)
        expected = bias[0, cfg[0]] + bias[1, cfg[1]] - bias.sum() / 2.0
        assert abs(H[idx, idx].real - expected) < 1e-12


def test_schedule_logs_every_nonvariational_step():
    operators = dcqo_pauli_operators(toy_qubo(), topology="chain",
                                     cd_probes=2, seed=9)
    steps = scheduled_step_terms(operators, n_steps=5, time_scale=3.0)
    assert len(steps) == 5
    assert all(step["dt"] > 0 for step in steps)
    assert all(0 <= step["lambda"] <= 1 for step in steps)


def test_full_qubit_trotter_reference_stays_normalized_and_feasible():
    q = toy_qubo()
    operators = dcqo_pauli_operators(q, topology="chain", cd_probes=2, seed=9)
    steps = scheduled_step_terms(operators, n_steps=2, time_scale=1.0,
                                 mode="full")
    exact = min(q.objective_from_config(cfg)
                for cfg in ((0, 0), (0, 1), (1, 0), (1, 1)))
    result = local_trotter_reference(q, steps, repetitions=2,
                                     exact_objective=exact)
    assert abs(np.linalg.norm(result["statevector"]) - 1.0) < 1e-12
    assert abs(result["feasible_probability"] - 1.0) < 1e-12
    assert 0.0 <= result["probability_of_optimum"] <= 1.0


def test_merging_cancels_identical_strings_below_tolerance():
    terms = merge_pauli_terms([(1.0, ((0, "X"),)),
                               (-1.0 + 1e-14, ((0, "X"),)),
                               (0.4, ((1, "Z"),))])
    assert terms == [(0.4, ((1, "Z"),))]
