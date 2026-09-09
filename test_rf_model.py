import numpy as np

from rf_model import make_network


def test_dynamic_service_evaluator_is_finite_and_reports_reassignment():
    net = make_network(1, tilt_levels_deg=[0.0, 5.0, 10.0], seed=3)
    out = net.evaluate_config_dynamic_service([1, 1, 1])
    for key in ("mean_sinr_db", "edge_sinr_db", "mean_spectral_efficiency",
                "outage_pct", "coverage_pct", "server_change_pct_vs_reference"):
        assert np.isfinite(out[key])
    assert 0.0 <= out["server_change_pct_vs_reference"] <= 100.0
