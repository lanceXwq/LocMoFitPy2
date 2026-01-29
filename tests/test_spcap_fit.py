from locmofitpy2 import run_locmofit
import pandas as pd


def test_spcap_fit():
    locs_df = pd.read_csv("tests/example_spcap_data.csv")
    locs = locs_df.values[:, 0:3]
    stddev = locs_df.values[:, 3:]
    expected_final_loss = 253.7626953125
    res = run_locmofit(
        "SphericalCap", locs, stddev, seed=3, freeze=(), max_iter=200, tol=1e-6
    )
    actual_final_loss = res["final_loss"]
    assert actual_final_loss == expected_final_loss
