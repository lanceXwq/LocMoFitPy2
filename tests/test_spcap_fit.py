import pandas as pd

from locmofitpy2 import run_locmofit


def test_spcap_fit():
    locs_df = pd.read_csv("tests/example_spcap_data.csv")
    locs = locs_df.values[:, 0:3]
    stddev = locs_df.values[:, 3:]
    expected_final_loss = 96.35659
    res = run_locmofit(
        "SphericalCap",
        locs,
        stddev,
        init_params={"c": 0.02},
        freeze=(),
        max_iter=200,
        tol=1e-6,
        spacing=3.0,
        dtype="float32",
    )
    actual_final_loss = res["losses"][-1]
    assert abs(actual_final_loss - expected_final_loss) / expected_final_loss < 1e-6
