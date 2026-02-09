import pandas as pd

from locmofitpy2 import run_locmofit


def test_ring_fit():
    locs_df = pd.read_csv("tests/example_ring_data.csv")
    locs = locs_df.values[:, 0:3]
    loc_precs = locs_df.values[:, 3:]
    expected_final_loss = 391.1737
    res = run_locmofit(
        "Ring",
        locs,
        loc_precs,
        init_params={"r": 10.0},
        freeze=(),
        max_iter=200,
        tol=1e-6,
        spacing=3.0,
        dtype="float32",
    )
    actual_final_loss = res["losses"][-1]
    assert abs(actual_final_loss - expected_final_loss) / expected_final_loss < 1e-6
