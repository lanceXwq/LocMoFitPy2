import numpy as np

import locmofitpy2

def run_locmofit(model_name: str, locs_np: np.ndarray, stddev_np: np.ndarray, **kwargs):
    return locmofitpy2.run_locmofit(model_name, locs_np, stddev_np, **kwargs)
