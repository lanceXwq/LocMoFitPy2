from dataclasses import dataclass
from typing import Iterable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Data:
    locs: Array
    stddev: Array
    half_precisions: Array
    log_consts: Array

    @staticmethod
    def from_arrays(locs, loc_precisions, *, dtype=jnp.float32):
        locs = jnp.asarray(locs, dtype=dtype)
        loc_precisions = jnp.asarray(loc_precisions, dtype=dtype)
        half_tau = (loc_precisions**-2) / 2.0
        log_const = -0.5 * jnp.log(jnp.prod(loc_precisions**2, axis=1))
        return Data(locs, loc_precisions, half_tau, log_const)


def _path_last_key_name(path) -> Optional[str]:
    if not path:
        return None
    last = path[-1]
    if isinstance(last, jax.tree_util.GetAttrKey):
        return last.name
    if isinstance(last, jax.tree_util.DictKey):
        return str(last.key)
    return None


def partition_with_freeze(model, *, freeze: Iterable[str] = ()):
    freeze_set = set(freeze)
    trainable_set = set(model.trainable_names())  # you define this per model

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(model)

    mask_leaves = []
    for path, leaf in path_leaves:
        if eqx.is_inexact_array(leaf):
            name = _path_last_key_name(path)
            trainable = (name in trainable_set) and (name not in freeze_set)
        else:
            trainable = False
        mask_leaves.append(trainable)

    mask = treedef.unflatten(mask_leaves)
    return eqx.partition(model, mask)
