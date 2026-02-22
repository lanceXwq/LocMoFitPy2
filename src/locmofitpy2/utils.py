from dataclasses import dataclass
from typing import Iterable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Data:
    locs: Array
    precs: Array

    def norm_const(self, blur: Array):
        half_prec = (self.precs + blur) ** -2 / 2
        log_const = jnp.log(jnp.prod(half_prec, axis=1) * 8) / 2
        return half_prec, log_const


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
