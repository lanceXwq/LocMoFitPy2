from typing import Iterable, Optional, Set

import equinox as eqx
import jax


def _path_last_key_name(path) -> Optional[str]:
    if not path:
        return None
    last = path[-1]
    if isinstance(last, jax.tree_util.GetAttrKey):
        return last.name
    if isinstance(last, jax.tree_util.DictKey):
        return str(last.key)
    # For SequenceKey / FlattenedIndexKey, there is no stable "field name"
    return None


def partition_with_freeze(model, *, freeze: Iterable[str] = ()):
    freeze_set: Set[str] = set(freeze)

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(model)

    mask_leaves = []
    for path, leaf in path_leaves:
        trainable = False

        if eqx.is_inexact_array(leaf) and getattr(leaf, "ndim", None) == 0:
            name = _path_last_key_name(path)
            # Train scalar params unless explicitly frozen by name.
            trainable = (name is not None) and (name not in freeze_set)

        mask_leaves.append(trainable)

    mask = treedef.unflatten(mask_leaves)
    return eqx.partition(model, mask)


combine = eqx.combine
