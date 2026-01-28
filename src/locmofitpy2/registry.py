from typing import Dict

_MODEL_REGISTRY: Dict[str, type] = {}
_ALIASES: Dict[str, str] = {}


def _norm(name: str) -> str:
    return name.strip().lower()


def register_model(name: str, cls: type, *, aliases: tuple[str, ...] = ()) -> None:
    """
    Register a model class under a canonical name plus optional aliases.
    """
    canon = _norm(name)
    _MODEL_REGISTRY[canon] = cls
    for a in aliases:
        _ALIASES[_norm(a)] = canon


def list_models() -> tuple[str, ...]:
    """
    Return canonical registered model names (not aliases).
    """
    return tuple(sorted(_MODEL_REGISTRY.keys()))


def get_model_cls(model_name: str) -> type:
    """
    Resolve a model name/alias to a registered model class.
    Validates minimal interface: callable init(), callable instances.
    """
    key = _norm(model_name)
    key = _ALIASES.get(key, key)

    cls = _MODEL_REGISTRY.get(key)
    if cls is None:
        avail = ", ".join(list_models())
        raise ValueError(
            f"Unknown model_name={model_name!r}. Available models: {avail}"
        )

    init = getattr(cls, "init", None)
    if init is None or not callable(init):
        raise TypeError(f"{cls.__name__} must define a callable `init(key, **kwargs)`.")

    # NOTE: this checks the class has __call__ defined, which implies instances are callable.
    if not hasattr(cls, "__call__"):
        raise TypeError(
            f"{cls.__name__} instances must be callable (implement __call__)."
        )

    return cls
