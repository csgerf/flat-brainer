from typing import Any, Dict


__ALL__ = ['maybe_eval', 'without']

def maybe_eval(x: str) -> Any:
    if isinstance(x, str):
        if x.startswith("$"):
            return eval(x[1:])
        return x
    elif isinstance(x, list):
        return list(map(maybe_eval, x))
    else:
        return x


def without(dictionary: Dict, key: str) -> Dict:
    """
    Return copy of dictionary without given key.

    Args:
        dictionary: Input dictionary
        key: Key to remove

    Returns:
        Always returns new dictionary even without given key
    """
    if isinstance(key, str):
        key = {key}
    return dict((k, v) for (k, v) in dictionary.items() if k not in key)
