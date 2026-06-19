"""Small cache helpers shared by UI/API layers."""

from functools import wraps
from time import monotonic
from typing import Any, Callable


def ttl_cache(seconds: int = 600) -> Callable:
    """Simple in-process TTL cache for low-cost deployments."""
    store: dict[tuple[Any, ...], tuple[float, Any]] = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = args + tuple(sorted(kwargs.items()))
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError("ttl_cache arguments must be hashable") from exc
            now = monotonic()
            if key in store:
                created_at, value = store[key]
                if now - created_at < seconds:
                    return value
            value = func(*args, **kwargs)
            store[key] = (now, value)
            return value

        return wrapper

    return decorator
