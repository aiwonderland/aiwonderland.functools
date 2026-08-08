from __future__ import annotations

from typing import Any, Callable, ParamSpec, TypeVar


__version__: str = "dev2"

# Type variables used across decorators in this module.
_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")


def once(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorate a callable so that it executes at most once.

    The first call to the wrapped callable runs ``func`` and caches the
    result. Every subsequent call -- regardless of the arguments passed --
    returns the cached result without re-invoking ``func``.

    This is useful for lazy/one-shot initialization patterns (for example,
    building a singleton or registering a handler) where the arguments of
    the first call are immaterial and any later call should be a no-op.

    Args:
        func: The callable to wrap. It is invoked at most once, on the first
            call to the returned wrapper.

    Returns:
        A wrapper with the same signature as ``func`` whose return value is
        memoized after the first invocation.

    Examples:
        >>> @once
        ... def get_config():
        ...     print("loading...")
        ...     return {"k": 1}
        >>> get_config()
        loading...
        {'k': 1}
        >>> get_config()  # cached, no print
        {'k': 1}
    """
    executed: bool = False
    cache_result: _R | None = None

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        nonlocal executed, cache_result
        if not executed:
            cache_result = func(*args, **kwargs)
            executed = True
        return cache_result  # type: ignore[return-value]

    return wrapper


def silent(default: _T) -> Callable[[Callable[_P, Any]], Callable[_P, _T | Any]]:
    """Decorate a callable so that raised exceptions are swallowed.

    Any :class:`Exception` raised by ``func`` is caught and replaced with
    ``default``. Non-exception control flow (e.g. :class:`KeyboardInterrupt`,
    :class:`SystemExit`) is not intercepted.

    Args:
        default: The value returned when the wrapped callable raises an
            exception. May be of any type; it is returned as-is.

    Returns:
        A decorator that takes a callable and returns a wrapper with the
        same signature. The wrapper returns ``default`` on failure and the
        wrapped callable's normal return value on success.

    Examples:
        >>> @silent(default=0)
        ... def to_int(s):
        ...     return int(s)
        >>> to_int("5")
        5
        >>> to_int("not a number")
        0
    """
    def decorator(func: Callable[_P, Any]) -> Callable[_P, _T | Any]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T | Any:
            try:
                return func(*args, **kwargs)
            except Exception:
                return default
        return wrapper
    return decorator


def pipe(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose callables left-to-right into a single pipeline.

    The resulting pipeline applies each function in ``funcs`` to the
    previous result, starting from the value passed to the pipeline.
    Composition is left-to-right (top-to-bottom), unlike
    :func:`functools.reduce`, so ``pipe(f, g, h)(x)`` is equivalent to
    ``h(g(f(x)))``.

    Args:
        *funcs: Zero or more single-argument callables to compose. When no
            functions are supplied, the pipeline simply returns its input
            unchanged.

    Returns:
        A callable that accepts a single value and threads it through
        ``funcs`` from left to right.

    Examples:
        >>> double = lambda x: x * 2
        >>> inc = lambda x: x + 1
        >>> pipeline = pipe(double, inc, double)
        >>> pipeline(3)
        14
    """
    def pipeline(value: Any) -> Any:
        result: Any = value
        for fn in funcs:
            result = fn(result)
        return result
    return pipeline
