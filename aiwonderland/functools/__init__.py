from __future__ import annotations

from collections import deque
import functools
from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, ParamSpec, TypeVar

__version__: str = "dev5"

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


def pass_param(
    func: Callable[..., _R | None],
    validator: Any = None,
    fallback: _R | None = None,
) -> Callable[..., _R | None]:
    """Wrap ``func`` so a sentinel first argument short-circuits to ``fallback``.

    The returned wrapper has the same positional/keyword signature as
    ``func`` for everything after the first parameter, but inspects the
    first positional argument before invoking ``func``:

    - If the first positional argument is **not** ``validator`` (compared
      with ``is``, i.e. identity), the wrapper forwards it (along with any
      additional positional and keyword arguments) to ``func`` and
      returns the result.
    - If the first positional argument **is** ``validator``, the wrapper
      returns ``fallback`` immediately without invoking ``func``.

    The identity check (``is``) makes the sentinel comparison cheap and
    unambiguous -- only the exact object passed as ``validator`` triggers
    the short-circuit. Any other object, including falsy values like
    ``0``, ``""``, or ``False``, is forwarded normally.

    Args:
        func: The callable to wrap. It is invoked only when the wrapper's
            first positional argument is not ``validator``.
        validator: The sentinel value that triggers the short-circuit
            when passed as the wrapper's first positional argument.
            Defaults to ``None``.
        fallback: The value returned when ``validator`` is passed.
            Defaults to ``None``.

    Returns:
        A wrapper with the same metadata as ``func`` (via
        :func:`functools.wraps`) that returns ``fallback`` when its first
        positional argument is ``validator``, and otherwise delegates to
        ``func``.

    Examples:
        >>> @pass_param
        ... def upper(s):
        ...     return s.upper()
        >>> upper("hello")
        'HELLO'
        >>> upper(None) is None
        True

        >>> sentinel = object()
        >>> def show(value):
        ...     return f"value={value}"
        >>> show_guarded = pass_param(show, validator=sentinel, fallback="missing")
        >>> show_guarded("ok")
        'value=ok'
        >>> show_guarded(sentinel)
        'missing'
    """
    @functools.wraps(func)
    def wrapper(param: Any, /, *args: Any, **kwargs: Any) -> _R | None:
        if param is not validator:
            return func(param, *args, **kwargs)
        return fallback
    return wrapper


def _consume(iterator: Iterable[Any], i: int | None = None) -> None:
    if i is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, i, i), None)


def _compose(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    def compose_two(
        func1: Callable[..., Any], func2: Callable[..., Any]
    ) -> Callable[..., Any]:
        return lambda *args, **kwargs: func1(func2(*args, **kwargs))

    return functools.reduce(compose_two, funcs)


def print_yielded(
    func: Callable[..., Iterable[Any]],
) -> Callable[..., None]:
    """Decorate a generator (or iterable-returning) callable.

    The wrapped callable is invoked, the resulting iterator is forced
    to completion via :func:`_consume`, and every yielded value is
    printed via the built-in :func:`print` in the order it is produced.
    The wrapper itself returns ``None``.

    The original callable's metadata (``__name__``, ``__doc__``, etc.)
    is preserved via :func:`functools.wraps`, so introspection on the
    wrapper still points at ``func``.

    Args:
        func: A callable that returns an iterable (typically a
            generator) when invoked. Its return value must be iterable;
            generators are the canonical use case.

    Returns:
        A wrapper that prints each value produced by ``func(*args,
        **kwargs)`` and returns ``None``.

    Examples:
        >>> @print_yielded
        ... def numbers():
        ...     yield 1
        ...     yield 2
        ...     yield 3
        >>> numbers()  # doctest: +SKIP
        1
        2
        3
    """
    print_all = functools.partial(map, print)
    print_res = _compose(_consume, print_all, func)
    return functools.wraps(func)(print_res)