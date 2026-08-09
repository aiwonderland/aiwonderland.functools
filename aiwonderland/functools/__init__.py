from __future__ import annotations

import asyncio
from collections import deque
import functools
from itertools import islice
from typing import Any, Callable, Iterable, ParamSpec, TypeVar

# Type variables used across decorators in this module.
_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

__version__: str = "1.0.0"

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
    @functools.wraps(func)  # type: ignore[attr-defined, untyped-decorator]
    def wrapper(param: Any, /, *args: Any, **kwargs: Any) -> _R | None:
        if param is not validator:
            return func(param, *args, **kwargs)
        return fallback
    return wrapper  # type: ignore[no-any-return]


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

    return functools.reduce(compose_two, funcs)  # type: ignore[attr-defined, no-any-return]


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
    print_all = functools.partial(map, print)  # type: ignore[attr-defined]
    print_res = _compose(_consume, print_all, func)
    return functools.wraps(func)(print_res)  # type: ignore[attr-defined, no-any-return]



def memoize(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Cache the results of ``func`` keyed by positional and keyword arguments.

    Each unique ``(args, kwargs)`` invocation is computed at most once;
    subsequent calls with the same arguments return the previously
    cached value. Unlike :func:`functools.lru_cache`, this cache is
    unbounded and never evicts entries -- arguments must therefore be
    hashable.

    The original callable's metadata (``__name__``, ``__doc__``, etc.)
    is preserved via :func:`functools.wraps`.

    Args:
        func: The callable to cache. All of its arguments must be
            hashable.

    Returns:
        A wrapper with the same signature as ``func`` whose results
            are memoized.

    Examples:
        >>> @memoize
        ... def square(n):
        ...     return n * n
        >>> square(4)
        16
        >>> square(4)
        16
    """
    cache: dict[tuple[Any, frozenset[tuple[str, Any]]], _R] = {}

    @functools.wraps(func)  # type: ignore[attr-defined, untyped-decorator]
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache_clear = cache.clear
    wrapper.cache_info = lambda: {"size": len(cache)}
    return wrapper  # type: ignore[no-any-return]


def tap(side_effect: Callable[..., Any] = print) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a callable to invoke ``side_effect`` on each call's result and args.

    The wrapper invokes ``func(*args, **kwargs)`` normally, then calls
    ``side_effect(result, *args, **kwargs)`` -- passing the result as
    the first positional argument followed by the original arguments.
    The original return value of ``func`` is forwarded unchanged. This
    is useful for logging, tracing, or otherwise "tapping into" a call
    without modifying its behavior.

    By default the side effect is the built-in :func:`print`, which
    prints the result followed by the arguments; supply any callable
    accepting ``(result, *args, **kwargs)`` to customize.

    Args:
        side_effect: A callable invoked as
            ``side_effect(result, *args, **kwargs)`` after each
            successful call to ``func``. Defaults to :func:`print`.

    Returns:
        A decorator that wraps ``func`` and forwards its return value
            unchanged.

    Examples:
        >>> @tap()
        ... def add(a, b):
        ...     return a + b
        >>> add(2, 3)  # doctest: +SKIP
        5 2 3
    """
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)  # type: ignore[attr-defined, untyped-decorator]
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            result = func(*args, **kwargs)
            side_effect(result, *args, **kwargs)
            return result

        return wrapper  # type: ignore[no-any-return]

    return decorator


def run_sync(
    coro_func: Callable[_P, Any],
) -> Callable[_P, Any]:
    """Wrap an async (coroutine) callable so it can be called synchronously.

    Each call to the returned wrapper opens a fresh event loop, runs
    ``coro_func(*args, **kwargs)`` to completion on it, then closes
    the loop and returns the coroutine's result.

    This is intended for one-shot use from synchronous code -- for
    example, invoking an ``async def`` from a CLI, a unit test, or a
    REPL. It must not be called from within an already-running event
    loop; for that, schedule the coroutine directly with
    :func:`asyncio.ensure_future` instead.

    The wrapped coroutine function's metadata is preserved via
    :func:`functools.wraps`.

    Args:
        coro_func: An ``async def`` callable (or any callable returning
            an awaitable).

    Returns:
        A synchronous wrapper with the same signature as
            ``coro_func``.

    Examples:
        >>> async def fetch():
        ...     return 42
        >>> blocking_fetch = run_sync(fetch)
        >>> blocking_fetch()
        42
    """
    @functools.wraps(coro_func)  # type: ignore[attr-defined, untyped-decorator]
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_func(*args, **kwargs))
        finally:
            loop.close()

    return wrapper  # type: ignore[no-any-return]
