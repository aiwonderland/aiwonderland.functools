import pytest

from aiwonderland.functools import (__version__,
                                    memoize,
                                    once,
                                    pass_param,
                                    pipe,
                                    print_yielded,
                                    run_sync,
                                    silent,
                                    tap)


def test_once_executes_function_exactly_once():
    calls = []

    @once
    def f(x):
        calls.append(x)
        return x * 2
    assert f(1) == 2
    assert f(2) == 2
    assert f(3) == 2
    assert calls == [1]


def test_once_caches_first_result_even_when_args_differ():
    @once
    def f(x):
        return x

    assert f(10) == 10
    assert f(999) == 10


def test_once_state_is_per_wrapper():
    @once
    def inc(x):
        return x + 1

    @once
    def dec(x):
        return x - 1

    assert inc(5) == 6
    assert dec(5) == 4
    assert inc(100) == 6
    assert dec(100) == 4


def test_once_returns_cached_none():
    calls = []

    @once
    def f():
        calls.append(1)
        return None

    assert f() is None
    assert f() is None
    assert f() is None
    assert calls == [1]


def test_once_preserves_keyword_arguments_on_first_call():
    @once
    def greet(name, *, greeting="Hello"):
        return f"{greeting}, {name}!"

    assert greet("Alice", greeting="Hi") == "Hi, Alice!"
    assert greet("Bob", greeting="Hey") == "Hi, Alice!"


def test_silent_returns_normal_result_when_no_exception():
    @silent(default=-1)
    def to_int(s):
        return int(s)

    assert to_int("7") == 7


def test_silent_returns_default_on_exception():
    @silent(default=0)
    def to_int(s):
        return int(s)

    assert to_int("not a number") == 0


def test_silent_default_can_be_arbitrary_value():
    fallback = {"err": True}

    @silent(default=fallback)
    def boom():
        raise RuntimeError("kaboom")

    assert boom() is fallback


def test_silent_default_can_be_none():
    @silent(default=None)
    def boom():
        raise ValueError("nope")

    assert boom() is None


def test_silent_does_not_swallow_keyboard_interrupt():
    @silent(default="caught")
    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupt()


def test_silent_does_not_swallow_system_exit():
    @silent(default="caught")
    def exit_():
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        exit_()


def test_silent_preserves_keyword_arguments_on_success():
    @silent(default=None)
    def add(a, *, b=0):
        return a + b

    assert add(3, b=4) == 7


def test_pipe_identity_with_no_functions():
    pipeline = pipe()
    assert pipeline(42) == 42
    assert pipeline("hello") == "hello"
    assert pipeline([1, 2, 3]) == [1, 2, 3]


def test_pipe_single_function():
    pipeline = pipe(lambda x: x + 1)
    assert pipeline(10) == 11


def test_pipe_left_to_right_composition():
    pipeline = pipe(lambda x: x * 2, lambda x: x + 1, lambda x: x * 2)
    assert pipeline(3) == 14


def test_pipe_chains_type_changing_functions():
    pipeline = pipe(str.strip, str.upper, list, len)
    assert pipeline("  hello  ") == 5


def test_pipe_threads_value_through_each_step():
    seen = []

    def record(x):
        seen.append(x)
        return x + 1

    pipeline = pipe(record, record, record)
    assert pipeline(0) == 3
    assert seen == [0, 1, 2]


def test_pass_param_forwards_non_none_first_argument():
    @pass_param
    def upper(s):
        return s.upper()

    assert upper("hello") == "HELLO"


def test_pass_param_returns_none_when_first_argument_is_none():
    @pass_param
    def upper(s):
        return s.upper()

    assert upper(None) is None


def test_pass_param_returns_none_without_invoking_func():
    calls = []

    @pass_param
    def record(value):
        calls.append(value)
        return value

    assert record(None) is None
    assert calls == []


def test_pass_param_forwards_extra_positional_arguments():
    @pass_param
    def add(a, b):
        return a + b

    assert add(3, 4) == 7
    assert add(None, 4) is None


def test_pass_param_forwards_keyword_arguments():
    @pass_param
    def configure(name, *, verbose=False):
        return (name, verbose)

    assert configure("alpha", verbose=True) == ("alpha", True)
    assert configure(None, verbose=True) is None


def test_pass_param_preserves_function_metadata():
    def my_func(x):
        """original docstring"""

    wrapped = pass_param(my_func)
    assert wrapped.__name__ == "my_func"
    assert wrapped.__doc__ == "original docstring"
    assert wrapped.__wrapped__ is my_func


def test_pass_param_does_not_swallow_truthy_falsy_values():
    @pass_param
    def echo(x):
        return x

    assert echo(0) == 0
    assert echo("") == ""
    assert echo(False) is False
    assert echo([]) == []
    assert echo(None) is None


def test_pass_param_uses_custom_validator_sentinel():
    sentinel = object()

    def show(value):
        return ("called", value)

    guarded = pass_param(show, validator=sentinel, fallback="skipped")
    assert guarded("ok") == ("called", "ok")
    assert guarded(sentinel) == "skipped"


def test_pass_param_validator_uses_identity_not_equality():
    class Marker:
        pass

    eq_marker = Marker()

    def show(value):
        return ("called", value)

    guarded = pass_param(show, validator=eq_marker, fallback="skipped")

    class EqualButNotSame:
        pass

    other = EqualButNotSame()
    other.__eq__ = lambda self, other: isinstance(other, Marker)

    assert guarded("not-marker") == ("called", "not-marker")


def test_pass_param_fallback_can_be_arbitrary_value():
    sentinel = "NOPE"

    def add(a, b):
        return a + b

    guarded = pass_param(add, validator=sentinel, fallback=[0])
    assert guarded(1, 2) == 3
    assert guarded(sentinel, 99) == [0]


def test_pass_param_validator_does_not_match_equal_value():
    sentinel_value = 0

    def echo(x):
        return ("called", x)

    guarded = pass_param(echo, validator=sentinel_value, fallback="skipped")
    assert guarded(False) == ("called", False)
    assert guarded(0.0) == ("called", 0.0)
    assert guarded(sentinel_value) == "skipped"


def test_print_yielded_prints_each_yielded_value(capsys):
    @print_yielded
    def gen():
        yield 1
        yield 2
        yield 3

    assert gen() is None
    out, _ = capsys.readouterr()
    assert out == "1\n2\n3\n"


def test_print_yielded_returns_none(capsys):
    @print_yielded
    def empty():
        if False:
            yield

    assert empty() is None
    out, _ = capsys.readouterr()
    assert out == ""


def test_print_yielded_preserves_function_metadata():
    @print_yielded
    def my_gen():
        """original docstring"""
        yield 1

    assert my_gen.__name__ == "my_gen"
    assert my_gen.__doc__ == "original docstring"
    assert my_gen.__wrapped__.__name__ == "my_gen"


def test_print_yielded_passes_args_to_wrapped_function(capsys):
    @print_yielded
    def gen(n):
        for i in range(n):
            yield i * 10

    assert gen(3) is None
    out, _ = capsys.readouterr()
    assert out == "0\n10\n20\n"


def test_print_yielded_passes_keyword_args(capsys):
    @print_yielded
    def gen(*, start, step):
        yield start
        yield start + step
        yield start + step * 2

    assert gen(start=1, step=2) is None
    out, _ = capsys.readouterr()
    assert out == "1\n3\n5\n"


def test_print_yielded_works_with_non_generator_iterables(capsys):
    @print_yielded
    def returns_list():
        return [10, 20, 30]

    assert returns_list() is None
    out, _ = capsys.readouterr()
    assert out == "10\n20\n30\n"


def test_print_yielded_drives_side_effects_of_generator(capsys):
    @print_yielded
    def side_effect_gen():
        print("before")
        yield "middle"
        print("after")

    side_effect_gen()
    out, _ = capsys.readouterr()
    assert out == "before\nmiddle\nafter\n"
def test_memoize_caches_result_by_args():
    calls = []

    @memoize
    def add(a, b):
        calls.append((a, b))
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert add(1, 3) == 4
    assert calls == [(1, 2), (1, 3)]


def test_memoize_caches_separately_for_positional_and_keyword():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(x=1) == 1
    assert calls == [1, 1]


def test_memoize_state_is_per_wrapper():
    calls_a = []
    calls_b = []

    @memoize
    def inc(x):
        calls_a.append(x)
        return x + 1

    @memoize
    def dec(x):
        calls_b.append(x)
        return x - 1

    assert inc(5) == 6
    assert dec(5) == 4
    assert inc(5) == 6
    assert dec(5) == 4
    assert calls_a == [5]
    assert calls_b == [5]


def test_memoize_cache_clear_empties_cache():
    calls = []

    @memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]
    f.cache_clear()
    assert f(3) == 6
    assert calls == [3, 3]


def test_memoize_cache_info_reports_size():
    @memoize
    def f(x):
        return x

    assert f.cache_info() == {"size": 0}
    f(1)
    f(2)
    f(1)
    assert f.cache_info() == {"size": 2}


def test_memoize_preserves_function_metadata():
    @memoize
    def original(x):
        """original docstring"""
        return x

    assert original.__name__ == "original"
    assert original.__doc__ == "original docstring"


def test_tap_returns_original_value_and_invokes_side_effect(capsys):
    @tap()
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    out, _ = capsys.readouterr()
    assert "2" in out and "3" in out and "5" in out


def test_tap_with_custom_side_effect():
    seen = []

    @tap(side_effect=lambda result, *args, **kwargs: seen.append((result, args, kwargs)))
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert seen == [(5, (2, 3), {})]


def test_tap_does_not_swallow_exception_from_func():
    @tap()
    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError):
        boom()


def test_tap_passes_keyword_arguments_through():
    seen = []

    @tap(side_effect=lambda result, *args, **kwargs: seen.append((result, args, kwargs)))
    def f(*, x, y):
        return x + y

    assert f(x=1, y=2) == 3
    assert seen == [(3, (), {"x": 1, "y": 2})]


def test_tap_side_effect_called_with_every_call():
    counter = []

    def record(result, *args, **kwargs):
        counter.append(result)

    @tap(side_effect=record)
    def f(x):
        return x * 10

    f(1)
    f(2)
    f(3)
    assert counter == [10, 20, 30]


def test_tap_preserves_function_metadata():
    @tap()
    def original(x):
        """original docstring"""
        return x

    assert original.__name__ == "original"
    assert original.__doc__ == "original docstring"


def test_run_sync_returns_coroutine_result():
    async def coro():
        return 42

    blocking = run_sync(coro)
    assert blocking() == 42


def test_run_sync_passes_args_to_coroutine():
    async def coro(a, b):
        return a + b

    blocking = run_sync(coro)
    assert blocking(2, 3) == 5


def test_run_sync_passes_keyword_arguments():
    async def coro(*, x, y):
        return x * y

    blocking = run_sync(coro)
    assert blocking(x=4, y=5) == 20


def test_run_sync_awaits_async_with_await():
    import asyncio

    async def inner():
        await asyncio.sleep(0)
        return "done"

    blocking = run_sync(inner)
    assert blocking() == "done"


def test_run_sync_preserves_function_metadata():
    async def original():
        """original docstring"""

    blocking = run_sync(original)
    assert blocking.__name__ == "original"
    assert blocking.__doc__ == "original docstring"


def test_run_sync_each_call_uses_fresh_event_loop():
    import asyncio
    seen = []

    async def coro():
        seen.append(asyncio.get_running_loop())
        return None

    blocking = run_sync(coro)
    blocking()
    blocking()
    assert len(seen) == 2
    assert seen[0] is not seen[1]
    assert seen[0].is_closed()
    assert seen[1].is_closed()


def test_run_sync_works_with_async_returning_list():
    async def coro():
        return [1, 2, 3]

    blocking = run_sync(coro)
    assert blocking() == [1, 2, 3]
def test_print_yielded_drives_side_effects_of_generator(capsys):
    @print_yielded
    def side_effect_gen():
        print("before")
        yield "middle"
        print("after")

    side_effect_gen()
    out, _ = capsys.readouterr()
    assert out == "before\nmiddle\nafter\n"