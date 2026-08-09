import pytest

from aiwonderland.functools import __version__, once, pipe, silent


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