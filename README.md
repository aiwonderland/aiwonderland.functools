# aiwonderland.functools

> My main Python functools toolset.

Additional `functools` in the spirit of the standard library's `functools`.

`pip install -e .` to install from a checkout. Python 3.10+. No
runtime dependencies.

## `aiwonderland.functools`

| Decorator / factory                   | What it does                                                 |
| ------------------------------------- | ------------------------------------------------------------ |
| `once(func)`                          | run `func` at most once, cache the result                    |
| `silent(default)`                     | catch `Exception`, return `default`                          |
| `pipe(*funcs)`                        | left-to-right composition                                    |
| `pass_param(func, validator, fb)`     | short-circuit on a sentinel first argument                   |
| `memoize(func)`                       | cache results keyed by arguments                             |
| `tap(side_effect=print)`              | log calls; return value unchanged                            |
| `print_yielded(func)`                 | print every value a generator yields                         |
| `run_sync(coro_func)`                 | call an `async def` synchronously                            |

See each callable's docstring for parameters, return type, and
examples.

```python
from aiwonderland.functools import memoize, pipe, run_sync

@memoize
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

@run_sync
async def fetch_status():
    return "ok"

print(fib(50), pipe(lambda x: x * 2, lambda x: x + 1)(3), fetch_status())
```


## License

MIT. See [LICENSE](LICENSE).
