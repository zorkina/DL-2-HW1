import math
from typing import Callable, Iterable

EPS = 1e-6


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def log(x: float) -> float:
    return math.log(x + EPS)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable:
    def apply(xs: Iterable[float]) -> list[float]:
        return [fn(x) for x in xs]
    return apply


def negList(ls: Iterable[float]) -> list[float]:
    return map(neg)(ls)


def zipWith(fn: Callable[[float, float], float]) -> Callable:
    def apply(xs: Iterable[float], ys: Iterable[float]) -> list[float]:
        return [fn(x, y) for x, y in zip(xs, ys)]
    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
    return zipWith(add)(ls1, ls2)


def reduce(fn: Callable[[float, float], float], start: float) -> Callable:
    def apply(xs: Iterable[float]) -> float:
        result = start
        for x in xs:
            result = fn(x, result)
        return result
    return apply


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)
