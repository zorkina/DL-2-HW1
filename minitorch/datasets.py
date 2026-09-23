import math
import random
from dataclasses import dataclass


def make_pts(N: int) -> list[tuple[float, float]]:
    return [(random.random(), random.random()) for _ in range(N)]


@dataclass
class Graph:
    N: int
    X: list[tuple[float, float]]
    y: list[int]


def simple(N: int) -> Graph:
    """Label points by whether their first coordinate is less than 0.5."""
    X = make_pts(N)
    return Graph(N, X, [int(x < 0.5) for x, y in X])


def diag(N: int) -> Graph:
    """Label points by whether the sum of their coordinates is less than 0.5."""
    X = make_pts(N)
    return Graph(N, X, [int(x + y < 0.5) for x, y in X])


def split(N: int) -> Graph:
    """Label points in the two outer vertical strips of width 0.2."""
    X = make_pts(N)
    return Graph(N, X, [int(x < 0.2 or x > 0.8) for x, y in X])


def xor(N: int) -> Graph:
    """Label points in the upper-left or lower-right open quadrant."""
    X = make_pts(N)
    y = [int((a < 0.5 and b > 0.5) or (a > 0.5 and b < 0.5)) for a, b in X]
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Label points outside the circle with center (0.5, 0.5) and radius sqrt(0.1)."""
    X = make_pts(N)
    return Graph(N, X, [int((x - 0.5)**2 + (y - 0.5)**2 > 0.1) for x, y in X])


def spiral(N: int) -> Graph:
    """Generate the two opposite spiral arms used by the upstream dataset."""
    def x(t: float) -> float:
        return t * math.cos(t) / 20.0
    def y(t: float) -> float:
        return t * math.sin(t) / 20.0
    X = [(x(10.0 * i / (N // 2)) + 0.5, y(10.0 * i / (N // 2)) + 0.5)
         for i in range(5, 5 + N // 2)]
    X += [(y(-10.0 * i / (N // 2)) + 0.5, x(-10.0 * i / (N // 2)) + 0.5)
          for i in range(5, 5 + N // 2)]
    return Graph(N, X, [0] * (N // 2) + [1] * (N // 2))


datasets = {"Simple": simple, "Diag": diag, "Split": split,
            "Xor": xor, "Circle": circle, "Spiral": spiral}
