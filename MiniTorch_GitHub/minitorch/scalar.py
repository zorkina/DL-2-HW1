from __future__ import annotations
from dataclasses import dataclass
from itertools import count
from typing import Any, Iterable
import numpy as np
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ, LT, Add, Exp, Inv, Log, Mul, Neg, ReLU, ScalarFunction, Sigmoid,
)


@dataclass
class ScalarHistory:
    last_fn: type[ScalarFunction] | None = None
    ctx: Context | None = None
    inputs: tuple[Scalar, ...] = ()


_ids = count(1)
_default_history = object()


class Scalar:
    def __init__(self, v: float, back: Any = _default_history,
                 name: str | None = None) -> None:
        self.unique_id = next(_ids)
        self.data = float(v)
        self.history = ScalarHistory() if back is _default_history else back
        self.derivative: float | None = None
        self.name = str(self.unique_id) if name is None else name

    def __repr__(self) -> str:
        return f"Scalar({self.data:f})"

    def __bool__(self) -> bool:
        return bool(self.data)

    def __mul__(self, b: Any) -> Scalar:
        return Mul.apply(self, b)

    def __rmul__(self, b: Any) -> Scalar:
        return Mul.apply(b, self)

    def __add__(self, b: Any) -> Scalar:
        return Add.apply(self, b)

    def __radd__(self, b: Any) -> Scalar:
        return Add.apply(b, self)

    def __truediv__(self, b: Any) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: Any) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __lt__(self, b: Any) -> Scalar:
        return LT.apply(self, b)

    def __gt__(self, b: Any) -> Scalar:
        return LT.apply(b, self)

    def __eq__(self, b: Any) -> Scalar:
        return EQ.apply(self, b)

    def __sub__(self, b: Any) -> Scalar:
        return Add.apply(self, Neg.apply(b))

    def __rsub__(self, b: Any) -> Scalar:
        return Add.apply(b, Neg.apply(self))

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def log(self) -> Scalar:
        return Log.apply(self)

    def exp(self) -> Scalar:
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        return ReLU.apply(self)

    def accumulate_derivative(self, x: Any) -> None:
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        self.derivative = (0.0 if self.derivative is None else self.derivative) + x

    def is_leaf(self) -> bool:
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        return () if self.history is None else self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[tuple[Variable, Any]]:
        h = self.history
        assert h is not None and h.last_fn is not None and h.ctx is not None
        gradients = h.last_fn._backward(h.ctx, d_output)
        assert len(gradients) == len(h.inputs)
        return [(v, g) for v, g in zip(h.inputs, gradients) if not v.is_constant()]

    def backward(self, d_output: float | None = None) -> None:
        backpropagate(self, 1.0 if d_output is None else d_output)


ScalarLike = float | int | Scalar


def derivative_check(f: Any, *scalars: Scalar) -> None:
    for x in scalars:
        x.derivative = None
    f(*scalars).backward()
    for i, x in enumerate(scalars):
        expected = central_difference(f, *scalars, arg=i)
        value = expected.data if isinstance(expected, Scalar) else expected
        assert x.derivative is not None
        np.testing.assert_allclose(x.derivative, value, rtol=1e-2, atol=1e-2)
