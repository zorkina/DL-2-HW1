from __future__ import annotations
from typing import Any
from . import operators
from .autodiff import Context


def wrap_tuple(x: Any) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def unwrap_tuple(x: tuple) -> Any:
    return x[0] if len(x) == 1 else x


class ScalarFunction:
    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> tuple:
        return wrap_tuple(cls.backward(ctx, d_out))

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)

    @classmethod
    def apply(cls, *vals: Any) -> Any:
        from .scalar import Scalar, ScalarHistory
        scalars = [v if isinstance(v, Scalar) else Scalar(v)
                   for v in vals]
        need_grad = any(not v.is_constant() for v in scalars)
        ctx = Context(not need_grad)
        result = cls._forward(ctx, *(v.data for v in scalars))
        history = ScalarHistory(cls, ctx, tuple(scalars)) if need_grad else None
        return Scalar(result, back=history)


class Add(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> tuple[float, float]:
        return d_output, d_output


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> tuple[float, float]:
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.inv_back(ctx.saved_values[0], d_output)


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -d_output


class Log(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.log_back(ctx.saved_values[0], d_output)


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return ctx.saved_values[0] * d_output


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        result = ctx.saved_values[0]
        return result * (1.0 - result) * d_output


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.relu_back(ctx.saved_values[0], d_output)


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> tuple[float, float]:
        return 0.0, 0.0


class EQ(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> tuple[float, float]:
        return 0.0, 0.0
