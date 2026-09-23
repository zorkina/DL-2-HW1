from typing import Iterable
from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Iterable[Parameter]) -> None:
        self.parameters = list(parameters)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            value = parameter.value
            if value is None:
                continue
            if hasattr(value, "derivative"):
                value.derivative = None
            if hasattr(value, "grad"):
                value.grad = None

    def step(self) -> None:
        raise NotImplementedError("Subclasses must implement step")


class SGD(Optimizer):
    def __init__(self, parameters: Iterable[Parameter], lr: float = 1.0) -> None:
        super().__init__(parameters)
        self.lr = lr

    def step(self) -> None:
        for parameter in self.parameters:
            value = parameter.value
            if value is None:
                continue
            if isinstance(value, Scalar) and value.derivative is not None:
                parameter.update(Scalar(value.data - self.lr * value.derivative))
            elif hasattr(value, "grad") and value.grad is not None:
                parameter.update(value - self.lr * value.grad)
