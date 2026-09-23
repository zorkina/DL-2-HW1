from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


class Variable(Protocol):
    unique_id: int

    def accumulate_derivative(self, x: Any) -> None: ...
    def is_leaf(self) -> bool: ...
    def is_constant(self) -> bool: ...
    @property
    def parents(self) -> Iterable["Variable"]: ...
    def chain_rule(self, d_output: Any) -> Iterable[tuple["Variable", Any]]: ...


@dataclass
class Context:
    no_grad: bool = False
    saved_values: tuple[Any, ...] = field(default_factory=tuple)

    def save_for_backward(self, *values: Any) -> None:
        if not self.no_grad:
            self.saved_values = values

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        return self.saved_values


def central_difference(f: Any, *vals: Any, arg: int = 0,
                       epsilon: float = 1e-6) -> Any:
    right, left = list(vals), list(vals)
    right[arg] = right[arg] + epsilon
    left[arg] = left[arg] - epsilon
    return (f(*right) - f(*left)) / (2.0 * epsilon)


def topological_sort(variable: Variable) -> list[Variable]:
    visited: set[int] = set()
    order: list[Variable] = []
    stack = [(variable, False)]
    while stack:
        node, expanded = stack.pop()
        if node.is_constant():
            continue
        if expanded:
            order.append(node)
            continue
        if node.unique_id in visited:
            continue
        visited.add(node.unique_id)
        stack.append((node, True))
        if not node.is_leaf():
            stack.extend((parent, False) for parent in node.parents)
    order.reverse()
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    derivatives = {variable.unique_id: deriv}
    for node in topological_sort(variable):
        d = derivatives.get(node.unique_id, 0.0)
        if node.is_leaf():
            node.accumulate_derivative(d)
        else:
            for parent, contribution in node.chain_rule(d):
                if not parent.is_constant():
                    key = parent.unique_id
                    derivatives[key] = derivatives.get(key, 0.0) + contribution
