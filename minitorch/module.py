from __future__ import annotations
from typing import Any


class Module:
    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}
        self._parameters: dict[str, Parameter] = {}
        self.training = True

    def modules(self) -> list[Module]:
        return list(self._modules.values())

    def train(self) -> None:
        self.training = True
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        self.training = False
        for module in self.modules():
            module.eval()

    def named_parameters(self) -> list[tuple[str, Parameter]]:
        result = list(self._parameters.items())
        for name, module in self._modules.items():
            result.extend((f"{name}.{key}", value)
                          for key, value in module.named_parameters())
        return result

    def parameters(self) -> list[Parameter]:
        return [value for _, value in self.named_parameters()]

    def add_parameter(self, k: str, v: Any) -> Parameter:
        parameter = Parameter(v, k)
        self._parameters[k] = parameter
        return parameter

    def __setattr__(self, key: str, val: Any) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            object.__setattr__(self, key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__.get("_parameters", {}):
            return self.__dict__["_parameters"][key]
        if key in self.__dict__.get("_modules", {}):
            return self.__dict__["_modules"][key]
        raise AttributeError(key)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement forward")

    def __repr__(self) -> str:
        rows = []
        for name, child in self._modules.items():
            text = repr(child).replace("\n", "\n  ")
            rows.append(f"  ({name}): {text}")
        inside = "\n" + "\n".join(rows) + "\n" if rows else ""
        return f"{type(self).__name__}({inside})"


class Parameter:
    def __init__(self, x: Any, name: str | None = None) -> None:
        self.name = name
        self.update(x)

    def update(self, x: Any) -> None:
        self.value = x
        if hasattr(x, "requires_grad_"):
            x.requires_grad_(True)
        if self.name and hasattr(x, "name"):
            x.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
