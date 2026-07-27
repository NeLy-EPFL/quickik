"""Griffe extension recovering PyO3 constructor signatures.

PyO3 classes (e.g. `quickik.SolverConfig`) expose their constructor via
`__new__`, not `__init__` -- the real signature only shows up on the class
object itself (what `inspect.signature(cls)` reads from `__text_signature__`).
Griffe's dynamic inspector only ever looks for `__init__` when building a
class's constructor, so it never recovers this -- see
https://github.com/mkdocstrings/griffe/issues/320. This extension fills the
gap: for any inspected class with no usable `__init__`, it synthesizes one
from `inspect.signature(runtime_class)`.
"""

from __future__ import annotations

import inspect
from typing import Any

import griffe

_KIND_MAP = {
    inspect.Parameter.POSITIONAL_ONLY: griffe.ParameterKind.positional_only,
    inspect.Parameter.POSITIONAL_OR_KEYWORD: griffe.ParameterKind.positional_or_keyword,
    inspect.Parameter.VAR_POSITIONAL: griffe.ParameterKind.var_positional,
    inspect.Parameter.KEYWORD_ONLY: griffe.ParameterKind.keyword_only,
    inspect.Parameter.VAR_KEYWORD: griffe.ParameterKind.var_keyword,
}


class Pyo3ConstructorExtension(griffe.Extension):
    """Synthesizes `__init__` for compiled classes that only define `__new__`."""

    def on_class_instance(
        self, *, node: Any, cls: griffe.Class, agent: Any, **kwargs: Any
    ) -> None:
        if "__init__" in cls.members:
            return
        runtime_class = getattr(node, "obj", None)
        if runtime_class is None:
            return
        try:
            signature = inspect.signature(runtime_class)
        except (TypeError, ValueError):
            return
        if not signature.parameters:
            return
        parameters = griffe.Parameters(
            *(
                griffe.Parameter(
                    name,
                    kind=_KIND_MAP[parameter.kind],
                    default=(
                        repr(parameter.default)
                        if parameter.default is not inspect.Parameter.empty
                        else None
                    ),
                )
                for name, parameter in signature.parameters.items()
            )
        )
        cls.set_member(
            "__init__", griffe.Function("__init__", parameters=parameters, parent=cls)
        )
