"""Local code interpreter for selecting ranges from an RLM's source context.

The interpreter deliberately has no LLM-query helper. Generated code can inspect
the original ``context`` and ``metadata`` values, then must set ``start``,
``end``, and ``query``. The RLM controller, outside this interpreter, uses that
selection to make the recursive model call.
"""

from __future__ import annotations

import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any


class CodeProtocolError(ValueError):
    """Raised when generated code does not obey the RLM selection protocol."""


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """A range of the original context and the query for its child agent."""

    start: int
    end: int
    query: str


@dataclass(slots=True)
class ExecutionResult:
    """The outcome of executing one generated Python cell."""

    stdout: str
    stderr: str
    selection: ContextSelection | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class _ProtectedInputGuard(ast.NodeVisitor):
    """Reject direct assignment, deletion, or shadowing of REPL inputs."""

    _protected_names = {"context", "metadata"}

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in self._protected_names:
            raise CodeProtocolError(f"Generated code must not modify '{node.id}'.")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        protected = self._protected_names.intersection(node.names)
        if protected:
            raise CodeProtocolError(
                f"Generated code must not redeclare protected input(s): {', '.join(sorted(protected))}."
            )


class REPL:
    """Persistent local Python interpreter for one top-level RLM request.

    ``context`` is the original source context and remains the same throughout
    all recursive calls belonging to that request. Code can define regular
    helper variables, which persist between cells, but it cannot replace
    ``context`` or ``metadata``. Each selection cell must define:

    - ``start``: inclusive integer offset into the original context;
    - ``end``: exclusive integer offset into the original context;
    - ``query``: non-empty query for the child RLM call.

    This is an in-process development interpreter, not a security sandbox.
    Never execute untrusted code with it in production.
    """

    _output_names = ("start", "end", "query")

    def __init__(self, context: str, metadata: str = "") -> None:
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        if not isinstance(metadata, str):
            raise TypeError("metadata must be a string")

        self._context = context
        self._metadata = metadata
        self._namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "context": self._context,
            "metadata": self._metadata,
        }

    @property
    def context(self) -> str:
        """The unchanged original context for the entire RLM request."""
        return self._context

    @property
    def metadata(self) -> str:
        """The unchanged metadata associated with the original context."""
        return self._metadata

    @property
    def variables(self) -> dict[str, Any]:
        """Persistent variables available to future generated code cells."""
        return self._namespace

    def execute_code(self, code: str) -> ExecutionResult:
        """Execute code that selects a range from the original context.

        The returned selection contains only coordinates and a child query. It
        never contains a new or copied context value; slicing is performed by
        the RLM controller after this method returns.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string")

        stdout = io.StringIO()
        stderr = io.StringIO()
        for name in self._output_names:
            self._namespace.pop(name, None)

        try:
            tree = ast.parse(code, filename="<rlm-repl>", mode="exec")
            _ProtectedInputGuard().visit(tree)

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compile(tree, "<rlm-repl>", "exec"), self._namespace)

            selection = self._read_selection()
            return ExecutionResult(
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                selection=selection,
            )
        except Exception:
            return ExecutionResult(
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                error=traceback.format_exc(),
            )

    # Short alias for interpreter-style callers.
    execute = execute_code

    def _read_selection(self) -> ContextSelection:
        missing = [name for name in self._output_names if name not in self._namespace]
        if missing:
            raise CodeProtocolError(
                "Generated code must set start, end, and query; missing: " + ", ".join(missing)
            )

        start = self._namespace["start"]
        end = self._namespace["end"]
        query = self._namespace["query"]

        if isinstance(start, bool) or not isinstance(start, int):
            raise CodeProtocolError("start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise CodeProtocolError("end must be an integer")
        if not isinstance(query, str) or not query.strip():
            raise CodeProtocolError("query must be a non-empty string")
        if start < 0 or end < start or end > len(self._context):
            raise CodeProtocolError(
                f"Invalid context range [{start}:{end}] for context length {len(self._context)}"
            )

        return ContextSelection(start=start, end=end, query=query)
