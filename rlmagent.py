"""Recursive Language Model controller.

The model may emit Python only to select a range from the original context. The
controller executes that code through ``REPL`` and performs every recursive LLM
call itself; generated code never invokes an LLM directly.
"""

from __future__ import annotations

import re

from llm import BaseLLM
from REPL import REPL


class RLMError(RuntimeError):
    """Base exception for RLM control-flow failures."""


class RecursionLimitError(RLMError):
    """Raised when a model asks for another range after the depth limit."""


class CodeExecutionError(RLMError):
    """Raised when generated range-selection code cannot be executed."""


class RLM:
    """A synchronous RLM that recurses through host-controlled child calls.

    The first/root model call sees the query and metadata but not the full
    context. It can return a normal answer or a Python code block that defines
    ``start``, ``end``, and ``query``. The controller retrieves
    ``original_context[start:end]`` and passes that slice to the child model.
    The same REPL, and therefore the same original context and metadata, stays
    alive throughout the whole recursive request.
    """

    _code_block = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        name: str,
        model: BaseLLM,
        context: str | None = None,
        recursion_depth: int = 1,
    ) -> None:
        if recursion_depth < 0:
            raise ValueError("recursion_depth must be zero or greater")

        self.name = name
        self.model = model
        self.context = context
        self.recursion_depth = recursion_depth

    def call_rlm(self, query: str, context: str | None = None, metadata: str = "") -> str:
        """Answer ``query`` over ``context`` using bounded recursive subcalls.

        ``context`` supplied here becomes the immutable global context in the
        REPL for this top-level request. A recursively selected slice is only
        supplied to the child model; it never replaces that global context.
        """
        source_context = self.context if context is None else context
        if source_context is None:
            raise ValueError("context must be provided to the RLM constructor or call_rlm")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        repl = REPL(context=source_context, metadata=metadata)
        return self._call(query=query, repl=repl, selected_context=None, depth=0)

    def _call(self, query: str, repl: REPL, selected_context: str | None, depth: int) -> str:
        response = self.model.call_llm(
            self._build_prompt(
                query=query,
                metadata=repl.metadata,
                selected_context=selected_context,
                depth=depth,
            )
        )
        if not isinstance(response, str):
            raise RLMError("LLM call must return a string response")

        code = self._extract_code(response)
        if code is None:
            return self._final_answer(response)
        if depth >= self.recursion_depth:
            raise RecursionLimitError(
                f"RLM '{self.name}' reached recursion depth {self.recursion_depth} before a final answer."
            )

        execution = repl.execute_code(code)
        if not execution.succeeded or execution.selection is None:
            details = execution.error or "Generated code did not produce a context selection."
            raise CodeExecutionError(details)

        selection = execution.selection
        child_context = repl.context[selection.start : selection.end]
        return self._call(
            query=selection.query,
            repl=repl,
            selected_context=child_context,
            depth=depth + 1,
        )

    @classmethod
    def _extract_code(cls, response: str) -> str | None:
        blocks = cls._code_block.findall(response)
        if not blocks:
            return None
        if len(blocks) != 1:
            raise RLMError("An RLM response may contain at most one Python code block.")
        return blocks[0].strip()

    @staticmethod
    def _final_answer(response: str) -> str:
        match = re.fullmatch(r"\s*FINAL\((.*)\)\s*", response, flags=re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    @staticmethod
    def _build_prompt(
        query: str,
        metadata: str,
        selected_context: str | None,
        depth: int,
    ) -> str:
        instructions = """You are an RLM controller. Answer the user's query when you have enough evidence.

The original context is available only to generated local Python code as the read-only variable `context`. Its metadata is available as the read-only variable `metadata`. Generated code never calls a model.

If another agent should inspect a portion of the original context, respond with exactly one Python code block. That code may inspect `context` and `metadata`, but it must not assign to either. It must set:
start = <inclusive integer offset into context>
end = <exclusive integer offset into context>
query = <non-empty question for the child agent>

The host retrieves context[start:end] from the unchanged original context and invokes the child agent outside the interpreter. Do not create, return, or assign a replacement context value. If you can answer, return plain text or FINAL(answer), with no Python code block."""

        prompt = f"{instructions}\n\nQuery:\n{query}\n\nMetadata:\n{metadata}\n\nRecursion depth: {depth}"
        if selected_context is not None:
            prompt += f"\n\nSelected context for this child agent:\n{selected_context}"
        return prompt
