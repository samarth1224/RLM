"""Recursive Language Model controller.

The model may emit Python only to select context ranges for sub-agents. The
controller executes that code through ``REPL`` and instantiates child ``RLM``
objects to perform recursive model calls; generated code never invokes an LLM
directly.
"""

from __future__ import annotations

import asyncio
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
    """A synchronous and asynchronous RLM that recurses through child instances.

    The root model call sees the query and metadata, and can inspect the full
    context via the local REPL. It can return a normal answer or a Python code
    block that defines a list of sub-agent calls ``sub_calls = [{"start": ..., "end": ..., "query": ...}, ...]``.
    For each sub-call, the controller retrieves ``context[start:end]`` and
    instantiates a child ``RLM`` instance with that sliced context.
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

    def call_rlm(
        self,
        query: str,
        context: str | None = None,
        metadata: str = "",
    ) -> str:
        """Answer ``query`` over ``context`` using synchronous recursive sub-agents."""
        source_context = self.context if context is None else context
        if source_context is None:
            raise ValueError("context must be provided to the RLM constructor or call_rlm")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        print(f"\n{'='*20} [RLM: {self.name} (depth_budget={self.recursion_depth})] {'='*20}")
        print(f"[RLM: {self.name}] Query: {query}")
        print(f"[RLM: {self.name}] Context length: {len(source_context)} characters")

        repl = REPL(context=source_context, metadata=metadata)
        selected_context = self.context if self.context is not None and context is None else None
        if selected_context is not None:
            print(f"[RLM: {self.name}] Selected context length: {len(selected_context)} characters")

        answer = self._call_rlm(
            query=query,
            repl=repl,
            selected_context=selected_context,
        )
        print(f"[RLM: {self.name}] Completed.")
        return answer

    def _call_rlm(
        self,
        query: str,
        repl: REPL,
        selected_context: str | None,
    ) -> str:
        """Private synchronous execution loop: prompt building, model calling, sub-agent dispatch & synthesis."""
        prompt = self._build_prompt(
            query=query,
            metadata=repl.metadata,
            selected_context=selected_context,
            depth=self.recursion_depth,
        )

        print(f"[RLM: {self.name}] Calling LLM...")
        response = self.model.call_llm(prompt)
        if not isinstance(response, str):
            raise RLMError("LLM call must return a string response")

        print(f"[RLM: {self.name}] LLM response:\n{response.strip()}\n")

        code = self._extract_code(response)
        if code is None:
            answer = self._final_answer(response)
            print(f"[RLM: {self.name}] Final answer determined directly.")
            return answer

        if self.recursion_depth <= 0:
            print(f"[RLM: {self.name}] Recursion depth limit reached (0 remaining). Cannot spawn sub-agents.")
            raise RecursionLimitError(
                f"RLM '{self.name}' reached recursion depth limit before a final answer."
            )

        print(f"[RLM: {self.name}] Executing generated selection code in REPL:\n{code}")
        execution = repl.execute_code(code)
        if not execution.succeeded or not execution.selections:
            details = execution.error or "Generated code did not produce valid sub-agent selections."
            print(f"[RLM: {self.name}] Code execution failed: {details}")
            raise CodeExecutionError(details)

        selections = execution.selections
        print(f"[RLM: {self.name}] Code execution succeeded. Dispatching {len(selections)} sub-agent(s) sequentially:")

        sub_results: list[dict[str, str]] = []
        for i, sel in enumerate(selections, 1):
            print(
                f"[RLM: {self.name}] -> Sub-agent {i}/{len(selections)}: range [{sel.start}:{sel.end}], "
                f"query: {sel.query!r}"
            )
            child_context = repl.context[sel.start : sel.end]
            sub_agent = RLM(
                name=f"{self.name}_sub_{i}",
                model=self.model,
                context=child_context,
                recursion_depth=self.recursion_depth - 1,
            )
            child_answer = sub_agent.call_rlm(
                query=sel.query,
                metadata=repl.metadata,
            )
            print(f"[RLM: {self.name}] <- Sub-agent {i}/{len(selections)} returned result.")
            sub_results.append({"query": sel.query, "answer": child_answer})

        # If single sub-agent was invoked with identical query, return its answer directly
        if len(sub_results) == 1 and sub_results[0]["query"].strip() == query.strip():
            print(f"[RLM: {self.name}] Single sub-agent answered original query directly.")
            return sub_results[0]["answer"]

        # Otherwise synthesize responses from all sub-agents
        print(f"[RLM: {self.name}] Synthesizing final answer from {len(sub_results)} sub-agent result(s)...")
        synthesis_prompt = self._build_synthesis_prompt(
            query=query,
            metadata=repl.metadata,
            sub_results=sub_results,
        )
        print(f"[RLM: {self.name}] Calling LLM for synthesis...")
        synthesis_response = self.model.call_llm(synthesis_prompt)
        if not isinstance(synthesis_response, str):
            raise RLMError("LLM synthesis call must return a string response")

        print(f"[RLM: {self.name}] LLM synthesis response:\n{synthesis_response.strip()}\n")
        answer = self._final_answer(synthesis_response)
        print(f"[RLM: {self.name}] Final synthesized answer determined.")
        return answer

    async def call_rlm_async(
        self,
        query: str,
        context: str | None = None,
        metadata: str = "",
    ) -> str:
        """Answer ``query`` over ``context`` using asynchronous concurrent sub-agents."""
        source_context = self.context if context is None else context
        if source_context is None:
            raise ValueError("context must be provided to the RLM constructor or call_rlm_async")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        print(f"\n{'='*20} [RLM (Async): {self.name} (depth_budget={self.recursion_depth})] {'='*20}")
        print(f"[RLM: {self.name}] Query: {query}")
        print(f"[RLM: {self.name}] Context length: {len(source_context)} characters")

        repl = REPL(context=source_context, metadata=metadata)
        selected_context = self.context if self.context is not None and context is None else None
        if selected_context is not None:
            print(f"[RLM: {self.name}] Selected context length: {len(selected_context)} characters")

        answer = await self._call_rlm_async(
            query=query,
            repl=repl,
            selected_context=selected_context,
        )
        print(f"[RLM: {self.name}] Completed.")
        return answer

    async def _call_rlm_async(
        self,
        query: str,
        repl: REPL,
        selected_context: str | None,
    ) -> str:
        """Private asynchronous execution loop with concurrent sub-agent dispatch."""
        prompt = self._build_prompt(
            query=query,
            metadata=repl.metadata,
            selected_context=selected_context,
            depth=self.recursion_depth,
        )

        print(f"[RLM: {self.name}] Calling LLM asynchronously...")
        response = await self.model.call_llm_async(prompt)
        if not isinstance(response, str):
            raise RLMError("LLM call must return a string response")

        print(f"[RLM: {self.name}] LLM response:\n{response.strip()}\n")

        code = self._extract_code(response)
        if code is None:
            answer = self._final_answer(response)
            print(f"[RLM: {self.name}] Final answer determined directly.")
            return answer

        if self.recursion_depth <= 0:
            print(f"[RLM: {self.name}] Recursion depth limit reached (0 remaining). Cannot spawn sub-agents.")
            raise RecursionLimitError(
                f"RLM '{self.name}' reached recursion depth limit before a final answer."
            )

        print(f"[RLM: {self.name}] Executing generated selection code in REPL:\n{code}")
        execution = repl.execute_code(code)
        if not execution.succeeded or not execution.selections:
            details = execution.error or "Generated code did not produce valid sub-agent selections."
            print(f"[RLM: {self.name}] Code execution failed: {details}")
            raise CodeExecutionError(details)

        selections = execution.selections
        print(f"[RLM: {self.name}] Code execution succeeded. Dispatching {len(selections)} sub-agent(s) concurrently:")

        sub_agents: list[tuple[RLM, str]] = []
        for i, sel in enumerate(selections, 1):
            print(
                f"[RLM: {self.name}] -> Preparing Sub-agent {i}/{len(selections)}: range [{sel.start}:{sel.end}], "
                f"query: {sel.query!r}"
            )
            child_context = repl.context[sel.start : sel.end]
            sub_agent = RLM(
                name=f"{self.name}_sub_{i}",
                model=self.model,
                context=child_context,
                recursion_depth=self.recursion_depth - 1,
            )
            sub_agents.append((sub_agent, sel.query))

        # Launch all sub-agents concurrently using asyncio.gather
        tasks = [
            agent.call_rlm_async(query=sub_q, metadata=repl.metadata)
            for agent, sub_q in sub_agents
        ]
        answers = await asyncio.gather(*tasks)

        sub_results: list[dict[str, str]] = [
            {"query": sub_q, "answer": ans}
            for (_, sub_q), ans in zip(sub_agents, answers)
        ]

        # If single sub-agent was invoked with identical query, return its answer directly
        if len(sub_results) == 1 and sub_results[0]["query"].strip() == query.strip():
            print(f"[RLM: {self.name}] Single sub-agent answered original query directly.")
            return sub_results[0]["answer"]

        # Otherwise synthesize responses from all sub-agents
        print(f"[RLM: {self.name}] Synthesizing final answer from {len(sub_results)} sub-agent result(s)...")
        synthesis_prompt = self._build_synthesis_prompt(
            query=query,
            metadata=repl.metadata,
            sub_results=sub_results,
        )
        print(f"[RLM: {self.name}] Calling LLM for synthesis (async)...")
        synthesis_response = await self.model.call_llm_async(synthesis_prompt)
        if not isinstance(synthesis_response, str):
            raise RLMError("LLM synthesis call must return a string response")

        print(f"[RLM: {self.name}] LLM synthesis response:\n{synthesis_response.strip()}\n")
        answer = self._final_answer(synthesis_response)
        print(f"[RLM: {self.name}] Final synthesized answer determined.")
        return answer

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
        instructions = """You are an RLM (Recursive Language Model) controller. Answer the user's query when you have enough evidence.

The original context is available to generated local Python code as the read-only variable `context`. Its metadata is available as the read-only variable `metadata`. Generated code never calls a model directly.

If you need one or more sub-agents to inspect portions of the context, respond with exactly one Python code block. That code may inspect `context` and `metadata`, but it must not assign to either. It must set `sub_calls` as a list of dictionaries, where each dictionary specifies:
- "start": <inclusive integer offset into context>
- "end": <exclusive integer offset into context>
- "query": <non-empty question/prompt for the sub-agent>

The host retrieves context[start:end] for each item, instantiates a child sub-agent for each range, and passes the selected slice and query to that sub-agent. Do not create, return, or assign a replacement context variable.

If you have enough information to answer directly, do NOT output Python code. Return plain text or FINAL(answer).

Few-shot examples:

Example 1 — locate a relevant section and spawn a single child agent:
```python
marker = context.find("quarterly revenue")
sub_calls = [
    {
        "start": max(0, marker - 300),
        "end": min(len(context), marker + 1200),
        "query": "Extract the quarterly revenue figures and explain the trend."
    }
]
```

Example 2 — spawn multiple sub-agents to inspect different sections:
```python
pos1 = context.find("Section 1: Methodology")
pos2 = context.find("Section 2: Results")
pos3 = context.find("Section 3: Discussion")
sub_calls = [
    {
        "start": pos1,
        "end": pos2,
        "query": "Summarize the methodology used in the study."
    },
    {
        "start": pos2,
        "end": pos3 if pos3 != -1 else len(context),
        "query": "Extract all numerical results and performance metrics."
    }
]
```

Example 3 — valid direct answer when no further context selection is needed:
```text
The answer is that revenue increased by 15% in Q3.
```

Invalid example — never replace the original context or assign to context:
```python
context = context[100:500]
sub_calls = [{"start": 0, "end": len(context), "query": "Analyze"}]
```

For selection, output only the `sub_calls` list assignment inside one Python code block. The host controller, not your code, handles slicing and invoking the sub-agents."""

        prompt = f"{instructions}\n\nQuery:\n{query}\n\nMetadata:\n{metadata}\n\nRecursion depth available: {depth}"
        if selected_context is not None:
            prompt += f"\n\nSelected context for this agent:\n{selected_context}"
        return prompt

    @staticmethod
    def _build_synthesis_prompt(
        query: str,
        metadata: str,
        sub_results: list[dict[str, str]],
    ) -> str:
        results_formatted = []
        for i, res in enumerate(sub_results, 1):
            results_formatted.append(
                f"Sub-agent {i} Task: {res['query']}\nSub-agent {i} Output: {res['answer']}"
            )
        combined_results = "\n\n".join(results_formatted)

        return (
            f"You are the main RLM controller. You previously dispatched sub-agents to inspect context segments.\n\n"
            f"Original Query:\n{query}\n\n"
            f"Metadata:\n{metadata}\n\n"
            f"Sub-agent Results:\n{combined_results}\n\n"
            f"Using the sub-agent results above, provide the final, complete answer to the original query. "
            f"Return plain text or FINAL(answer), with no Python code blocks."
        )
