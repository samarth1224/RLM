# Recursive Language Model (RLM)

A Python library for Recursive Language Modeling that enables LLMs to process, inspect, and reason over arbitrary-length contexts via host-controlled sub-agent recursion.

---

## What is an RLM?

**Recursive Language Models (RLMs)** is an inference paradigm originally introduced by **Alex L. Zhang, Tim Kraska, and Omar Khattab (MIT CSAIL)** in the paper *"Recursive Language Models"*. 

Instead of forcing massive prompts into a model's limited attention window, an RLM treats long context as an external, queryable environment (such as a local Python REPL). The model programmatically explores the context, identifies relevant sections, and recursively delegates sub-tasks to process specific slices of information.

---

## How Our Implementation Differs

While inspired by the original RLM paradigm, this library adopts a different execution architecture:

| Aspect | Original RLM Paper | Our Implementation |
| :--- | :--- | :--- |
| **Model Invocation** | Generated Python code inside the REPL directly calls LLMs (e.g. `llm_query()`). | Generated code **never** invokes an LLM directly. The host controller manages all model calls. |
| **Code Generation Scope** | Code runs arbitrary sub-queries and handles sub-agent orchestration in-process. | Generated code only inspects context/metadata to set **coordinate ranges and child prompts** (`sub_calls`). |
| **Recursive Architecture** | Internal helper calls inside the interpreter. | **Object-Instantiated Sub-Agents**: The host instantiates independent `RLM` objects with dedicated context slices. |
| **Concurrency** | Sequential interpreter execution. | **Concurrent Async Dispatch**: Sub-agents can run in parallel via `asyncio.gather`. |
| **Result Synthesis** | Assembled inside the script execution. | The parent `RLM` gathers sub-agent outputs and invokes a dedicated synthesis step. |

---

## Quickstart

### Synchronous Usage

```python
from llm import GeminiLLM
from rlmagent import RLM

# 1. Initialize the LLM adapter
model = GeminiLLM(api_key="YOUR_GEMINI_API_KEY", model_name="gemini-3.6-flash")

# 2. Create an RLM instance
rlm = RLM(name="main-agent", model=model, recursion_depth=2)

# 3. Execute query over long context
answer = rlm.call_rlm(
    query="What were the total quarterly revenues across 2025?",
    context=long_document_text,
    metadata="Financial reports 2025",
)

print("Answer:", answer)
```

### Asynchronous Usage (Concurrent Sub-Agents)

```python
import asyncio
from llm import GeminiLLM
from rlmagent import RLM

async def main():
    model = GeminiLLM(api_key="YOUR_GEMINI_API_KEY")
    rlm = RLM(name="async-main-agent", model=model, recursion_depth=2)

    # Dispatches child sub-agents in parallel via asyncio.gather
    answer = await rlm.call_rlm_async(
        query="Summarize key findings across all chapters.",
        context=book_text,
    )
    print("Answer:", answer)

asyncio.run(main())
```

---

## Core Components

- **[`rlmagent.py`](rlmagent.py)**: The `RLM` controller class supporting synchronous (`call_rlm`) and asynchronous (`call_rlm_async`) recursive workflows.
- **[`REPL.py`](REPL.py)**: In-process Python interpreter that exposes immutable `context` and `metadata` to generated code and extracts `sub_calls` ranges.
- **[`llm.py`](llm.py)**: Provider abstraction layer with `BaseLLM` and `GeminiLLM` adapters.
