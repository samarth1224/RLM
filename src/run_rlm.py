"""Interactive runner for testing the local RLM library with Gemini.

Set ``GEMINI_API_KEY`` before running, or enter the key when prompted:

    python run_rlm.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from getpass import getpass
from pathlib import Path

from llm import GeminiLLM, LLMCallError
from rlmagent import RLM, RLMError


# This is an official OOLONG-synth validation shard used by the RLM paper's
# evaluation ecosystem. The runner reads one selected row at runtime instead
# of embedding the large context in this source file.
PREPOPULATED_DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "oolong-synth-validation-00004-of-00009.parquet"
)
PREPOPULATED_ROW_INDEX = 0

RLM_NAME = "local-rlm-test"
RECURSION_DEPTH = 1


def read_multiline(prompt: str) -> str:
    """Read text until the user enters END on a line by itself."""
    print(f"{prompt}\nEnter END on a line by itself when you are finished.")
    lines: list[str] = []
    while True:
        line = input()
        if line == "END":
            break
        lines.append(line)
    value = "\n".join(lines).strip()
    if not value:
        raise ValueError("A non-empty context is required.")
    return value


def read_runtime_input() -> tuple[str, str, str]:
    """Collect context, query, and optional metadata from the terminal."""
    context = read_multiline("Paste or type the context to analyze:")
    query = input("\nQuestion/prompt: ").strip()
    if not query:
        raise ValueError("A non-empty question/prompt is required.")
    metadata = input("Metadata (optional): ").strip()
    return context, query, metadata


def load_prepopulated_input() -> tuple[str, str, str]:
    """Load one OOLONG example without loading the whole Parquet dataset."""
    if not PREPOPULATED_DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Pre-populated dataset not found: {PREPOPULATED_DATA_FILE}"
        )

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "Loading the OOLONG pre-populated input requires pyarrow. "
            "Install it with: pip install pyarrow"
        ) from error

    parquet_file = parquet.ParquetFile(PREPOPULATED_DATA_FILE)
    if PREPOPULATED_ROW_INDEX < 0 or PREPOPULATED_ROW_INDEX >= parquet_file.metadata.num_rows:
        raise IndexError(
            f"PREPOPULATED_ROW_INDEX must be between 0 and "
            f"{parquet_file.metadata.num_rows - 1}"
        )

    row_group_index = 0
    row_offset = PREPOPULATED_ROW_INDEX
    while row_offset >= parquet_file.metadata.row_group(row_group_index).num_rows:
        row_offset -= parquet_file.metadata.row_group(row_group_index).num_rows
        row_group_index += 1

    row_group = parquet_file.read_row_group(
        row_group_index,
        columns=[
            "id",
            "context_len",
            "dataset",
            "context_window_text",
            "question",
            "task",
            "answer_type",
        ],
        use_threads=False,
    )

    def get_value(column: str):
        return row_group[column][row_offset].as_py()

    context = get_value("context_window_text")
    query = get_value("question")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("The selected OOLONG row has no context_window_text")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The selected OOLONG row has no question")

    metadata = json.dumps(
        {
            "dataset": "oolong-synth",
            "file": PREPOPULATED_DATA_FILE.name,
            "row_index": PREPOPULATED_ROW_INDEX,
            "id": get_value("id"),
            "context_len": get_value("context_len"),
            "task": get_value("task"),
            "answer_type": get_value("answer_type"),
        },
        ensure_ascii=False,
    )
    return context, query, metadata


def choose_input() -> tuple[str, str, str]:
    """Offer the required binary choice between stored and runtime input."""
    has_prepopulated_input = PREPOPULATED_DATA_FILE.is_file()
    if not has_prepopulated_input:
        return read_runtime_input()

    print("Choose input source:")
    print(f"1. Use the pre-populated OOLONG row ({PREPOPULATED_ROW_INDEX})")
    print("2. Provide context and prompt at runtime")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return load_prepopulated_input()
        if choice == "2":
            return read_runtime_input()
        print("Please enter exactly 1 or 2.")


def choose_mode() -> bool:
    """Ask the user whether to run in asynchronous (concurrent) or synchronous mode."""
    mode_env = os.getenv("RLM_MODE", "").strip().lower()
    if mode_env in ("async", "1", "true"):
        return True
    if mode_env in ("sync", "0", "false"):
        return False

    print("\nChoose execution mode:")
    print("1. Asynchronous (concurrent sub-agents) [Recommended]")
    print("2. Synchronous (sequential sub-agents)")
    while True:
        choice = input("Enter 1 or 2 (default 1): ").strip()
        if choice in ("1", ""):
            return True
        if choice == "2":
            return False
        print("Please enter 1 or 2.")


def get_api_key() -> str:
    """Use GEMINI_API_KEY when configured; otherwise request it privately."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    api_key = getpass("Gemini API key: ").strip()
    if not api_key:
        raise ValueError("A Gemini API key is required.")
    return api_key


async def async_main() -> int:
    try:
        context, query, metadata = choose_input()
        is_async = choose_mode()
        api_key = get_api_key()
        model_name = os.getenv("GEMINI_MODEL", GeminiLLM.DEFAULT_MODEL)

        model = GeminiLLM(api_key=api_key, model_name=model_name)
        rlm = RLM(
            name=RLM_NAME,
            model=model,
            recursion_depth=RECURSION_DEPTH,
        )

        mode_label = "Asynchronous (Concurrent)" if is_async else "Synchronous"
        print(f"\nRunning {RLM_NAME} with {model_name} in {mode_label} mode...\n")

        if is_async:
            answer = await rlm.call_rlm_async(query=query, context=context, metadata=metadata)
        else:
            answer = rlm.call_rlm(query=query, context=context, metadata=metadata)

        print("\nFinal Answer:\n")
        print(answer)
        return 0
    except (ImportError, LLMCallError, RLMError, ValueError) as error:
        print(f"\nRLM run failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nRLM run cancelled.", file=sys.stderr)
        return 130


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
