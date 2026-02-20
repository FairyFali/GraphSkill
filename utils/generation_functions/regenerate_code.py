from __future__ import annotations

import os
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Callable
from ..llm_agent.openai_code_generator import OpenAICodeGenerator
from ..llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from ..llm_agent.llama_code_generator import LlaMaGenerator
from ..llm_agent.opencoder_code_generator import OpenCoderGenerator

# Assumes run_tests_io_or_call is imported from your snippet above.
from run_python_code import run_tests_io_or_call

class RegenerationError(RuntimeError):
    pass


def _summarize_failures_for_prompt(
    results: List[Dict[str, Any]],
    *,
    limit_per_kind: int = 6,
    max_value_len: int = 300,
) -> str:
    """
    Build a compact summary of failing cases for the LLM prompt.
    Truncates long values and groups by mode.
    """
    def trunc(x):
        s = json.dumps(x, ensure_ascii=False)
        if len(s) > max_value_len:
            return s[: max_value_len - 3] + "..."
        return s

    io_fail = []
    call_fail = []
    for r in results:
        if r.get("passed", False):
            continue
        mode = r.get("mode")
        if mode == "io":
            io_fail.append(
                {
                    "stdin": r.get("stdin", ""),
                    "expected": r.get("expected", ""),
                    "actual": r.get("actual", ""),
                    "error": r.get("error", None),
                    "details": r.get("details", None),
                }
            )
        elif mode == "call":
            call_fail.append(
                {
                    "expected": r.get("expected", None),
                    "actual": r.get("actual", None),
                    "error": r.get("error", None),
                    "details": r.get("details", None),
                }
            )

    parts = []
    if io_fail:
        parts.append("IO FAILURES:")
        for item in io_fail[:limit_per_kind]:
            parts.append(
                f"- stdin={trunc(item['stdin'])} | expected={trunc(item['expected'])} | "
                f"actual={trunc(item['actual'])} | error={trunc(item['error'])} | details={trunc(item['details'])}"
            )
    if call_fail:
        parts.append("CALL FAILURES:")
        for item in call_fail[:limit_per_kind]:
            parts.append(
                f"- expected={trunc(item['expected'])} | actual={trunc(item['actual'])} | "
                f"error={trunc(item['error'])} | details={trunc(item['details'])}"
            )
    return "\n".join(parts) if parts else "No failures."



def _strip_code_fences(s: str) -> str:
    """
    Remove common Markdown fences if the model includes them despite instructions.
    """
    s = s.strip()
    # Triple backticks variants
    if s.startswith("```"):
        # Remove first line fence
        lines = s.splitlines()
        # Drop first line
        lines = lines[1:]
        # If ending fence present, drop it
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def regenerate_graph_code_with_llm(
    *,
    problem_brief: str,
    initial_code: str,
    test_cases: List[Dict[str, Any]],
    docs_context: Optional[str] = None,     # e.g., NetworkX doc snippets you retrieved
    max_iters: int = 5,
    timeout: float = 3.0,
    python_executable: str = "python3",
    float_tol: Optional[float] = None,
    model: str = "gpt-4o-mini",
    system_preamble: str = (
        "You are a senior Python engineer specializing in graph algorithms and NetworkX. "
        "Fix bugs with minimal, correct changes. Prefer clean, deterministic solutions. "
        "When IO is required, read from stdin and write to stdout exactly as specified. "
        "When functions are expected, implement pure functions with clear signatures. "
        "Avoid randomness and side effects. Use only standard library and NetworkX unless specified."
    ),
    # Custom LLM caller: fn(prompt:str, model:str, system:str)->str
    openai_model: OpenAICodeGenerator|DeepSeekCodeGenerator|LlaMaGenerator|OpenCoderGenerator,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Iteratively regenerate (repair) Python code for a graph task until tests pass or attempts are exhausted.

    Parameters
    ----------
    problem_brief : str
        Natural language spec of the task (IO and/or function contracts).
    initial_code : str
        The starting Python module source.
    test_cases : list
        Cases compatible with run_tests_io_or_call (both IO and CALL schemas).
    docs_context : Optional[str]
        Optional retrieval/context (e.g., NetworkX doc snippets) to guide the model.
    max_iters : int
        Max repair attempts (including the initial check).
    timeout, python_executable, float_tol :
        Passed through to run_tests_io_or_call.
    model, system_preamble :
        LLM configuration.
    caller :
        Custom LLM caller. If None, uses a default OpenAI-style caller.

    Returns
    -------
    (final_code_str, last_results)

    Raises
    ------
    RegenerationError if no valid code passes after max_iters.
    """
    code = initial_code
    last_results: List[Dict[str, Any]] = []

    for _ in range(max_iters):
        # 1) Evaluate current code
        last_results = run_tests_io_or_call(
            code,
            test_cases,
            timeout=timeout,
            python_executable=python_executable,
            float_tol=float_tol,
        )
        passed = sum(1 for r in last_results if r.get("passed"))
        total = len(last_results)
        if passed == total:
            return code, last_results  # success 🎉

        # 2) Build concise failure report
        failure_summary = _summarize_failures_for_prompt(last_results)

        # 3) Ask the LLM to repair the code
        #    We instruct: return a *complete* Python module, no commentary, no fences.
        instruction = textwrap.dedent(f"""
        TASK:
        You will repair a Python module that solves a graph algorithm task.

        REQUIREMENTS:
        - Return ONLY the full corrected Python source code (**no explanations**).
        - Ensure the module satisfies ALL tests, including both IO-mode and function-call mode cases.
        - Deterministic behavior. No randomness. No extra prints or logging.
        - If IO cases exist, parse stdin exactly as implied by the tests and print exact expected formats.
        - If function-call cases exist, implement the requested functions with correct signatures/returns.
        - Prefer linear or near-linear solutions where appropriate; use NetworkX when suitable and allowed.
        - Handle directed/undirected and weighted/unweighted graphs per the {{"problem_brief"}}.
        - Avoid external dependencies beyond the standard library and NetworkX.

        PROBLEM BRIEF:
        {problem_brief.strip()}

        {"REFERENCE DOCS / CONTEXT:\n" + docs_context.strip() if docs_context else ""}

        FAILURES TO FIX (examples):
        {failure_summary}

        CURRENT CODE (to fix):
        <<CODE>>
        {code}
        <<ENDCODE>>

        OUTPUT:
        Return ONLY the corrected full Python module source. Do NOT include backticks or commentary.
        """).strip()

        raw = openai_model.generate_code(instruction)
        candidate = _strip_code_fences(raw)

        # Quick sanity check: produce non-empty Python
        if not candidate or "import" not in candidate and "def " not in candidate:
            # If the model ignored instructions, try once more by hard-reminding.
            hard_reminder = (
                "Return ONLY the corrected Python source code. No markdown, no explanations.\n\n" + instruction
            )
            candidate = _strip_code_fences(openai_model.generate_code(hard_reminder))

        # Update code and continue loop
        code = candidate

    # Final eval (last_results already from last loop)
    raise RegenerationError(
        f"Code regeneration did not pass all tests after {max_iters} attempt(s). "
        f"Last pass rate: {sum(1 for r in last_results if r.get('passed'))}/{len(last_results)}"
    )

