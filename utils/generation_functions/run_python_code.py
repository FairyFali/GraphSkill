from __future__ import annotations

import json
import tempfile
import subprocess
import importlib.util
import multiprocessing as mp
from typing import Any, Dict, List, Optional


def _load_module_from_source(code_str: str):
    """Load a one-off module from source string."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code_str)
        path = f.name
    spec = importlib.util.spec_from_file_location("user_module", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module, path


def _worker_call(queue, module_path: str, func_name: str, args, kwargs):
    """Run target function in a separate process and return (ok, payload)."""
    try:
        spec = importlib.util.spec_from_file_location("user_module", module_path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[assignment]
        func = getattr(module, func_name)
        result = func(*args, **kwargs)
        queue.put((True, result))
    except Exception as e:
        queue.put((False, e))


def _run_function_with_timeout(module_path: str, func_name: str, args, kwargs, timeout: float):
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker_call, args=(queue, module_path, func_name, args, kwargs))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(0.1)
        return "timeout", None
    if queue.empty():
        return "error", RuntimeError("No result returned from worker.")
    ok, payload = queue.get()
    return ("ok" if ok else "exception"), payload


def run_tests_io_or_call(
    code_str: str,
    test_cases: List[Dict[str, Any]],
    *,
    timeout: float = 3.0,
    python_executable: str = "python3",
    float_tol: Optional[float] = None,  # e.g., 1e-9 for approximate float compare
) -> List[Dict[str, Any]]:
    """
    Execute tests in either 'io' (stdin/stdout) or 'call' (function return) style.

    Case formats supported:
    1) IO mode (stdout):
        {
          "stdin": "<string>",
          "expect": {"stdout_exact": "<string>"}
        }

    2) Call mode (return/exception):
        {
          "call": {"function": "fname", "args": [...], "kwargs": {...}},
          "expect": {"return_equals": <any>}            # OR
          # "expect": {"raises": "ValueError"}          # exception by name
        }

    Returns per-case results:
        {
          "mode": "io" | "call",
          "passed": bool,
          "actual": <value or stdout>,
          "expected": <expected>,
          "error": Optional[str],
          "details": Optional[str]
        }
    """
    results = []
    module, module_path = _load_module_from_source(code_str)

    def equal(a, b):
        if float_tol is not None:
            try:
                # both numeric-like? try approximate
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return abs(float(a) - float(b)) <= float_tol
            except Exception:
                pass
        return a == b

    for case in test_cases:
        # --- Call mode ---
        if "call" in case:
            mode = "call"
            meta = case["call"]
            func_name = meta.get("function")
            args = meta.get("args", [])
            kwargs = meta.get("kwargs", {})
            exp = case.get("expect", {})
            exp_return = exp.get("return_equals", None)
            exp_raises = exp.get("raises", None)  # e.g., "ValueError"

            status, payload = _run_function_with_timeout(
                module_path, func_name, args, kwargs, timeout
            )

            if status == "timeout":
                results.append({
                    "mode": mode, "passed": False,
                    "actual": None, "expected": exp,
                    "error": f"Timeout after {timeout}s", "details": None
                })
                continue

            if exp_raises:
                # Expecting an exception by name
                if status == "exception":
                    passed = (type(payload).__name__ == exp_raises)
                    results.append({
                        "mode": mode, "passed": passed,
                        "actual": type(payload).__name__ if payload else None,
                        "expected": {"raises": exp_raises},
                        "error": None if passed else f"Raised {type(payload).__name__}, not {exp_raises}",
                        "details": str(payload) if payload else None
                    })
                else:
                    results.append({
                        "mode": mode, "passed": False,
                        "actual": payload, "expected": {"raises": exp_raises},
                        "error": "Expected exception, but function returned normally.",
                        "details": None
                    })
                continue

            # Expecting a normal return
            if status == "exception":
                results.append({
                    "mode": mode, "passed": False,
                    "actual": None, "expected": exp_return,
                    "error": f"Function raised {type(payload).__name__}",
                    "details": str(payload)
                })
                continue

            passed = equal(payload, exp_return)
            results.append({
                "mode": mode, "passed": passed,
                "actual": payload, "expected": exp_return,
                "error": None if passed else "Return value mismatch",
                "details": None
            })
            continue

        # --- IO mode (default) ---
        mode = "io"
        stdin_data = case.get("stdin", "")
        expected = case.get("expect", {}).get("stdout_exact", "")

        try:
            proc = subprocess.run(
                [python_executable, module_path],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            actual = proc.stdout.strip()
            passed = (actual == str(expected).strip())
            results.append({
                "mode": mode, "passed": passed,
                "actual": actual, "expected": expected,
                "error": None if passed else "Stdout mismatch",
                "details": proc.stderr.strip() or None
            })
        except subprocess.TimeoutExpired:
            results.append({
                "mode": mode, "passed": False,
                "actual": "", "expected": expected,
                "error": f"Timeout after {timeout}s", "details": None
            })
        except Exception as e:
            results.append({
                "mode": mode, "passed": False,
                "actual": "", "expected": expected,
                "error": str(e), "details": None
            })

    return results