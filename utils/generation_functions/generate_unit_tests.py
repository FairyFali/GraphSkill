from ..llm_agent.openai_code_generator import OpenAICodeGenerator
from ..llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from ..llm_agent.llama_code_generator import LlaMaGenerator
from ..llm_agent.opencoder_code_generator import OpenCoderGenerator
from typing import Any, List
from aux_dataclasses.run_results import MIN_CASES
from langchain.docstore.document import Document
from typing import Any, Dict, List, Optional, Callable
import json

class TestCaseFormatError(ValueError):
    pass

def _validate_cases(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate the LLM's JSON structure and return the cases list.
    Raises TestCaseFormatError if invalid.
    """
    if not isinstance(obj, dict):
        raise TestCaseFormatError("Top-level JSON must be an object.")
    if obj.get("type") != "io":
        raise TestCaseFormatError('Top-level "type" must be "io".')
    cases = obj.get("cases")
    if not isinstance(cases, list) or not cases:
        raise TestCaseFormatError('"cases" must be a non-empty list.')
    for i, c in enumerate[Any](cases):
        if not isinstance(c, dict):
            raise TestCaseFormatError(f"cases[{i}] must be an object.")
        if "stdin" not in c or not isinstance(c["stdin"], str):
            raise TestCaseFormatError(f'cases[{i}].stdin must be a string.')
        expect = c.get("expect")
        if not isinstance(expect, dict):
            raise TestCaseFormatError(f'cases[{i}].expect must be an object.')
        if "stdout_exact" not in expect or not isinstance(expect["stdout_exact"], str):
            raise TestCaseFormatError(
                f'cases[{i}].expect.stdout_exact must be a string.'
            )
    return cases

def generate_test_cases_with_llm(
    query: str,
    openai_model: OpenAICodeGenerator|DeepSeekCodeGenerator|LlaMaGenerator|OpenCoderGenerator,
    generated_code: str,
    instruction: str = (
    'You are a unit test generator. Output STRICT JSON only. Do not include code fences or commentary. Schema:\n'
    '{\n'
    '  "type": "call",\n'
    '  "cases": [\n'
    '    {"call": {"function": "<name>", "args": [...], "kwargs": {}},\n'
    '     "expect": {"return_equals": <json-serializable>}}\n'
    '    // OR\n'
    '    {"call": {"function": "<name>", "args": [...], "kwargs": {}},\n'
    '     "expect": {"raises": "<ExceptionName>"}}\n'
    '  ]\n'
    '}\n'
    "Rules:\n"
    "- Produce at least {min_cases} diverse cases covering typical inputs and edge cases.\n"
    "- For floats, use explicit numbers; avoid randomness and nondeterminism.\n"
    "- If using exceptions, use built-in exception names where applicable.\n"
    "- Do NOT include explanations or extra keys.").strip(), 
    max_repair_attempts: int = 3,
    ) -> List[Dict[str, Any]]:
    """
    If retrieved_docs is provided, append them to the user query as context before generation.
    """

    instruction = instruction.format(min_cases=MIN_CASES)
    prompt = (
        f"{instruction}\n\n"
        f"User Query:\n{query}\n\n"
        f"Generated Code:\n{generated_code}\n\n"
    )
    attempt = 0
    last_error: Optional[Exception] = None
    content: Optional[str] = None

    while attempt <= max_repair_attempts:
        if attempt == 0:
            content = openai_model.generate_code(prompt)
        else:
            # Ask the model to repair the previous output.
            repair_prompt = (
                "Your previous output was invalid. Return STRICT JSON matching the exact schema.\n"
                "Do not include commentary, markdown, or extra keys.\n"
                "Common mistakes to avoid:\n"
                "- trailing commas\n"
                "- missing quotes\n"
                "- wrong property names or nested shapes\n\n"
                "Return ONLY the corrected JSON now."
            )
            # Feed the broken JSON back to help self-repair
            broken_json = content or ""
            content = openai_model.generate_code(
                f"{repair_prompt}\n\nBroken JSON:\n{broken_json}"
            )

        try:
            data = json.loads(content or "")
            cases = _validate_cases(data)
            return cases
        except Exception as err:
            last_error = err
            attempt += 1

    # If we got here, all repair attempts failed.
    raise TestCaseFormatError(
        f"LLM did not produce valid test JSON after {max_repair_attempts+1} attempt(s): {last_error}"
    )