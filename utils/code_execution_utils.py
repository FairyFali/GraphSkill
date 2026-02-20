"""
Utility functions for code generation, execution, and evaluation.

This module provides reusable functions for:
- Generating code with LLM models
- Extracting code from LLM responses
- Executing generated code with AST parsing
- Timeout-protected execution using multiprocessing
- Type-aware result comparison

Used by:
- runners/complexgraph/run_zs_coding.py (zero-shot code generation)
- runners/complexgraph/run_fs_coding.py (few-shot code generation)
"""

import re
import ast
import traceback
import types
import textwrap
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Union

# Type alias for LLM code generators
# These are the supported model types that can generate code
try:
    from llm_agent.openai_code_generator import OpenAICodeGenerator
    from llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
    from llm_agent.llama_code_generator import LlaMaGenerator
    from llm_agent.opencoder_code_generator import OpenCoderGenerator
    from llm_agent.qwen_code_generator import QwenGenerator

    LLMCodeGenerator = Union[
        OpenAICodeGenerator,
        DeepSeekCodeGenerator,
        LlaMaGenerator,
        OpenCoderGenerator,
        QwenGenerator
    ]
except ImportError:
    # Fallback if imports fail (for standalone testing)
    LLMCodeGenerator = Any


def generate_code_with_llm(
    query: str,
    llm_model: LLMCodeGenerator,
    retrieved_docs: Optional[Union[List[str], str]] = None,
    test_case: str = "",
    instruction: str = "You are a sophisticated AI expert in graph theory and algorithms. "
                      "Solve the following graph problem by writing code without any explanation.\n"
) -> str:
    """
    Generate Python code using an LLM model with optional documentation context.

    This function constructs a prompt for code generation that can include:
    - Custom instructions
    - The user's task query
    - Retrieved documentation (optional, for RAG-based approaches)
    - Test cases (optional)

    The prompt construction varies based on whether documentation is provided:
    - With docs: Encourages using the provided documentation as reference
    - Without docs: Direct code generation from the query

    Args:
        query: The task description or user query for code generation
            Example: "Given an undirected graph, find the shortest path between two nodes"

        llm_model: The LLM code generator instance (OpenAI, DeepSeek, Llama, OpenCoder)
            Must have a .generate() method that accepts a prompt string

        retrieved_docs: Optional documentation to include as context
            - List[str]: Multiple documentation snippets (will be joined)
            - str: Single documentation string
            - None: No documentation (direct code generation)

        test_case: Optional test case to include in the prompt
            If provided, instructs LLM to test the code and print output

        instruction: System instruction for the LLM
            Default: Instructs to solve graph problems without explanation

    Returns:
        Generated Python code as a string (may include markdown formatting)

    Example:
        >>> from utils.get_llm_response_generator import create_code_generator
        >>> llm = create_code_generator("gpt-4")
        >>> code = generate_code_with_llm(
        ...     query="Find shortest path in a graph",
        ...     llm_model=llm,
        ...     retrieved_docs="NetworkX shortest_path documentation...",
        ...     test_case="graph = [[1,2],[2,3]]; source=1; target=3"
        ... )
        >>> print(code)
        ```python
        def shortest_path(edge_list, source, target):
            ...
        ```

    Note:
        - The function prints a message when generating without documentation
        - The returned code may need to be extracted from markdown blocks
        - Use extract_code_from_response() to clean the output
    """
    # Build prompt based on whether documentation is provided
    if retrieved_docs:
        # Case 1: Retrieved documentation is a list of strings
        if isinstance(retrieved_docs, list):
            # Join multiple documentation snippets with blank lines
            context_snippets = "\n\n".join([doc for doc in retrieved_docs])

            prompt = (
                f"{instruction}\n\n"
                f"User Query:\n{query}\n\n"
                f"Retrieved Documentation:\n{context_snippets}\n\n"
            )

            # Add test case section if provided
            if test_case:
                prompt += f"Test case: {test_case}\n\n"

            # Add task instructions for documentation-based generation
            prompt += (
                "Task:\n"
                "Write Python code that solves the user query.\n"
                "Read the Retrieved Documentation carefully. "
                "Rely primarily on the retrieved documentation as your reference.\n"
                "You are NOT allowed to use any NetworkX functions, modules, or APIs "
                "that do NOT appear in the Retrieved Documentation.\n"
                "Do NOT rely on prior knowledge, memory, or assumptions about NetworkX.\n"
            )

            # Add test case execution instruction if test case was provided
            if test_case:
                prompt += (
                    "You must use the given test case for testing your Python function, "
                    "and *print* out the output.\n"
                )

            prompt += (
                "Your final output should be correct, efficient, and consistent with "
                "the documentation whenever possible.\n"
            )

        # Case 2: Retrieved documentation is a single string
        else:  # isinstance(retrieved_docs, str)
            prompt = (
                f"{instruction}\n\n"
                f"User Query:\n{query}\n\n"
                f"Retrieved Documentation:\n{retrieved_docs}\n\n"
            )

            # Add test case section if provided
            if test_case:
                prompt += f"Test case: {test_case}\n\n"

            # Add task instructions for documentation-based generation
            prompt += (
                "Task:\n"
                "Write Python code that solves the user query.\n"
                "Read the Retrieved Documentation carefully. "
                "Rely primarily on the retrieved documentation as your reference.\n"
            )

            # Add test case execution instruction if test case was provided
            if test_case:
                prompt += (
                    "You must use the given test case for testing your Python function, "
                    "and *print* out the output.\n"
                )

            prompt += (
                "Your final output should be correct, efficient, and consistent with "
                "the documentation whenever possible.\n"
            )

    else:
        # Case 3: No documentation - direct code generation
        # print("No documentation provided, generating code without RAG")

        # prompt = (
        #     f"{instruction}\n\n"
        #     f"User Query:\n{query}\n\n"
        #     # "Task:\n"
        #     # "Write Python code that solves the user query.\n"
        # )
        prompt = query

        # Note: Test case instructions are commented out for non-RAG mode
        # This gives the LLM more freedom in the implementation

    # Generate code using the LLM model
    generated_code = llm_model.generate(prompt)

    return generated_code


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.

    The function searches for code blocks in the following order:
    1. ```python ... ``` blocks (case-insensitive)
    2. ``` ... ``` blocks (generic code blocks)
    3. If no markers found, returns the entire response

    Args:
        response: Raw LLM response text that may contain code blocks

    Returns:
        Extracted Python code as a string

    Example:
        >>> response = "Here's the code:\n```python\ndef foo():\n    pass\n```"
        >>> extract_code_from_response(response)
        'def foo():\n    pass'
    """
    # Try to find code within ```python ... ``` blocks (case-insensitive)
    # The (?i:python) makes the 'python' keyword case-insensitive
    python_pattern = r'```(?i:python)\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    if matches:
        # Take the first match and strip whitespace
        return matches[0].strip()

    # Try to find code within generic ``` ... ``` blocks
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code blocks found, assume the entire response is code
    # This handles cases where LLM doesn't use markdown formatting
    return response.strip()


def execute_code_with_ast(code: str, edge_list: List[List], args: Optional[Dict[str, Any]]) -> Any:
    """
    Execute generated code by discovering and calling the defined function.

    This approach is more robust than executing code and looking for result variables.
    It uses AST (Abstract Syntax Tree) to discover function definitions, then calls
    the discovered function with the provided arguments.

    Process:
    1. Extract code from markdown blocks (handle ```python formatting)
    2. Parse code with AST to discover top-level function names
    3. Execute code in a fresh module namespace to avoid pollution
    4. Call the discovered function with the graph and arguments
    5. Return the function's result

    Args:
        code: Python code containing function definition(s)
        edge_list: Graph structure as list of edges
            - Weighted: [[src, tgt, weight], ...]
            - Unweighted: [[src, tgt], ...]
        args: Task-specific arguments (e.g., {'node': 5, 'source': 1, 'target': 10})

    Returns:
        The result from calling the discovered function

    Raises:
        ValueError: If no top-level functions are found in the code
        Exception: If code execution or function call fails

    Example:
        >>> code = "def shortest_path(edge_list, source, target):\\n    return 5"
        >>> execute_code_with_ast(code, [[1,2],[2,3]], {'source': 1, 'target': 3})
        5
    """
    # Step 1: Clean up the code by removing escape sequences
    # The textwrap.dedent removes common leading whitespace
    code_cleaned = textwrap.dedent(code.replace(r"\n", "\n"))

    # Step 2: Parse code with AST to discover all top-level function definitions
    # This allows us to find function names without executing the code first
    try:
        parsed = ast.parse(code_cleaned)
    except SyntaxError as e:
        raise ValueError(f"Code has syntax errors: {str(e)}")

    # Extract names of all top-level function definitions
    # We only look at top-level (not nested functions)
    func_names = [
        node.name for node in parsed.body
        if isinstance(node, ast.FunctionDef)
    ]

    if not func_names:
        raise ValueError("No top-level functions found in generated code")

    # For our use case, we expect exactly one function
    # If multiple functions exist, we use the first one
    target_func_name = func_names[0]
    print(f"    Discovered function: {target_func_name}")

    # Step 3: Execute the code in a fresh module namespace
    # Using types.ModuleType creates an isolated namespace
    # This prevents pollution of global namespace
    module = types.ModuleType("dynamic_code")
    try:
        exec(code_cleaned, module.__dict__)
    except Exception as e:
        raise Exception(f"Code execution failed during module loading: {str(e)}\n{traceback.format_exc()}")

    # Step 4: Retrieve the discovered function from the module
    target_function = getattr(module, target_func_name)

    # Step 5: Prepare arguments for the function call
    # Build function arguments list in the correct order
    function_args = [edge_list]  # First argument is always the edge list

    # Add task-specific arguments if they exist
    # The order matters here - we add them in the order they appear in the args dict
    if args:
        # For tasks with specific argument names (e.g., source, target, node)
        function_args.extend(args.values())

    # Step 6: Call the function and return result
    try:
        result = target_function(*function_args)
        return result
    except Exception as e:
        raise Exception(f"Function call failed: {str(e)}\n{traceback.format_exc()}")


def _worker_process(queue: mp.Queue, fn, code: str, edge_list: List[List], args: Optional[Dict[str, Any]]):
    """
    Worker function that runs in a separate process for timeout execution.

    This function is used by multiprocessing to execute code in isolation.
    It catches all exceptions and returns them via the queue.

    Args:
        queue: Multiprocessing queue to send results back
        fn: Function to execute (execute_code_with_ast)
        code: Python code to execute
        edge_list: Graph edge list
        args: Task arguments

    Queue Returns:
        ("ok", result): On successful execution
        ("err", (exception_name, message, traceback)): On error
    """
    try:
        result = fn(code, edge_list, args)
        queue.put(("ok", result))
    except Exception as e:
        # Package exception information for the parent process
        queue.put(("err", (e.__class__.__name__, str(e), traceback.format_exc())))


def execute_code_with_timeout(
    code: str,
    edge_list: List[List],
    args: Optional[Dict[str, Any]],
    timeout_seconds: int = 5
) -> Any:
    """
    Execute generated code with a timeout limit using multiprocessing.

    This prevents infinite loops or very slow algorithms from hanging the evaluation.
    Code is executed in a separate process that can be terminated if it exceeds
    the timeout.

    Process:
    1. Create a multiprocessing Queue for communication
    2. Spawn a new process to run the code
    3. Wait for the process to complete or timeout
    4. If timeout occurs, terminate the process
    5. Return the result or raise the appropriate exception

    Args:
        code: Python code to execute
        edge_list: Graph structure as edge list
        args: Task-specific arguments
        timeout_seconds: Maximum execution time in seconds (default: 5)

    Returns:
        Result from code execution

    Raises:
        TimeoutError: If execution exceeds timeout_seconds
        RuntimeError: If child process crashes without returning result
        Exception: Re-raises exceptions from the child process with traceback

    Example:
        >>> code = "def is_connected(edge_list): return True"
        >>> result = execute_code_with_timeout(code, [[1,2],[2,3]], None, timeout_seconds=5)
        >>> result
        True
    """
    # Create a queue for inter-process communication
    queue = mp.Queue()

    # Spawn a new process to execute the code
    # target: function to run in the new process
    # args: arguments to pass to the target function
    process = mp.Process(
        target=_worker_process,
        args=(queue, execute_code_with_ast, code, edge_list, args)
    )

    # Start the process
    process.start()

    # Wait for process to complete, with timeout
    # join() blocks until process terminates or timeout expires
    process.join(timeout_seconds)

    # Check if process is still running after timeout
    if process.is_alive():
        print(f"    ⚠ Code execution timed out after {timeout_seconds}s, terminating...")
        # Forcefully terminate the process
        process.terminate()
        # Give it 1 second to clean up
        process.join(1)
        raise TimeoutError(f"Code execution timed out after {timeout_seconds} seconds")

    # Check if the process posted a result to the queue
    if queue.empty():
        # Process exited without putting anything in the queue
        # This usually means it crashed hard
        raise RuntimeError("Child process exited without returning a result (likely crashed)")

    # Retrieve the result from the queue
    status, payload = queue.get()

    if status == "ok":
        # Successful execution, return the result
        return payload
    else:
        # Execution failed, unpack exception information
        exception_name, message, tb = payload
        # Re-raise with original traceback for debugging
        raise RuntimeError(f"{exception_name}: {message}\n{tb}")


def compare_results_type_aware(predicted, ground_truth) -> Tuple[bool, str]:
    """
    Compare predicted result with ground truth using type-aware comparison rules.

    This function handles different data types appropriately:
    - Integers and booleans: Direct equality
    - Floats: Rounded to 2 decimal places for comparison
    - Lists: Sorted comparison (order doesn't matter for most graph tasks)
    - Mixed int/float: Absolute difference < 1
    - None and empty list: Special equivalence
    - Tuples: Extract first element for some tasks

    Args:
        predicted: Result from code execution
        ground_truth: Expected correct answer

    Returns:
        Tuple[bool, str]: (is_correct, reason)
            - is_correct: Whether the prediction matches ground truth
            - reason: Explanation of the comparison result

    Examples:
        >>> compare_results_type_aware(5, 5)
        (True, "Exact match (int/bool)")

        >>> compare_results_type_aware(3.14159, 3.14)
        (True, "Match after rounding to 2 decimals")

        >>> compare_results_type_aware([1, 2, 3], [3, 1, 2])
        (True, "Match after sorting")

        >>> compare_results_type_aware(5, 5.0)
        (True, "Match with type conversion (int/float difference < 1)")
    """
    # Case 1: Both are integers or both are booleans - direct comparison
    if (isinstance(ground_truth, int) and isinstance(predicted, int)) or \
       (isinstance(ground_truth, bool) and isinstance(predicted, bool)):
        is_correct = str(ground_truth) == str(predicted)
        reason = "Exact match (int/bool)" if is_correct else f"Mismatch: {predicted} != {ground_truth}"
        return is_correct, reason

    # Case 2: Both are floats - compare with rounding to 2 decimal places
    # This handles floating-point precision issues
    elif isinstance(ground_truth, float) and isinstance(predicted, float):
        is_correct = round(ground_truth, 2) == round(predicted, 2)
        reason = "Match after rounding to 2 decimals" if is_correct else \
                 f"Mismatch: {round(predicted, 2)} != {round(ground_truth, 2)}"
        return is_correct, reason

    # Case 3: Both are lists - compare after sorting
    # Order doesn't matter for most graph tasks (e.g., list of nodes)
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        is_correct = sorted(ground_truth) == sorted(predicted)
        reason = "Match after sorting" if is_correct else \
                 f"Mismatch: {sorted(predicted)} != {sorted(ground_truth)}"
        return is_correct, reason

    # Case 4: Mixed int and float - check if difference is small
    # Handles cases where result is 5.0 but expected is 5
    elif (isinstance(ground_truth, float) and isinstance(predicted, int)) or \
         (isinstance(ground_truth, int) and isinstance(predicted, float)):
        is_correct = abs(ground_truth - predicted) < 1
        reason = "Match with type conversion (int/float difference < 1)" if is_correct else \
                 f"Mismatch: |{predicted} - {ground_truth}| >= 1"
        return is_correct, reason

    # Case 5: Same types - direct comparison
    elif type(ground_truth) == type(predicted):
        is_correct = ground_truth == predicted
        reason = f"Exact match ({type(ground_truth).__name__})" if is_correct else \
                 f"Mismatch: {predicted} != {ground_truth}"
        return is_correct, reason

    # Case 6: None vs empty list - special equivalence
    # Some tasks return None for empty results, others return []
    elif ground_truth is None and isinstance(predicted, list):
        is_correct = len(predicted) == 0
        reason = "Match (None equivalent to empty list)" if is_correct else \
                 f"Mismatch: expected None, got non-empty list {predicted}"
        return is_correct, reason

    # Case 7: Tuple result vs float ground truth
    # Some algorithms return (value, extra_info) tuples
    elif isinstance(predicted, tuple) and isinstance(ground_truth, float):
        is_correct = abs(predicted[0] - ground_truth) < 1
        reason = "Match (extracted first element from tuple)" if is_correct else \
                 f"Mismatch: |{predicted[0]} - {ground_truth}| >= 1"
        return is_correct, reason

    # Case 8: Type mismatch - cannot compare
    else:
        return False, f"Type mismatch: predicted is {type(predicted).__name__}, " \
                     f"ground_truth is {type(ground_truth).__name__}"
