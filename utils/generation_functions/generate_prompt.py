"""
Backward compatibility wrapper for generate_code_with_openai.

DEPRECATED: This module is maintained for backward compatibility with existing scripts.
New code should import from utils.code_execution_utils instead:

    from utils.code_execution_utils import generate_code_with_llm

The new function has:
- Better documentation
- Cleaner code structure
- Optional parameters properly marked
- Type hints
"""

from ..llm_agent.openai_code_generator import OpenAICodeGenerator
from ..llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from ..llm_agent.llama_code_generator import LlaMaGenerator
from ..llm_agent.opencoder_code_generator import OpenCoderGenerator
from typing import List, Union, Optional

# Import the new implementation
from ..code_execution_utils import generate_code_with_llm


def generate_code_with_openai(
    query: str,
    openai_model: Union[OpenAICodeGenerator, DeepSeekCodeGenerator, LlaMaGenerator, OpenCoderGenerator],
    retrieved_docs: Optional[Union[List[str], str]] = None,
    test_case: str = "",
    instruction: str = 'You are a sophisticated AI expert in graph theory and algorithms. Solve the following graph problem by writing code without any explanation. \n',
) -> str:
    """
    DEPRECATED: Use utils.code_execution_utils.generate_code_with_llm instead.

    Wrapper function for backward compatibility with existing scripts.
    Delegates to the new implementation in code_execution_utils.

    Args:
        query: The task description or user query
        openai_model: LLM code generator instance
        retrieved_docs: Optional documentation for RAG
        test_case: Optional test case string
        instruction: System instruction for the LLM

    Returns:
        Generated Python code as a string
    """
    # Delegate to the new implementation
    return generate_code_with_llm(
        query=query,
        llm_model=openai_model,
        retrieved_docs=retrieved_docs,
        test_case=test_case,
        instruction=instruction
    )