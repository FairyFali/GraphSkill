from ..llm_agent.openai_code_generator import OpenAICodeGenerator
from ..llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from ..llm_agent.llama_code_generator import LlaMaGenerator
from ..llm_agent.opencoder_code_generator import OpenCoderGenerator
from typing import List
from langchain.docstore.document import Document
import re

def generate_corrected_code_with_openai(
    query: str,
    openai_model: OpenAICodeGenerator|DeepSeekCodeGenerator|LlaMaGenerator|OpenCoderGenerator,
    retrieved_docs: List[Document]|str,
    error_code: str, 
    test_case: str,
    instruction: str = 'You are a highly skilled Python debugging expert specialized in graph algorithms and the NetworkX library.\n', 
    error_output:str = None,
) -> str:
    """
    If retrieved_docs is provided, append them to the user query as context before generation.
    """

    if retrieved_docs and isinstance(retrieved_docs,list):
        context_snippets = "\n\n".join([doc for doc in retrieved_docs])
        retrieved_doc_prompt = f"Retrieved Documentation:\n{context_snippets}\n\n"

    elif retrieved_docs and isinstance(retrieved_docs,str):
        retrieved_doc_prompt = f"Retrieved Documentation:\n{retrieved_docs}\n\n"
    else:
        print("No related file detected, generating code without RAG")
        # prompt = (f"{instrcution}", query)
        retrieved_doc_prompt = ""
    pattern = r"^(?:.*\n)*?([\w]+Error): (.*)$"
    match = re.search(pattern, error_output, re.MULTILINE)
    if match:
        print("Error Matches")
        prompt = (
            f"{instruction}\n\n"
            f"User Query:\n{query}\n\n"
            f"{retrieved_doc_prompt}\n"
            f"The code snippet to be debugged:\n{error_code}\n"
            f"The error message:\n{error_output}\n"
            f"Test case: {test_case}"
            "Task:\n"
            """You will repair a Python module that solves a graph algorithm task. The code cantains Syntax or Runtime Error.

            REQUIREMENTS:
            - Return ONLY the full corrected Python source code (**no explanations**).
            - Deterministic behavior. No randomness. No extra prints or logging.
            - If function-call cases exist, implement the requested functions with correct signatures/returns.
            - Prefer linear or near-linear solutions where appropriate; use NetworkX when suitable and allowed.
            - Handle directed/undirected and weighted/unweighted graphs per User Query.
            - Avoid external dependencies beyond the standard library and NetworkX."""
        )
        generated_code = openai_model.generate_code(prompt)
    elif not isinstance(openai_model, OpenCoderGenerator) and not match:
        prompt = (
            f"{instruction}\n\n"
            f"User Query:\n{query}\n\n"
            f"{retrieved_doc_prompt}\n"
            f"The code snippet to be correct:\n{error_code}\n"
            f"Test case: {test_case}"
            f"The incorrect output:\n{error_output}\n"
            "Task:\n"
            "The output of the code snippet does not have syntax or runtime error, but the output of the code does not match the test case"
            """REQUIREMENTS:
            - Return ONLY the full corrected Python source code (**no explanations**).
            - Deterministic behavior. No randomness. No extra prints or logging.
            - If function-call cases exist, implement the requested functions with correct signatures/returns.
            - Prefer linear or near-linear solutions where appropriate; use NetworkX when suitable and allowed.
            - Handle directed/undirected and weighted/unweighted graphs per User Query.
            - Avoid external dependencies beyond the standard library and NetworkX."""
        )
        print("Error not match for DeepSeek and Llama")
        generated_code = openai_model.generate_code(prompt)
    else: 
        print("Error not match for OpenCoder")
        generated_code = error_code
    return generated_code