from utils.llm_agent.openai_code_generator import OpenAICodeGenerator
from utils.llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from utils.llm_agent.llama_code_generator import LlaMaGenerator
from utils.llm_agent.opencoder_code_generator import OpenCoderGenerator
from typing import List
import json

def get_most_relevant_doc(
    query: str,
    openai_model: OpenAICodeGenerator|DeepSeekCodeGenerator|LlaMaGenerator|OpenCoderGenerator,
    doc_list: list[str] = None
) -> str:
    """
    If retrieved_docs is provided, append them to the user query as context before generation.
    """
    
    instrcution = 'You will be provided with a description of graph task and a list of related documentations that can be used to resolve the graph problem.\n'

    if doc_list:
        string_representation = json.dumps(doc_list)
        # print(f"************String Representation: {string_representation}*************")
        prompt = (
            f"{instrcution}"
            f"User Query:\n{query}\n\n"
            f"Documentation list:\n{string_representation}\n\n"
            " Choose all relevant documentations whose content would fit the given task query the most, and must return exact documentation content without any explanation. The returned word must be in the given documentation list."
        )
        retreived_doc = openai_model.generate_code(prompt)
    else:
        print("Bug in chapter dict")
        retreived_doc = None
    if retreived_doc == "None":
        print("str Nooone")
    if retreived_doc == None:
        print("None typeeeeeee")
    print(f"************Retrieved Doc: {retreived_doc}*************")
    return retreived_doc