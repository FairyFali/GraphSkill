from utils.llm_agent.openai_code_generator import OpenAICodeGenerator
from utils.llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from utils.llm_agent.llama_code_generator import LlaMaGenerator
from utils.llm_agent.opencoder_code_generator import OpenCoderGenerator
from typing import List
from rapidfuzz import fuzz, process
import ast
import json
import re

def to_python_list(s: str):
    # Regex to check if string looks exactly like a Python list
    # (starts with [ , ends with ], allows nested structures)
    list_pattern = r'^\s*\[.*\]\s*$'

    if re.match(list_pattern, s):
        try:
            # Safely evaluate to Python object
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            pass  # fall back to fixing

    # --- Try to fix "similar to list" strings ---
    fixed = s.strip()

    # Common fixes
    if not fixed.startswith("["):
        fixed = "[" + fixed
    if not fixed.endswith("]"):
        fixed = fixed + "]"

    # Replace semicolons with commas, remove extra spaces
    fixed = re.sub(r';', ',', fixed)
    fixed = re.sub(r'\s+', ' ', fixed)

    try:
        return ast.literal_eval(fixed)
    except Exception as e:
        raise ValueError(f"Could not convert to list: {s} (fixed: {fixed})") from e


def retrieve_documentation_chapter(
    query: str,
    openai_model: OpenAICodeGenerator|DeepSeekCodeGenerator|OpenCoderGenerator|LlaMaGenerator,
    chapters_and_descs: dict[str: str] = None,
    explored_choices: List[str] = None,
    top_k: int = 3,
    llm_descision: bool = False,
) -> str:
    """
    If retrieved_docs is provided, append them to the user query as context before generation.
    """
    
    instrcution = 'You will be provided with a description of graph task and a dictionary of chapter names and their descriptions.\n'
    if llm_descision:
        select_chapter_prompt = "Choose UNIQUE items from **Available Chapters** that are essential for the given task."
    else:
        select_chapter_prompt = f"Choose exactly {top_k} UNIQUE items from **Available Chapters** that are essential for the given task."
    
    if chapters_and_descs and not explored_choices:
        string_representation = json.dumps(chapters_and_descs)
        prompt = (
            f"{instrcution}"
            f"User Query:\n{query}\n\n"
            f"Chapter name and Chapter Description:\n{string_representation}\n\n"
            """Select the three chapters that most effectively address the given task. Respond only with a Python-style list containing the exact chapter names (they must match the dictionary keys), in this format: "["choice1", "choice2", "choice3"]" Do not include any explanation."""
        )
        
        chapter_name = openai_model.generate(prompt)
    elif chapters_and_descs and explored_choices:
        string_representation = json.dumps(chapters_and_descs)
        explored_choices_str = json.dumps(explored_choices)
        prompt = (
            f"{instrcution}\n"
            f"Task: Select the top {top_k} chapter keys that most directly help solve the user query.\n\n"
            f"User Query:\n{query}\n\n"
            "Available Chapters (key -> description):\n"
            f"{string_representation}\n\n"
            "Already Explored (exclude these exact keys):\n"
            f"{explored_choices_str}\n\n"
            "Requirements:\n"
            f"- {select_chapter_prompt}\n"
            "- Keys MUST match exactly and be case-sensitive (use the dictionary keys as-is).\n"
            "- Do NOT include any key that appears in **Already Explored**.\n"
            f"- If fewer than {top_k} eligible keys exist, return as many as exist (>= 0).\n\n"
            "Selection Guidelines (apply in order):\n"
            "1) Direct capability: prefer chapters whose descriptions indicate functions/algorithms/APIs that can directly fulfill the query.\n"
            "2) Specificity over breadth: prefer narrowly-scoped chapters that target the requested operation.\n"
            "3) Term alignment: prefer chapters mentioning terms/synonyms present in the query (e.g., “shortest path,” “connectivity,” “pagerank”).\n"
            "4) Disambiguation: when multiple are similar, pick the one most likely to lead to an immediately usable function.\n\n"
            "Output format (STRICT):\n"
            f"- Return ONLY a Python list literal of length <= {top_k}, using double quotes, e.g.: "
            '["key1", "key2", "key3"]\n'
            "- No explanations, no code fences, no extra text.\n"
            "- If nothing fits, return [].\n\n"
            "Validate internally before answering:\n"
            "- All items are present in **Available Chapters** keys.\n"
            "- No item appears in **Already Explored**.\n"
            f"- Items are unique and count <= {top_k}.\n"
        )
        chapter_name = openai_model.generate(prompt)
    else:
        print("Bug in chapter dict")
        chapter_name = None
    if chapter_name == "None":
        chapter_name = None
        print("str Nooone")
    if chapter_name == None:
        print("None typeeeeeee")
    # import pdb;pdb.set_trace()
    chapter_name_lst_type = to_python_list(chapter_name.strip())
    chapter_name = []
    all_chapters = list(chapters_and_descs.keys())
    if not set(chapter_name_lst_type).issubset(chapters_and_descs):
        for name in chapter_name_lst_type:
            best_match = process.extractOne(name, all_chapters, scorer=fuzz.ratio)
            if name in all_chapters:
                chapter_name.append(name)
            elif best_match[1] >= 75:
                chapter_name.append(best_match[0])
    else:
        chapter_name = chapter_name_lst_type
        

    print(f"************chpater name: {chapter_name}*************")
    return chapter_name