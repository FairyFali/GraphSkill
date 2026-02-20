import json
import random as rd
import ast
import os
from utils.llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from typing import Dict, List, Tuple
from utils.llm_agent.openai_code_generator import OpenAICodeGenerator
from utils.generation_functions.retrieve_doc_chapter import retrieve_documentation_chapter
from utils.generation_functions.get_most_relevant_doc import get_most_relevant_doc
import re, unicodedata
from utils.sentbert_retrieval import find_best_docstring

ZERO_WIDTH = r'[\u200B-\u200D\uFEFF]'         # common invisible chars
WHITESPACE = r'\s+'                           # every Unicode whitespace

def normalise(text: str) -> str:
    text = unicodedata.normalize('NFC', text)

    text = re.sub(ZERO_WIDTH, '', text)

    text = re.sub(WHITESPACE, ' ', text, flags=re.UNICODE)
    # text = text.replace('"', '').replace("'", '')

    return text.strip()  


import json
from typing import List, Tuple, Optional

def retrieve_doc(
    doc_path: str,
    user_query: str,
    llm_model,  # DeepSeekCodeGenerator | OpenAICodeGenerator
    _depth: int = 0,
    max_depth: int = 5,
    top_k: int = 3,
    max_rounds: int = 3,
    llm_descision: bool = False,
    composite_task: bool = False,
) -> Tuple[List[str], str]:
    """
    Walk the nested NetworkX docs menu, restarting from the top if the chosen
    document is None/"None" at the leaf.

    Returns
    -------
    (first_level_keys, final_doc_path)
    """

    def _one_round(explored_initial_choices: List[str] = [], llm_descision=llm_descision) -> Tuple[List[str], Optional[str]]:
        """Run a single traversal round. Returns (first_level_keys, doc or None)."""
        with open(doc_path, "r", encoding="utf-8") as nx_file:
            nx_doc = json.load(nx_file)

        cat = list(nx_doc.keys())
        desc = [nx_doc[key][0] for key in cat]
        inital_chap = dict(zip(cat, desc))
        if len(explored_initial_choices) == 0:
            initial_choices = retrieve_documentation_chapter(
                query=user_query, 
                openai_model=llm_model, 
                chapters_and_descs=inital_chap,
                llm_descision=llm_descision)
        else: 
            initial_choices = retrieve_documentation_chapter(
                query=user_query, 
                openai_model=llm_model, 
                chapters_and_descs=inital_chap, 
                explored_choices=explored_initial_choices,
                llm_descision=llm_descision)
        first_level_keys = initial_choices
        depth = 0
        current_menu_list = [nx_doc]
        current_choices = [[normalise(choice.strip()) for choice in initial_choices]]
        retrieved_doc_list: List[Optional[str]] = []

        while True:
            # -------- safety guard -----------
            if max_depth is not None and depth >= max_depth:
                raise RecursionError("Maximum depth exceeded while traversing menu")

            # -------- base case: reached a leaf -----------
            if len(list(filter(None, current_choices))) == 0:
                # Choose most relevant among collected leaves.
                most_related_docs = get_most_relevant_doc(user_query, llm_model, retrieved_doc_list, composite_task=composite_task)
                doc = (most_related_docs or []) if most_related_docs is not None else None
                if not doc or doc == "None":
                    # indicate to caller that this round failed and should restart
                    return (first_level_keys or []), None
                return (first_level_keys or []), doc

            # -------- gather children from current choices -----------
            try:
                children = []
                for idx in range(len(current_menu_list)):
                    if current_choices[idx] is None:
                        continue
                    for choice in current_choices[idx]:
                        
                        if choice is None:
                            continue
                        child = current_menu_list[idx][choice][1]
                        if child not in children:
                            children.append(child)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid menu structure or unknown choice during traversal."
                ) from exc

            # -------- ask LLM for next choices / collect leaves -----------
            next_menu = []
            next_choices = []
            # Capture first level keys only once (topmost non-leaf level)

            for child in children:
                if isinstance(child, dict):
                    sub_menu_keys = list(child.keys())
                    if len(sub_menu_keys) > top_k:
                        sub_menu_desc = [child[k][0] for k in sub_menu_keys]
                        chapter_names_and_descs = dict(zip(sub_menu_keys, sub_menu_desc))
                        next_choice = retrieve_documentation_chapter(
                            query=user_query, 
                            openai_model=llm_model, 
                            chapters_and_descs=chapter_names_and_descs, 
                            llm_descision=llm_descision
                        )
                    else:
                        next_choice = sub_menu_keys
                    next_menu.append(child)
                    next_choices.append(next_choice)
                elif isinstance(child, str):
                    retrieved_doc_list.append(child)
                else:
                    # non-dict, non-str → ignore but keep alignment
                    retrieved_doc_list.append(None)

            # -------- move one level deeper -----------
            current_menu_list = next_menu
            current_choices = next_choices
            depth += 1

    # ================== multi-round controller ==================
    last_first_keys: List[str] = []
    explored_initial_choices = []
    for _ in range(0, max_rounds):
        first_keys, maybe_doc = _one_round(explored_initial_choices, llm_descision)
        last_first_keys = first_keys
        if maybe_doc is not None and maybe_doc != "None":
            return maybe_doc
        explored_initial_choices += last_first_keys
        # otherwise, restart a new round from the top

    # If we exhausted rounds without a valid doc, raise a clear error
    retrieved_doc = find_best_docstring(doc_path, user_query, top_k=1)[0][1] or []
    return retrieved_doc










# def is_all_document(list):
#     return all(isinstance(e, str) and len(e) > 50 for e in list)

# def merge_lists_no_duplicates(list1, list2):
#     merged_list = list1 + list2
#     unique_list = []
#     seen = set()
#     for item in merged_list:
#         if item not in seen:
#             unique_list.append(item)
#             seen.add(item)
#     return unique_list


# def retrieve_doc(doc_path: str, user_query: str, llm_model: DeepSeekCodeGenerator|OpenAICodeGenerator, _depth: int = 0, max_depth: int = 5, top_k: int = 3):
#     """
#     Recursively walk a nested ``menu`` structure until a terminal (string) node is found.

#     Parameters
#     ----------
#     menu : dict
#         Nested dictionary built like:
#         {
#             "key": [<description>, <either str leaf OR another dict>],
#             ...
#         }
#     choice : str
#         The key at the current level to start from.
#     _depth : int, optional
#         Internal counter used to prevent accidental infinite recursion.
#     max_depth : int | None, optional
#         Hard-stop depth guard (useful if your data might contain cycles).

#     Returns
#     -------
#     list[str]
#         List of the keys that were available at **this** level (empty if we started on a leaf).
#     str
#         The final terminal choice reached after recursively descending.
#     """
#     nx_file = open(doc_path)
#     nx_doc = json.load(nx_file)

#     cat = list(nx_doc.keys())
#     desc = [nx_doc[key][0] for key in cat]
#     inital_chap = dict(zip(cat, desc))
#     initial_choices = retrieve_documentation_chapter(user_query, llm_model, inital_chap)
#     depth = 0
#     first_level_keys: List[str] | None = None          # captured once
#     current_menu_list = [nx_doc]
#     current_choices = [[normalise(choice.strip()) for choice in initial_choices]]
#     retrieved_doc_list = []
#     while True:
#         # ---- safety guard ----------------------------------------------------
#         if max_depth is not None and depth >= max_depth:
#             raise RecursionError("Maximum depth exceeded while traversing menu")
#         # print(current_menu_list)
#         # ---- base case: reached a leaf --------------------------------------
#         # import pdb;pdb.set_trace()
#         if len(list(filter(None, current_choices))) == 0:
#             # If we never descended, first_level_keys is still None
#             most_related_docs = get_most_relevant_doc(user_query, llm_model, retrieved_doc_list).strip()
#             doc = most_related_docs
#             if doc == None or doc == "None":
                
#                 continue
#             return (first_level_keys or []), doc
        
#         # current_menu_list = [item for item in current_menu_list if not isinstance(list(item.values())[0][1], str)]
#         # current_choices = [item for item in current_choices if not is_all_document(item)]
#         # if not len(current_choices) == len(current_menu_list):
#         #     raise RuntimeError(f"Current choices does not match with current menu list: length of choices is {len(current_choices)}, while length of menu is {len(current_menu_list)}")
#         # # import pdb;pdb.set_trace()
#         try:
#             children = []
#             for idx in range(0, len(current_menu_list)):
#                 if not current_choices[idx] == None:
#                     for choice in current_choices[idx]:
#                         child = current_menu_list[idx][choice][1]
#                         if not child in children:
#                             children.append(child)
                    
#         except (KeyError, ValueError, TypeError) as exc:
#             import pdb;pdb.set_trace()
#             raise ValueError(f"Invalid menu structure or unknown choice: {choice} in {current_menu_list[idx].keys()}") from exc
#         # ---- iterative “descent” --------------------------------------------
#         # Capture the keys at *this* level only once (topmost non-leaf level).
#         print(f"Current choices: {current_choices}")
#         # Ask the LLM which branch to follow next.
#         next_menu = []
#         next_choices = []
#         for child in children:
#             if isinstance(child, dict):
#                 sub_menu_keys = list(child.keys())
#                 # if less than two we do not need LLM
#                 if len(sub_menu_keys) > top_k:
#                     sub_menu_desc = [child[k][0] for k in sub_menu_keys]
#                     chapter_names_and_descs = dict(zip(sub_menu_keys, sub_menu_desc))
#                     next_choice = retrieve_documentation_chapter(user_query, llm_model, chapter_names_and_descs)
#                 else:
#                     next_choice = sub_menu_keys
#                 next_menu.append(child)
#                 next_choices.append(next_choice)
#             elif isinstance(child, str):
#                 retrieved_doc_list.append(child)
#             else:
#                 print("None appended in choice list")
#                 next_choice = None
#                 retrieved_doc_list.append(next_choice)
             
#         print(next_choices)
#         # Move one level deeper and continue.
#         current_menu_list = next_menu
#         current_choices = next_choices
#         print(depth)
#         depth += 1