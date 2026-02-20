from utils.llm_agent.openai_code_generator import OpenAICodeGenerator
from utils.llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from utils.llm_agent.llama_code_generator import LlaMaGenerator
from utils.llm_agent.opencoder_code_generator import OpenCoderGenerator
from typing import List
from rapidfuzz import fuzz, process
import json

def get_most_relevant_doc(
    query: str,
    openai_model: OpenAICodeGenerator|DeepSeekCodeGenerator|LlaMaGenerator|OpenCoderGenerator,
    doc_list: list[str] = None,
    composite_task: bool = False,
) -> str:
    """
    If retrieved_docs is provided, append them to the user query as context before generation.
    """
    
    instrcution = 'You will be provided with a description of graph task and a list of related documentations that can be used to resolve the graph problem.\n'

    if doc_list:
        answer_list = []
        print(f"get_most_relevant_cod.py, The length of doc list {len(doc_list)}\n")
        # if composite_task:
        #     strictness_prompt = "Determine if the function is helpful for addressing the query. You should think about how you can generate a Python function using NetworkX resolving the query, then consider the graph type and carefully review the user query to reach a well-supported conclusion."
        # else:
        #     strictness_prompt = "Determine whether the function correctly addresses the query, without requiring additional functions. You should think about how you can generate a Python function using NetworkX resolving the query, then consider the graph type and carefully review the user query to reach a well-supported conclusion."
        if composite_task:
            strictness_prompt = "Determine if the function is helpful for addressing the query. Think about the graph type and read carefully the User Query"
        else:
            strictness_prompt = "Determine whether the function correctly addresses the query, without requiring additional functions. You should think about how you can generate a Python function using NetworkX resolving the query, then consider the graph type and carefully review the user query to reach a well-supported conclusion."
        # if composite_task:
        #     strictness_prompt = "Determine if the doc is helpful for addressing the query." # Think about the graph type and read carefully the User Query.
        # else:
        #     strictness_prompt = "Determine if the doc fully, directly, and correctly addresses the query without requiring unrelated code or additional functions. Think about the graph type and read carefully the User Query. You can be harsh and should be very confident with the answer."
        # strictness_prompt = "Determine if the documentation is helpful for addressing the query."
        for doc in doc_list:
            # print(f"************String Representation: {string_representation}*************")
            prompt = (
                f"{instrcution}\n"
                f"User Query:\n{query}\n\n"
                f"Documentation:\n\"{doc}\"\n\n"
                f"{strictness_prompt}\n"
                "Return only one word: Yes or No. No explanations." #  If uncertain, answer No.
            )
            # print("### DEBUG, prompt:", prompt)
            answer = openai_model.generate(prompt)
            print("### DEBUG, answer:", answer)
            if "yes" in answer.lower():
                # print(f"The decision for keep {doc[:20]}\n *************{answer}*********************")
                answer_list.append(doc)
            else:
                # print(f"*******Unsupported Doc after careful reconsideration: \n{doc[:20]}\n")
                continue 
        
        if composite_task:
            print(f'### Processing composite task to retrieve multiple relevant {len(answer_list)} docs.')
            string_representation = json.dumps(answer_list)
            prompt = (
                f"{instrcution}"
                f"User Query:\n{query}\n\n"
                f"Documentation list:\n{string_representation}\n\n"
                "Choose all suitable relevant documentations whose content would fit the given task query the most, and must return exact documentation content without any explanation. The returned docs must be exact same as the documentation in the given documenation list."
            )
            doc_candidate = openai_model.generate(prompt)
            doc_candidate = json.loads(doc_candidate)
            print("### DEBUG, len(doc_candidate):", len(doc_candidate))
            retrieved_doc = []
            for ans in answer_list:
                if ans in doc_candidate:
                    retrieved_doc.append(ans)
        else: 
            print('### Processing non-composite task to retrieve the single most relevant doc.')
            if len(answer_list) <= 1:
                retrieved_doc = answer_list
            else: 
                string_representation = json.dumps(answer_list)
                prompt = (
                    f"{instrcution}"
                    f"User Query:\n{query}\n\n"
                    f"Documentation list:\n{string_representation}\n\n"
                    "Choose only one most relevant documentation whose content would fit the given task query, and must return exact documentation content without any explanation. The returned doc must be exact same as the documentation in the given documenation list."
                )
                doc_candidate = openai_model.generate(prompt)
                # doc_candidate = json.loads(doc_candidate)
                # print("### DEBUG, doc_candidate:", doc_candidate)
                # retrieved_doc = doc_candidate
                retrieved_doc = []
                if not doc_candidate in answer_list:
                    best_matches = process.extract_iter(doc_candidate, answer_list, scorer=fuzz.ratio)
                    for best_match in best_matches:
                        retrieved_doc.append(best_match[0])
                        if len(retrieved_doc) >= len(answer_list):
                            break
                retrieved_doc = retrieved_doc[:1]  # Keep only the top 1 most relevant doc

        for doc in retrieved_doc:
            print(f"*******Retrieved Doc after careful reconsideration: \n{doc[:100]}\n")

        # print(f"************String Representation: {string_representation}*************")
        # if composite_task:
        #     prompt = (
        #         f"{instrcution}"
        #         f"User Query:\n{query}\n\n"
        #         f"Documentation list:\n{string_representation}\n\n"
        #         "Choose all suitable relevant documentations whose content would fit the given task query the most, and must return exact documentation content without any explanation. The returned docs must be exact same as the documentation in the given documenation list."
        #     )
        # else: 
        #     prompt = (
        #         f"{instrcution}"
        #         f"User Query:\n{query}\n\n"
        #         f"Documentation list:\n{string_representation}\n\n"
        #         "Choose only one most relevant documentation whose content would fit the given task query, and must return exact documentation content without any explanation. The returned doc must be exact same as the documentation in the given documenation list."
        #     )

        # retrieved_doc = []
        # if len(answer_list) > 0:
        #     doc_candidate = openai_model.generate(prompt)
        #     if not doc_candidate in answer_list:
        #         pass
        #         # best_matches = process.extract_iter(doc_candidate, answer_list, scorer=fuzz.ratio)
        #         # for best_match in best_matches:
        #         #     retrieved_doc.append(best_match[0])
        #         #     if len(retrieved_doc) >= len(answer_list):
        #         #         break
        #     else:
        #         retrieved_doc.append(doc_candidate)
        # else:
        #     retrieved_doc = []
    else:
        print("Bug in chapter dict")
        retrieved_doc = []
    if retrieved_doc == "None":
        print("str Nooone")
    if retrieved_doc == None:
        print("None typeeeeeee")
    
    # print(f"************Retrieved Doc: {retrieved_doc[:20]} ... *************")
    print("### Verbose, len(retrieved_doc):", len(retrieved_doc))
    return retrieved_doc