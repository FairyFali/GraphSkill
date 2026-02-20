from utils.llm_agent.openai_code_generator import OpenAICodeGenerator

def convert_graph_task_to_real_world_problem(
    example_conversion: str,
    user_query:str,
    openai_model: OpenAICodeGenerator,
    graph_task_name: str,
    real_world_task_name: str,
) -> str:
    """
    If retrieved_docs is provided, append them to the user query as context before generation.
    """
    
    instrcution = f'You are a sophisticated AI expert in graph theory and algorithms. Convert the {graph_task_name} graph problem to the {real_world_task_name} problem following the example without explanation \n'

    prompt = (
        f"{instrcution}"
        f"example convertion:\n{example_conversion}\n\n"
        f"{user_query}"
        "Please convert the graph problem to real world problem using the above examples."
    )
    
    generated_code = openai_model.generate_code(prompt)
    return generated_code