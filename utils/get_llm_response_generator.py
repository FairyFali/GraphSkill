from .llm_agent.llama_code_generator import LlaMaGenerator
from .llm_agent.deepseek_code_generator import DeepSeekCodeGenerator
from .llm_agent.openai_code_generator import OpenAICodeGenerator
from .llm_agent.opencoder_code_generator import  OpenCoderGenerator
from .llm_agent.qwen_code_generator import QwenGenerator
import os
from huggingface_hub import login, InferenceClient

def create_code_generator(model_name: str, system_prompt: str = "You are a helpful assistant in graph tasks."):
    """
    Factory function to create a code generator instance based on the model name.

    Parameters:
        model_name (str): The full name of the model.
        system_prompt (str): The system prompt to use for the model.

    Returns:
        An instance of the appropriate code generator.
    """
    if "llama" in model_name.lower():
        # login(token = os.environ.get('HF_token'))
        model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        # meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
        print(f"Using Llama model: {model_name}")
        return LlaMaGenerator(model_name=model_name, system_prompt=system_prompt)
    elif "qwen-7b" in model_name.lower(): 
        model_name = "Qwen/Qwen2.5-7B-Instruct-Turbo"  # default
        print(f"Using Qwen model: {model_name}")
        return QwenGenerator(model_name=model_name, system_prompt=system_prompt)
    elif "qwen-72b" in model_name.lower(): 
        model_name = "Qwen/Qwen2.5-72B-Instruct-Turbo"
        print(f"Using Qwen model: {model_name}")
        return QwenGenerator(model_name=model_name, system_prompt=system_prompt)
    elif "qwen-coder" in model_name.lower(): 
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        print(f"Using Qwen model: {model_name}")
        return QwenGenerator(model_name=model_name, system_prompt=system_prompt)
    elif "gpt" in model_name.lower() or "openai" in model_name.lower():
        openai_api_key = os.environ.get("OPENAI_API_KEY") 
        model_name = "gpt-5.1-codex-mini"
        print(f"Using OpenAI model: {model_name}")
        return OpenAICodeGenerator(openai_api_key=openai_api_key, model_name=model_name, system_prompt=system_prompt)
    elif "deepseek" in model_name.lower():
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        model_name = "deepseek-chat"
        print(f"Using DeepSeek model: {model_name}")
        return DeepSeekCodeGenerator(openai_api_key=deepseek_api_key, model_name=model_name, system_prompt=system_prompt)
    elif "opencoder" in model_name.lower():
        client = InferenceClient(api_key=os.environ.get('HF_token'))
        model_name = "infly/OpenCoder-8B-Instruct"
        print(f"Using OpenCoder model: {model_name}")
        return OpenCoderGenerator(model_name=model_name)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")