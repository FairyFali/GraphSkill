import openai
from openai import OpenAI
import os

class DeepSeekCodeGenerator:
    def __init__(self, 
                 openai_api_key: str, 
                 model_name: str = "deepseek-chat", 
                 system_prompt: str = "You are a helpful assistant in graph tasks."):
        """
        Initialize with your OpenAI API key and preferred model.
        Optionally set a system prompt to steer code generation style.
        """
        self.client = OpenAI(api_key=openai_api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        """
        Sends the prompt to OpenAI's ChatCompletion and returns the assistant's reply as code.
        """
        # print("Generating Code")
        try:

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # For reproducible, deterministic responses
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"# Error during OpenAI API call: {str(e)}"
