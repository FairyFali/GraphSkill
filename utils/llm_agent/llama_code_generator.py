from together import Together
import torch
import os
import numpy as np
import pandas as pd

class LlaMaGenerator:
    def __init__(self, 
                 model_name, # meta-llama/Meta-Llama-3-70B-Instruct
                 system_prompt):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.client = Together()

    def generate(self, prompt: str = None) -> str:
        """
        Sends the prompt to Llama model and returns the assistant's analysis as text.
        """
        # print("Generating prompt")
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages)
            return outputs.choices[0].message.content
        except Exception as e:
            return f"# Error during Llama response generation: {str(e)}"
