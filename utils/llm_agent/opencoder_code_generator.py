from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import numpy as np
import pandas as pd

class OpenCoderGenerator:
    
    def __init__(self, 
                 model_name: str = "infly/OpenCoder-8B-Instruct", 
                 system_prompt: str = "You are a helpful assistant in graph tasks."):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def generate(self, prompt: str) -> str:
        """
        Sends the prompt to OpenCoder model and returns the assistant's analysis as text.
        """
        # print("Generating Code")
        # try:
        messages = [
        {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        # print("Inputs: ", inputs)
        outputs = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        # print("Outputs: ", outputs)
        # result = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        result = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return result
        # except Exception as e:
        #     print("Error during OpenCoder response generation:", str(e))
        #     return f"# Error during OpenCoder response generation: {str(e)}"