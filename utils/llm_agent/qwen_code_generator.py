from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from together import Together


class QwenGenerator:
    def __init__(self,
                 model_name: str,
                 system_prompt: str):
        """
        Initialize with your preferred model.
        Optionally set a system prompt to steer code generation style.
        """

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.client = Together()

    def generate(self, prompt: str) -> str:
        """
        Sends the prompt to Qwen's ChatCompletion and returns the assistant's reply as code.
        """
        # print("Generating Code")
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages)
            return outputs.choices[0].message.content
        #     tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        #     model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True).eval()
        #    # model.generation_config = GenerationConfig.from_pretrained(self.model_name, trust_remote_code=True)
        #     response, history = model.chat(tokenizer, prompt, history=None)
        #     return response
        except Exception as e:
            return f"# Error during Qwen API call: {str(e)}"
