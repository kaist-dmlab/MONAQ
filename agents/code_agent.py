import json, os, time

from configs import AVAILABLE_LLMs
from utils import print_message, get_client

agent_profile = """You are the world's best machine learning engineer specializing in on-device time series analysis for resource-constrained devices. You have the following main responsibilities to complete.
1. Write accurate Python codes to build model in get_model() function based on the given instruction.
2. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
"""


class CodeAgent:
    def __init__(self, task, llm, enc_images, data_name, device="0"):
        self.task = task
        self.agent_type = "code"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.enc_images = enc_images
        self.data_name = data_name
        self.device = device

    def write_code(self, user_requirements, code_instruction, code_path):
        with open(f"templates/{self.task}.py") as file:
            code = file.read()
        print_message(
            self.agent_type, f"I am coding the following model.\n\r{code_instruction}"
        )
        code_prompt = f"""As an experienced machine learning engineer, carefully read the following instructions and user requirements to write Python code of a Tensorflow/Keras model for time-series {self.task}.
[Code Instruction]
{code_instruction}

[User Requirements]
{user_requirements}

[{self.task}.py] ```python
{code}
```

Start the python code with "```python". Focus only on completing the get_model() function while returning the remaining parts of the script exactly as provided.
Ensure the code is complete, error-free, and ready to run without requiring additional modifications.
Note that we only need the actual complete python code without textual explanations."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": [{"type": "text", "text": code_prompt}]},
        ]

        if len(self.enc_images) > 0:
            code_prompt += "Finally, consider the characteristics of the time series using the provided plot images."
            for img in self.enc_images:
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )

        response = get_client(self.llm).chat.completions.create(
            model=self.model, messages=messages, temperature=0.5
        )
        raw_completion = response.choices[0].message.content.strip()
        completion = raw_completion.split("```python")[1].split("```")[0]

        dirname = f"code_results/monaq_{self.llm}/{self.task}/{code_path}"
        os.makedirs(dirname, exist_ok=True)
        filename = f"{dirname}/{self.data_name}.py"
        with open(filename, "wt") as file:
            file.write(completion)
        print_message(
            self.agent_type, "I have done with my code and saved it at: " + filename
        )

        return completion, response.usage.to_dict(mode="json")