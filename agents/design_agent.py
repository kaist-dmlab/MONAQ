from configs import AVAILABLE_LLMs
from utils import print_message, get_client

agent_profile = """You are the world's best data scientist of an on-device time series analysis for resource-constrained devices. You have the following main responsibilities to complete.
1. Analyze user instructions and requirements.
2. Based on the requirements, design a neural network search space for resource-constrained devices."""

example_format = """
{
    "layer_type": ["Conv1D", "DepthwiseConv1D", "SeparableConv1D", "LSTM", "Dense"]
    "Conv1D_kernel_size": [1, 3, 5],
    "Conv1D_filter": [8, 16]
    "DepthwiseConv1D_kernel_size": [1, 3, 5]
}
"""


class DesignAgent:
    def __init__(self, task, llm, enc_images):
        self.task = task
        self.agent_type = "design"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.enc_images = enc_images

    def design_space(self, user_requirements):
        print_message(self.agent_type, "I am designing search space!")
        design_prompt = f"""As a proficient data scientist, your task is to design a search space comprising choices of high-performing neural networks for time-series {self.task}.
Ensure that the neural network design choices can be implemented using the TensorFlow/Keras library and are compatible with conversion to the TFLite model format.

[Organized User Requirements from Your Project Manager]
{user_requirements}

Please ensure that all options within the search space can be executed under the constraints provided above.
Once you have completed the search space design, please present it in an organized format, listing the available configurations and the number of options for each. 
For example, {example_format}, where each key represents an option name and each list of values shows the possible configurations based on your design choices."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": [{"type": "text", "text": design_prompt}]},
        ]

        if len(self.enc_images) > 0:
            design_prompt += "Finally, consider the characteristics of the time series using the provided plot images."
            for img in self.enc_images:
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )

        response = get_client(self.llm).chat.completions.create(
            model=self.model, messages=messages, temperature=0.3
        )
        print_message(self.agent_type, "I have done with my design!")

        return response
