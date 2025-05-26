from configs import AVAILABLE_LLMs
from utils import print_message, get_client

agent_profile = """You are the world's best machine learning research engineer specializing in on-device time series analysis for resource-constrained devices. Your main responsibilities are as follows:
1. Analyze user instructions and requirements.
2. Understand the specified model and constraints.
3. Based on your understanding, evaluate and measure the performance of TensorFlow/Keras model configurations under the given constraints."""


class EvaluationAgent:
    def __init__(self, task, llm, enc_images):
        self.task = task
        self.agent_type = "evaluation"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.enc_images = enc_images

    def evaluate_candidate(self, user_requirements, candidate, pid):
        print_message(self.agent_type, "I am evaluating the candidate model!", pid)

        # pre-execution verification
        evaluation_prompt = f"""As a proficient machine learning research engineer, your task is to estimate the performance and efficiency of the specified model for time-series {self.task} by thoroughly examining the following model configurations under the provided constraints.

[Proposed Model]
{candidate}        

[User Requirements]
{user_requirements}

Your analysis should include the candidate model's characteristics, such as computational complexity, memory usage, and inference latency, given the constraints. 
Then, please specify the (expected) quantitative {self.task} performance using relevant performance (.e.g., {'accuracy' if self.task == 'classification' else 'RMSE'}) and complexity metrics (e.g., number of parameters, FLOPs, model size, training time, inference speed, etc.).

Do not use placeholders for quantitative performance values. If exact values are unknown, use your knowledge and expertise to estimate the performance and complexity values."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": [{"type": "text", "text": evaluation_prompt}]},
        ]

        if len(self.enc_images) > 0:
            evaluation_prompt += "Finally, consider the characteristics of the time series using the provided plot images."
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
        print_message(self.agent_type, "I have done with evaluation!", pid)

        return response
