from configs import AVAILABLE_LLMs
from utils import print_message, get_client

agent_profile = """You are the world's best machine learning research engineer specializing in on-device time series analysis for resource-constrained devices. Your main responsibilities are as follows:
1. Analyze user instructions and requirements.
2. Understand the specified search space and constraints.
3. Based on your understanding, design optimal TensorFlow/Keras model configurations within the given constraints."""


class SearchAgent:
    def __init__(self, task, llm, enc_images):
        self.task = task
        self.agent_type = "search"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.enc_images = enc_images

    def search_model_with_feedback(
        self, user_requirements, search_space, feedback, pid
    ):
        print_message(
            self.agent_type,
            "I am re-designing a model using the given search space and feedback!",
            pid,
        )
        search_prompt = f"""As a proficient machine learning research engineer, your task is to design a set of model configurations for time-series {self.task} using the specified search space and the feedback from your previous trials.
Ensure that your model design is optimal and fully meets the user requirements and constraints in terms of performance and efficiency.

[Search Space]
{search_space}

[Organized User Requirements from Your Project Manager]
{user_requirements}

[Feedback]
{feedback}

Make sure that you have resolved the feedback and all operations or layers in your model are executable within the provided constraints.
Once you have completed the model design, please present it in an organized format, similar to the search space, with your selected options clearly indicated."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": [{"type": "text", "text": search_prompt}]},
        ]

        if len(self.enc_images) > 0:
            search_prompt += "Finally, consider the characteristics of the time series using the provided plot images."
            for img in self.enc_images:
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )

        response = get_client(self.llm).chat.completions.create(
            model=self.model, messages=messages, temperature=1
        )
        print_message(self.agent_type, "I have done with my re-modeling!", pid)

        return response

    def search_model(self, user_requirements, search_space, pid):
        print_message(
            self.agent_type, "I am designing a model using the given search space!", pid
        )
        search_prompt = f"""As a proficient machine learning research engineer, your task is to design a set of model configurations for time-series {self.task} using the specified search space.
Ensure that your model design is optimal and fully meets the user requirements and constraints in terms of performance and efficiency.

[Search Space]
{search_space}

[Organized User Requirements from Your Project Manager]
{user_requirements}

Make sure that all operations or layers in your model are executable within the provided constraints.
Once you have completed the model design, please present it in an organized format, similar to the search space, with your selected options clearly indicated."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": [{"type": "text", "text": search_prompt}]},
        ]

        if len(self.enc_images) > 0:
            search_prompt += "Finally, consider the characteristics of the time series using the provided plot images."
            for img in self.enc_images:
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )

        response = get_client(self.llm).chat.completions.create(
            model=self.model, messages=messages, temperature=1
        )
        print_message(self.agent_type, "I have done with my modeling!", pid)

        return response
