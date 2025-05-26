import json, os, time
import pandas as pd

from multiprocessing import Pool, current_process
from configs import AVAILABLE_LLMs
from agents.design_agent import DesignAgent
from agents.search_agent import SearchAgent
from agents.evaluation_agent import EvaluationAgent
from agents.code_agent import CodeAgent
from utils import print_message, get_client, encode_image
from glob import glob

agent_profile = """
You are an experienced senior project manager overseeing on-device time series analysis for resource-constrained devices. Your primary responsibilities are as follows:
1. Receive requirements and inquiries from users regarding their task descriptions and potential target devices for deployment.
2. Extract and clarify user requirements from both data and modeling perspectives, organizing these requirements and task-specific constraints in an easy-to-understand format to enable other team members to execute subsequent processes based on the information you have gathered.
3. Verify the suggested model whether it meets the user requirements and constraints.
"""

basic_profile = """You are a helpful, respectful and honest "human" assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# JSON format for extracting task-related requirements and constraints
requirement_formant = """{
"task_description": "", // Clearly describe the user's requirements and the problem they are addressing
"data_aspects": {
        "name": "", // Dataset name, if provided
        "description": "", // Complete description of the dataset
        "features": "", // Details on features, properties, and characteristics of the dataset to consider for model building
        "context": "", // Relevant contextual information about the dataset
        "patterns": "" // Observed patterns in the dataset to consider for model building, based on any provided numerical data or images
    },
"model_aspects": {
    "name": "", // Suggested model name for the task, if provided by the user
    "hardware_specs": {
            "device_name": "", // Device name, if specified by the user, or inferred from hardware specifications
            "ram": "", // Maximum RAM available (in bytes), which affects the model's MAC/FLOPs limit (you can also infer it from the device name)
            "flash": "", // Maximum FLASH storage (in bytes), which affects model size and parameter count (you can also infer it from the device name)
        },
        "MAC": "", // Maximum MAC (Multiply-Accumulate) operations (or FLOPs) allowed for model compatibility with hardware constraints
        "parameters": "", // Maximum model parameters, in line with hardware constraints
        "latency": "", // Desired inference latency in milliseconds (ms), or the maximum latency allowed based on hardware limitations
        "performance": "" // Expected model performance, such as accuracy for classification or RMSE for regression; specify any target metric values to consider for model building
    }
}
"""


class AgentManager:
    def __init__(
        self,
        task,
        data_name,
        n_candidates=3,
        n_budgets=1,
        llm="gpt-4o-mini",
        modality=["time", "text", "image"],
        image_paths=[],
    ):
        self.agent_type = "manager"
        self.task = task
        self.data_name = data_name
        self.n_candidates = n_candidates  # number of search suggestions
        self.n_budgets = n_budgets  # number of trials
        self.budgets = n_budgets
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.chats = []
        self.state = "INIT"
        self.query_modality = modality  # time, text, image
        self.timer = {}
        self.money = {}
        self.candidates = []
        self.enc_images = [encode_image(img) for img in image_paths]
        self.code_path = (
            f"Q{'_'.join(self.query_modality)}_C{self.n_candidates}_B{self.n_budgets}"
        )

    def _rewrite_query(self, user_prompt):
        extraction_prompt = f"""Please carefully analyze the user's task descriptions based on your understanding of the following input:
[User Input Prompt]
{user_prompt}

After fully understanding the task descriptions and constraints, extract and organize the information in the specified format below.
Please respond as the following JSON object and make sure your JSON object is in a valid form.
```json
{requirement_formant}
```
"""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": [{"type": "text", "text": extraction_prompt}]},
        ]

        if "image" in self.query_modality:
            for img in self.enc_images:
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )

        n_attempts = 0    
        while n_attempts < 5:
            try:
                response = get_client(self.llm).chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                self.money["INIT"] = response.usage.to_dict(mode="json")
                self.user_requirements = json.loads(response.choices[0].message.content.strip())
                break
            except Exception as e:
                n_attempts += 1
                print_message('system', f"Error occurs when calling API: {e}")

    def _design_search_space(self):
        designer = DesignAgent(task=self.task, llm=self.llm, enc_images=self.enc_images)
        response = designer.design_space(self.user_requirements)
        self.search_space = response.choices[0].message.content
        self.money[f"DESIGN_{self.n_budgets}"] = response.usage.to_dict(mode="json")

    def _build_model(self, _):
        start_time = time.time()

        pid = current_process()._identity[0]  # for checking the current plan
        searcher = SearchAgent(task=self.task, llm=self.llm, enc_images=self.enc_images)
        response = searcher.search_model(self.user_requirements, self.search_space, pid)

        return response, time.time() - start_time

    def _build_model_with_feedback(self, feedback):
        start_time = time.time()

        pid = current_process()._identity[0]  # for checking the current plan
        searcher = SearchAgent(task=self.task, llm=self.llm, enc_images=self.enc_images)
        response = searcher.search_model_with_feedback(self.user_requirements, self.search_space, feedback, pid)

        return response, time.time() - start_time

    def _evalaute_model(self, candidate):
        start_time = time.time()

        pid = current_process()._identity[0]  # for checking the current plan
        evaluator = EvaluationAgent(task=self.task, llm=self.llm, enc_images=self.enc_images)
        response = evaluator.evaluate_candidate(self.user_requirements, candidate, pid)

        return response, time.time() - start_time

    def _verify_solution(self, candidate):
        start_time = time.time()

        # pre-execution verification
        verification_prompt = f"""Given the proposed model and user requirements, please carefully check and verify whether the proposed model 'pass' or 'fail' the user requirements and constraints.

[Proposed Model]
{candidate}

[User Requirements]
{self.user_requirements}

Please answer in the following JSON format.
```json
{{
    "pass": True or False, // a Boolean value of 'True' or 'False' indicating the verification result
    "rationale": "feedback why it fails" // a feedback explanation why the model does not pass the verification
}}
```
"""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": verification_prompt},
        ]

        res = get_client(self.llm).chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )

        return res, time.time() - start_time

    def _write_code(self):
        coder = CodeAgent(
            task=self.task,
            llm=self.llm,
            enc_images=self.enc_images,
            data_name=self.data_name,
        )
        completed_code, tokens = coder.write_code(
            self.user_requirements, self.code_instruction, self.code_path
        )
        self.money[f"CODE_{self.n_budgets}"] = tokens
        return completed_code

    def _on_stop(self, msg):
        return msg.lower() in [
            "stop",
            "close",
            "exit",
            "terminate",
            "end",
            "done",
            "finish",
            "complete",
            "bye",
            "goodbye",
        ]

    def initiate_chat(self, prompt):
        last_msg = prompt
        pool = Pool(self.n_candidates)
        start_time = time.time()  # init time

        while not self._on_stop(last_msg) and self.state != "END":
            # reply process: current state + current state description + response
            if last_msg == "":
                sys_query = "Please give feedback or answer to proceed. You may type 'exit' to end the session."
                last_msg = input(sys_query)
                if last_msg == "" or self._on_stop(last_msg):
                    continue
                else:
                    prompt = last_msg

            # talking with user here, keep appending message to messages with the oai format
            if self.state == "INIT":
                # display user's input prompt
                self.chats.append({"role": "user", "content": prompt})
                print_message("user", prompt)

                self._rewrite_query(prompt)
                self.chats.append(
                    {
                        "role": "assistant",
                        "content": f"""We have analyzed your requirements as follows!\n\r{self.user_requirements}""",
                    }
                )
                print_message(
                    self.agent_type,
                    f"""We have analyzed your requirements as follows!\n\r{self.user_requirements}""",
                )

                # TODO: add user's prompt verification step --> # classify user's prompt into "chit-chat / simple query" vs. "ML/AI related request"
                self.timer["INIT"] = time.time() - start_time
                self.state = "DESIGN"
                
                self.n_budgets = self.n_budgets - 1

            elif self.state == "DESIGN":
                # design search space
                start_time = time.time()
                self._design_search_space()
                self.timer[f"DESIGN_{self.n_budgets}"] = time.time() - start_time
                self.state = "BUILD"

            elif self.state == "BUILD":
                # build candidate models from the *designed* search space
                start_time = time.time()
                # Parallelization
                with Pool(self.n_candidates) as pool:
                    responses = pool.map(self._build_model, range(self.n_candidates))
                self.timer[f"BUILD_{self.n_budgets}"] = time.time() - start_time

                self.candidates = []
                for idx, (res, t) in enumerate(responses):
                    self.money[f"BUILD_{idx}_{self.n_budgets}"] = res.usage.to_dict(mode="json")
                    self.timer[f"BUILD_{idx}_{self.n_budgets}"] = t
                    self.candidates.append(res.choices[0].message.content)

                self.state = "EVAL"

            elif self.state == "EVAL":
                # evaluate the candidate models and select the best one for that round
                start_time = time.time()
                with Pool(self.n_candidates) as pool:
                    responses = pool.map(self._evalaute_model, self.candidates)
                self.timer[f"EVAL_{self.n_budgets}"] = time.time() - start_time

                self.candidate_results = []
                for idx, (res, t) in enumerate(responses):
                    self.money[f"EVAL_{idx}_{self.n_budgets}"] = res.usage.to_dict(mode="json")
                    self.timer[f"EVAL_{idx}_{self.n_budgets}"] = t
                    self.candidate_results.append(res.choices[0].message.content)

                print_message(self.agent_type, "I am now verifying the models found by our agent team 🦙.")

                # Parallelization
                with Pool(self.n_candidates) as pool:
                    responses = pool.map(self._verify_solution, self.candidate_results)
                self.timer[f"VERIFY_{self.n_budgets}"] = time.time() - start_time

                self.verification_results = []
                for idx, (res, t) in enumerate(responses):
                    self.money[f"VERIFY_{idx}_{self.n_budgets}"] = res.usage.to_dict(mode="json")
                    self.timer[f"VERIFY_{idx}_{self.n_budgets}"] = t
                    self.verification_results.append(json.loads(res.choices[0].message.content))                

                self.feedback = ""
                self.is_solution_found = False
                pass_candidates = []
                for i, result in enumerate(self.verification_results):
                    if type(result["pass"]) == bool:
                        if result["pass"] == True:
                            pass_candidates.append(i)
                            self.is_solution_found = True
                    elif (
                        "pass" in result["pass"].lower().strip()
                        or "true" in result["pass"].lower().strip()
                    ):
                        pass_candidates.append(i)
                        self.is_solution_found = True
                    else:
                        self.feedback += result["rationale"]
                
                if self.n_budgets == 0 or self.is_solution_found:
                    found_models = ""
                    for idx, num in enumerate(pass_candidates):
                        found_models += f"""[Model Configuration #{idx + 1}]\n\r{self.candidates[num]}\n\r{self.candidate_results[num]}"""

                    summary_prompt = f"""As the project manager, please carefully select the best model that meets the given user requirements from the list of candidates below.
[Candidate Models with Expected Performances]
{found_models}                    

[User Requirements]
{self.user_requirements}

Please note that you must select only one solution based on the provided suggestions. After choosing the best model, only provide a complete configuration for the selected model. 
This configuration will guide machine learning engineers, who will implement the code based on your instructions. Do not write the code yourself."""
                    messages = [
                        {"role": "system", "content": agent_profile},
                        {"role": "user", "content": summary_prompt},
                    ]
                    response = get_client(self.llm).chat.completions.create(
                        model=self.model, messages=messages, temperature=0.3
                    )
                    self.code_instruction = response.choices[0].message.content
                    self.timer[f"INST_{self.n_budgets}"] = time.time() - start_time
                    self.money[f"INST_{self.n_budgets}"] = response.usage.to_dict(mode="json")
                    self.state = "CODE"
                else:
                    self.state = "REVISE"

            elif self.state == "REVISE":
                # build candidate models from the *designed* search space
                start_time = time.time()
                # Parallelization
                with Pool(self.n_candidates) as pool:
                    responses = pool.map(self._build_model_with_feedback, [self.feedback for _ in range(self.n_candidates)])
                self.timer[f"BUILD_{self.n_budgets}"] = time.time() - start_time

                self.candidates = []
                for idx, (res, t) in enumerate(responses):
                    self.money[f"BUILD_{idx}_{self.n_budgets}"] = res.usage.to_dict(mode="json")
                    self.timer[f"BUILD_{idx}_{self.n_budgets}"] = t
                    self.candidates.append(res.choices[0].message.content)
                
                self.state = "EVAL"

            elif self.state == "CODE":
                # write the final code for the selected candidate
                start_time = time.time()
                self.result = self._write_code()
                self.timer[f"CODE_{self.n_budgets}"] = time.time() - start_time
                self.state = "END"
            elif self.state == "END":
                print_message(
                    self.agent_type,
                    f"Here is your final model!\n\r```python\n\r{self.result}```",
                )
                break
