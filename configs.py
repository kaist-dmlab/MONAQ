class Configs:
    OPENAI_KEY = ""  # your openai's account api key
    HF_KEY = ""

AVAILABLE_LLMs = {  
    "gpt-4o": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
    "gpt-4o-mini": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o-mini"},
}

TASK_METRICS = {
    "classification": "Accuracy",
    "regression": "RMSE",
}
