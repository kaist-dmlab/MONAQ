import shutil
import base64

import tensorflow as tf

from openai import OpenAI
from configs import AVAILABLE_LLMs
from mltk.core import profile_model

out_cost = 10.00 / 1_000_000
in_cost = 2.50 / 1_000_000

class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_message(sender, msg, pid=None):
    pid = f"-{pid}" if pid else ""
    sender_color = {
        "user": color.PURPLE,
        "system": color.RED,
        "design": color.GREEN,
        "search": color.BLUE,
        "evaluation": color.DARKCYAN,
        "code": color.YELLOW,
        "manager": color.CYAN
    }
    sender_label = {
        "user": "💬 You:",
        "system": "⚠️ SYSTEM NOTICE ⚠️\n",
        "design": "🎨 Design Agent:",
        "search": f"🔍 Search Agent{pid}:",
        "evaluation": f"📊 Evaluation Agent{pid}:",
        "code": "🧑🏻‍💻 Code Agent:",
        "manager": "🤴🏻 MONAQ Manager"
    }

    msg = f"{color.BOLD}{sender_color[sender]}{sender_label[sender]}{color.END}{color.END} {msg}"
    print(msg)
    print()


def get_client(llm: str = "gpt-4o-mini"):
    if llm.startswith("gpt"):
        return OpenAI(api_key=AVAILABLE_LLMs[llm]["api_key"])
    else:
        return OpenAI(
            base_url=AVAILABLE_LLMs[llm]["base_url"],
            api_key=AVAILABLE_LLMs[llm]["api_key"],
        )


def quantize_model(model, x_train):
    seq_length = x_train.shape[1]
    n_features = x_train.shape[2]

    # save keras model with a specific signature function
    TEMP_DIR = "tmp_model"
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(
            [1, seq_length, n_features],
            "float32" if model.inputs == None else model.inputs[0].dtype,
        )
    )
    model.save(TEMP_DIR, save_format="tf", signatures=concrete_func)

    def _representative_dataset():
        for i in range(10):
            yield [tf.dtypes.cast([x_train[i]], tf.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(TEMP_DIR)
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce full-int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = _representative_dataset
    quantized_model = converter.convert()

    # remove temp. saved model files
    shutil.rmtree(TEMP_DIR)

    return quantized_model


def brief_profile_model(tflite_path):
    profile_result = profile_model(tflite_path, return_estimates=True).to_dict()[
        "summary"
    ]
    print(profile_result["tflite_size"] / 1e3)
    print(profile_result["runtime_memory_size"] / 1e3)
    print(profile_result["macs"])
    print(profile_result["time"] * 1e3)
    print("-" * 30)


def time_token_log(config, data_name, tokens, time_taken):
    with open("logs/time_token.csv", "a") as f:
        f.write(
            f"{config},{data_name},{time_taken},{tokens.prompt_tokens},{tokens.completion_tokens},{tokens.total_tokens}\n"
        )


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
