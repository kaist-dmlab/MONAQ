# length of ts numerical values / class = 100 samples/observations

import pandas as pd
import numpy as np

from utils.data_desc import feature_descriptions, data_contexts
from glob import glob

DATA_PROMPTS = {
    "BinaryHeartbeat": """I need a model to classify heartbeat signals, intended for deployment on an edge device with 1 MB of storage and 128 KB of RAM. Since this is a critical healthcare task, the model must be highly accurate while maintaining a very low inference latency of under 100 ms.""",
    "AtrialFibrillation": """I have a dataset of ECG records and want to build a classification model to categorize ECG signals into three types of atrial fibrillation. The model should be deployable on wearable devices, such as Fitbit trackers.""",
    "Cricket": """I want a model that can classify cricket umpire signals based on 3-axis accelerometer data from both hands. Since this model needs to run in real-time on a device during competitions, it should be as compact as possible while maintaining acceptable accuracy.""",
    "FaultDetectionA": """We have a time series dataset collected from an electromechanical drive system. Create a model for deployment on edge devices to identify types of damage in rolling bearings.""",
    "UCIHAR": """I have 3-axis body linear acceleration signals collected for human activity recognition. I need a classifier that can run on wearable devices with 1 MB of RAM and 2 MB of flash storage. The inference latency should not exceed 500 ms.""",
    "AppliancesEnergy": """I have an IoT device collecting appliance energy data from a house.
Please develop a predictive model to forecast the total energy consumption in kWh for the house.
Additionally, the model should be compact enough to be deployed on a ZigBee wireless sensor network.""",
    "LiveFuelMoistureContent": """Build a regression model to predict the moisture content in vegetation. The model should be deployable on a small device with 512 KB of RAM and 1 MB of storage. As this will be used in a smart farming context, the prediction speed should be under 1000 ms.""",
    "BenzeneConcentration": """We aim to develop a model to predict benzene concentrations in an Italian city based on air quality measurements. This model will be deployed on IoT sensors using the Arduino Nano 33 BLE, so it should be compact and achieve a very low error rate, ideally with an RMSE of 1.00 or lower.""",
    "BIDMC32SpO2": """Our company has a project to deploy a predictive model on wearable devices, such as fitness trackers, to estimate blood oxygen saturation levels using PPG and ECG data. 
Please create a lightweight model suitable for deployment on these devices. The model should use no more than 32KB of RAM and be no larger than 64KB in size.""",
    "FloodModeling": """I have an IoT sensor monitoring rainfall events. Could you develop a model to predict the maximum water depth for flood modeling? The model should be lightweight enough to run on the sensor and provide real-time predictions.""",
}


# numerical time series	textual descriptions	image features
# /
def complete_by_values(data_name, task, max_length=30):
    file_names = glob(f"ts_values/{data_name}/*_mean.npy")

    if task == "classification":
        prompt = f"""{DATA_PROMPTS[data_name]}

Additionally, please consider the following dataset features when building the model:
{feature_descriptions[data_name]}

Representative samples of the dataset are provided below. The data is in CSV format, including a header row, with columns representing features and rows containing observations for each timestamp.
"""
    else:
        prompt = f"""{DATA_PROMPTS[data_name]}

Additionally, please consider the following dataset features when building the model:
{feature_descriptions[data_name]}

Representative samples of the dataset are provided below. The data is in CSV format, including a header row, with columns representing features and rows containing observations for each timestamp.
"""

    for fname in file_names:
        ts_values = pd.DataFrame(np.load(fname)[:max_length].round(2)).to_csv(
            index=False
        )
        label = fname.split("/")[-1].split("_")[0]
        prompt = f"""{prompt}
Data of {"Class: " + label if task == 'classification' else 'Label range: ' + label}
{ts_values}
----------
"""
    return prompt


# 	/
def complete_by_contexts(data_name):
    return f"""{DATA_PROMPTS[data_name]}

Additionally, please consider the following dataset features when building the model:
{feature_descriptions[data_name]}

More contextual information is also provided below.
{data_contexts[data_name]}
"""


# 		/
def complete_by_images(data_name):
    return f"""{DATA_PROMPTS[data_name]}

Additionally, please consider the following dataset features when building the model:
{feature_descriptions[data_name]}

Also, analyze the patterns in the images uploaded as time series plots to understand the underlying properties of the dataset before building the model.
"""


def complete_mm_prompt(data_name, query_type, task, max_length=30):
    file_names = glob(f"ts_values/{data_name}/*_mean.npy")

    if task == "classification":
        prompt = f"""{DATA_PROMPTS[data_name]}

Additionally, please consider the following dataset features when building the model:
{feature_descriptions[data_name]}

More contextual information is also provided below.
{data_contexts[data_name]}

Representative samples of the dataset are provided below. The data is in CSV format, including a header row, with columns representing features and rows containing observations for each timestamp.
"""
    else:
        prompt = f"""{DATA_PROMPTS[data_name]}

Additionally, please consider the following dataset features when building the model:
{feature_descriptions[data_name]}

More contextual information is also provided below.
{data_contexts[data_name]}

Representative samples of the dataset are provided below. The data is in CSV format, including a header row, with columns representing features and rows containing observations for each timestamp.
"""
    for fname in file_names:
        ts_values = pd.DataFrame(np.load(fname)[:max_length].round(2)).to_csv(
            index=False
        )
        label = fname.split("/")[-1].split("_")[0]
        prompt = f"""{prompt}
Data of {"Class: " + label if task == 'classification' else 'Label range: ' + label}
{ts_values}
----------
"""
    # /	/
    # /	/	/
    if "time" in query_type and "text" in query_type and "image" in query_type:
        return f"""{prompt}
Also, analyze the patterns in the images uploaded as time series plots to understand the underlying properties of the dataset before building the model.
"""
    return prompt
