# import utilitiy packages
import os, sys, gc, warnings, logging, shutil
import json, time, glob, math

# determine GPU number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO and WARNING messages

# define paths to model files
MODELS_DIR = "models/"
MODEL_TF = MODELS_DIR + "model.pb"
MODEL_NO_QUANT_TFLITE = MODELS_DIR + "model_no_quant.tflite"
MODEL_TFLITE_MICRO = MODELS_DIR + "model.cc"
SEED = 7

os.makedirs(MODELS_DIR, exist_ok=True)

logging.disable(logging.WARNING)
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

# import basic libraries
import random

import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import keras

# Set a "seed" value, so we get the same random numbers each time we run this notebook for reproducible results.
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from utils.data_loader import load_dataset
from utils.data_desc import AVAILABEL_DATASETS, CLS_DATASETS, REG_DATASETS
from utils import quantize_model, brief_profile_model

# Do not change this
from sklearn.metrics import root_mean_squared_error

N_EPOCHS = 100
BATCH_SIZE = 32
task = "regression"

keras.backend.clear_session()

data_name = os.path.basename(__file__).split(".")[0]  # or replace with the user given dataset name

# 1. Loading the Target Dataset
X_train, y_train, X_test, y_test = load_dataset(data_name, task)
print("Experiment on:", data_name, X_train.shape)
seq_length = X_train.shape[1]
n_features = X_train.shape[2]


# 2. Design the Model
def get_model():
    # TODO: Define a Tensorflow/Keras compatible model based on the given configurations
    # Note that your model will be converted to a TFLite Micro model
    return your_model


model = get_model()
model.compile(optimizer="adam", loss="mean_squared_error", metrics=keras.metrics.RootMeanSquaredError(name="rmse", dtype=None))
es = keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=10, restore_best_weights=True)

# 3. Train the Model
model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[es])

# 4. Evaluate the Model and Save Results (Do not change this)
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)

# 5. Convert model to TFLite model
quantized_model = quantize_model(model, X_train)
# Save the model to disk
MODEL_TFLITE = MODELS_DIR + f"{model.name}_{task}_{data_name}.tflite"
open(MODEL_TFLITE, "wb").write(quantized_model)

# 6. Profile the converted model with a simulator
print(model.name, data_name)
print(rmse)
brief_profile_model(MODEL_TFLITE)

del model
keras.backend.clear_session()
gc.collect()
