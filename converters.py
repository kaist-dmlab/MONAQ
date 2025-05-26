import tensorflow as tf


def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model


def quantize_model(model, x_train):
    def _representative_dataset():
        for i in range(20):
            yield [tf.dtypes.cast([x_train[i]], tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce full-int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = _representative_dataset
    quantized_model = converter.convert()

    return quantized_model
