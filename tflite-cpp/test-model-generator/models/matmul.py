import tensorflow as tf

model_file = "../test-models/matmul.tflite"


class MatMulModel(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2, 2), dtype=tf.float32, name="A"),
            tf.TensorSpec(shape=(2, 2), dtype=tf.float32, name="B"),
        ]
    )
    def matmul(self, A, B):
        return tf.matmul(A, B)


# Create the model
model = MatMulModel()

# Set up the converter
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [model.matmul.get_concrete_function()]
)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(model_file, "wb") as f:
    f.write(tflite_model)
