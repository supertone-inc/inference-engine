import tensorflow as tf

model_file = "../test-models/matmul_dynamic.tflite"


class MatMulModelDynamic(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="A"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="B"),
        ]
    )
    def matmul(self, A, B):
        return tf.matmul(A, B)


# Create the model with dynamic shaped inputs
model = MatMulModelDynamic()

# Set up the converter
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [model.matmul.get_concrete_function()]
)
converter.experimental_new_converter = True  # Required for support of dynamic shapes
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Enable dynamic shapes
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_enable_resource_variables = True
converter.experimental_supports_dynamic_shapes = True

# Convert the dynamic shaped model
tflite_model = converter.convert()

# Save the TFLite model with dynamic shaped inputs
with open(model_file, "wb") as f:
    f.write(tflite_model)
