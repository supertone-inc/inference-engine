from onnx import helper, TensorProto, OperatorSetIdProto

model_file = "../test-models/mat_mul.onnx"

input_names = ["A", "B"]
input_dims = [(2, 2), (2, 2)]
inputs = [
    helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
    for name, dims in zip(input_names, input_dims)
]

output_names = ["C"]
output_dims = [(2, 2)]
outputs = [
    helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
    for name, dims in zip(output_names, output_dims)
]

node = helper.make_node(
    op_type="MatMul",
    inputs=input_names,
    outputs=output_names,
)

graph = helper.make_graph(
    name="graph",
    nodes=[node],
    inputs=inputs,
    outputs=outputs,
)

model = helper.make_model(
    graph,
    ir_version=8,
    opset_imports=[OperatorSetIdProto(version=17)],
)

with open(model_file, "wb") as f:
    f.write(model.SerializeToString())
