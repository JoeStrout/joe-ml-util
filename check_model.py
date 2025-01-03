# This script takes a path (local file or gs: path) to an ONNX model, loads that
# model (as an ONNX Runtime session), and then prints some basic info about it.

import onnx
from onnx import TensorProto
import io
from google.cloud import storage
from cloudfiles import CloudFile
import sys

# Mapping of tensor element types to readable names
onnx_dtype_map = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.COMPLEX64: "complex64",
    TensorProto.COMPLEX128: "complex128",
    TensorProto.BFLOAT16: "bfloat16",
}

def load_model(model_path):
    cf = CloudFile(model_path)
    print(f'Loaded file ({cf.size()} bytes) from {model_path}')
    return onnx.load(io.BytesIO(cf.get()))

def load_inference_session(model_path):
    cf = CloudFile(model_path)
    print(f'Loaded file ({cf.size()} bytes) from {model_path}')
    return onnxruntime.InferenceSession(cf.get())

def decode_onnx_shape(onnx_tensor):
    tensor_type = onnx_tensor.type.tensor_type
    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in tensor_type.shape.dim]
    return shape

def find_shape(tensor_name, graph):
    """
    Retrieve the shape of a tensor by matching its name in the graph's inputs, outputs, or value_info.
    """
    # Check in inputs, outputs, and value_info
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.name == tensor_name:
            return decode_onnx_shape(value_info)
    return 'Unknown'

def summarize_node(node, graph):
    """
    Generate a summary string for a given ONNX graph node, showing input and output shapes.
    """
    input_shapes = [str(find_shape(name, graph)) for name in node.input]
    output_shapes = [str(find_shape(name, graph)) for name in node.output]
    input_shapes_str = ', '.join(input_shapes)
    output_shapes_str = ', '.join(output_shapes)
    return f"{node.name}: {node.op_type} ({input_shapes_str}) -> {output_shapes_str}"

def compute_receptive_field(model):
    """
    Estimate the receptive field size of the model by tracking convolution and pooling layers.
    This version handles 3D convolutions separately for X, Y, and Z dimensions.
    """
    if not model.graph.value_info:
        model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    receptive_field = [1, 1, 1]  # For Z, Y, X
    for node in graph.node:
        if node.op_type in ["Conv", "MaxPool"]:
            for attr in node.attribute:
                if attr.name == "kernel_shape" and len(attr.ints) == 3:
                    kernel_size = attr.ints  # Z, Y, X order
                    for i in range(3):
                        receptive_field[i] += (kernel_size[i] - 1)
    return receptive_field

def check_model(model: onnx.ModelProto):
    # Infer the intermediate node shapes, if the model doesn't already have them
    if not model.graph.value_info:
        model = onnx.shape_inference.infer_shapes(model)

    # Summarize each node in the graph
    print('Model Structure:\n')
    for node in model.graph.node:
        print(summarize_node(node, model.graph))
    print()

    # Check the model's inputs
    for input in model.graph.input:
        input_name = input.name
        input_shape = decode_onnx_shape(input)
        input_type = onnx_dtype_map.get(input.type.tensor_type.elem_type, "unknown")
        print(f'Input: {input_name}, Shape: {input_shape}, Type: {input_type}')


    # Check the model's outputs
    for output in model.graph.output:
        output_name = output.name
        output_shape = decode_onnx_shape(output)
        output_type = onnx_dtype_map.get(output.type.tensor_type.elem_type, "unknown")
        print(f'Output: {output_name}, Shape: {output_shape}, Type: {output_type}')

    rf = compute_receptive_field(model)
    print(f"Estimated receptive field: {rf}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_model.py <model_path>")
        print("  <model_path> may be a local path, or a gs: URL")
        sys.exit(1)

    model_path = sys.argv[1]
    model = load_model(model_path)
    check_model(model)
