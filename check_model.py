# This script takes a path (local file or gs: path) to an ONNX model, loads that
# model (as an ONNX Runtime session), and then prints some basic info about it.

import onnxruntime
import io
from google.cloud import storage
from cloudfiles import CloudFile
import sys

def load_model(model_path):
    cf = CloudFile(model_path)
    print(f'Loaded file ({cf.size()} bytes) from {model_path}')
    return onnxruntime.InferenceSession(cf.get())

def check_model(model):
    # Check the model's inputs
    for input in model.get_inputs():
        input_name = input.name
        input_shape = input.shape
        input_type = input.type
        print(f'Input: {input_name}, Shape: {input_shape}, Type: {input_type}')

    # Check the model's outputs
    for output in model.get_outputs():
        output_name = output.name
        output_shape = output.shape
        output_type = output.type
        print(f'Output: {output_name}, Shape: {output_shape}, Type: {output_type}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_model.py <model_path>")
        print("  <model_path> may be a local path, or a gs: URL")
        sys.exit(1)

    model_path = sys.argv[1]
    model = load_model(model_path)
    check_model(model)
