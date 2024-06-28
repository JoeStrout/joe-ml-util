# This script takes a path (local file or gs: path) to an ONNX model,
# loads that model, and then prints some basic info about it.

import onnx
import io
from google.cloud import storage
import sys

def download_model_from_gcs(gcs_path):
    bucket_name = gcs_path.split('/')[2]
    source_blob_name = '/'.join(gcs_path.split('/')[3:])
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    model_data = blob.download_as_bytes()
    model = onnx.load_model(io.BytesIO(model_data))
    return model

def load_model(model_path):
    if model_path.startswith("gs:"):
        print(f"Loading model from Google Cloud Storage: {model_path}")
        model = download_model_from_gcs(model_path)
    else:
        print(f"Loading model from local file system: {model_path}")
        model = onnx.load(model_path)
    return model

def check_model(model):
    # Check the model's inputs
    for input in model.graph.input:
        input_name = input.name
        input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        input_type = input.type.tensor_type.elem_type
        print(f'Input: {input_name}, Shape: {input_shape}, Type: {input_type}')

    # Check the model's outputs
    for output in model.graph.output:
        output_name = output.name
        output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        output_type = output.type.tensor_type.elem_type
        print(f'Output: {output_name}, Shape: {output_shape}, Type: {output_type}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_model.py <model_path>")
        print("  <model_path> may be a local path, or a gs: URL")
        sys.exit(1)

    model_path = sys.argv[1]
    model = load_model(model_path)
    check_model(model)
