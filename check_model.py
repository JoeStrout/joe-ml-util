# This script takes a path (local file or gs: path) to an ONNX model, loads that
# model (as an ONNX Runtime session), and then prints some basic info about it.

import onnxruntime
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
    model = onnxruntime.InferenceSession(io.BytesIO(model_data).read())
    return model

def load_model(model_path):
    if model_path.startswith("gs:"):
        print(f"Loading model from Google Cloud Storage: {model_path}")
        model = download_model_from_gcs(model_path)
    else:
        print(f"Loading model from local file system: {model_path}")
        model = onnxruntime.InferenceSession(model_path)
    return model

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
