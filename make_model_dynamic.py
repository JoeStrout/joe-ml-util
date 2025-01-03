import onnx

path = input("Model path:" )
model = onnx.load(path)

def decode_onnx_shape(onnx_tensor):
    tensor_type = onnx_tensor.type.tensor_type
    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in tensor_type.shape.dim]
    return shape

# Modify input and output shape to allow dynamic spatial sizes
for tensor in model.graph.input:
	shape = tensor.type.tensor_type.shape
	shape.dim[-3].dim_param = 'D'  # Dynamic depth
	shape.dim[-2].dim_param = 'H'  # Dynamic height
	shape.dim[-1].dim_param = 'W'  # Dynamic width
	print(f'Updated {tensor.name} to shape {decode_onnx_shape(tensor)}')

for tensor in model.graph.output:
	shape = tensor.type.tensor_type.shape
	shape.dim[-3].dim_param = 'D_out'  # Dynamic depth for output
	shape.dim[-2].dim_param = 'H_out'  # Dynamic height for output
	shape.dim[-1].dim_param = 'W_out'  # Dynamic width for output
	print(f'Updated {tensor.name} to shape {decode_onnx_shape(tensor)}')

outpath = input("Output path: ")
if outpath:
	onnx.save(model, outpath)
	print(f'Wrote updated model to: {outpath}')

