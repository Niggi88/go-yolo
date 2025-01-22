import onnxruntime
import numpy as np
from PIL import Image

# Load model
session = onnxruntime.InferenceSession("models/lower_cart_empty_loaded.onnx", providers=['CPUExecutionProvider'])

# Prepare input data
img = Image.open("../examples/images/bundle.jpeg")
img = img.resize((224, 224))
input_data = np.array(img).astype('float32')
input_data = input_data.transpose(2, 0, 1)  # HWC to CHW
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
input_data = input_data / 255.0  # Normalize

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
results = session.run([output_name], {input_name: input_data})

print("Input shape:", input_data.shape)
print("Output:", results[0])