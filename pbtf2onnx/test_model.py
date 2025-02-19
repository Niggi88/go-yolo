import onnxruntime
import numpy as np
from PIL import Image



model_name = "/home/niklas/code/imageClassifierService/networks/binary.onnx"
image_name1 = "/home/niklas/code/imageClassifierService/testImages/bgr_rgb_0_127_255.jpeg"
image_name2 = "/home/niklas/code/imageClassifierService/testImages/bundle_loaded.jpeg"

# Load model
session = onnxruntime.InferenceSession(model_name, providers=['CPUExecutionProvider'])

def load_proc(path):

    # Prepare input data
    img = Image.open(path)
    img.save('input_python.png')

    img = img.resize((224, 224))
    img.save('DEFAULT_python.png')
    input_data = np.array(img).astype('float32')
    input_data = input_data.transpose(2, 0, 1)  # HWC to CHW
    # input_data = input_data.reshape(3, 224, 224)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    input_data = input_data / 255.0  # Normalize
    return input_data

img1 = load_proc(image_name1)
print("\n", img1.flatten()[50170:50190])

img2 = load_proc(image_name2)
print("\n", img2.flatten()[50170:50190])
# print(input_data)
exit()
# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print([input.name for input in session.get_inputs()])
print([input.name for input in session.get_outputs()])
results = session.run([output_name], {input_name: img1})

print("Input shape:", img1.shape)
print("Output:", results[0])