# Go-YOLO

## Getting Started

### Download ONNX Runtime

To download and extract ONNX Runtime, run the following commands:

#### Download and Extract

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xvf onnxruntime-linux-x64-1.20.0.tgz
```

Move the extracted files to the `detector/` directory.

---

### Model Setup

1. **Place Models in the Correct Directory**  
   Models should be stored in `./examples/models`.

   - YOLO model: Save as `object_detection1.onnx`
   - Binary classifiers: Save as `*name.pb`

---

### Setup Test Images

1. **Store Test Images in the Correct Directory**  
   Test images should be saved in `./examples/images`.

2. **File Organization**
   - **Annotations:** Save as `*name.txt`
   - **Images:** Save as `*name.jpeg`
