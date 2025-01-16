# go-yolo

### get onnx runtime

To download and extract ONNX Runtime, you can use the following commands:

- download

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xvf onnxruntime-linux-x64-1.20.0.tgz

```

- place in detector/

### model setup

- models go into ./examples/models
  - yolo model as object_detetion1.onnx
  - binary classifiers as \*name.pb

### setup test images

- test images go into ./examples/images
  - annotations \*name.txt
  - image \*name.jpeg
