#!/bin/bash

# python -m tf2onnx.convert --input ./models/lower_cart_empty_loaded.pb --output ./models/lower_cart_empty_loaded.onnx --inputs x:0 --outputs Identity:0
python -m tf2onnx.convert --input ./models/lower_cart_empty_loaded.pb --output ./models/lower_cart_empty_loaded.onnx --inputs x:0 --outputs Identity:0

# output layer name: Identity
#   or myOutput/Softmax -> eher der
# input layer name: x
