#!/bin/bash

# python -m tf2onnx.convert --input ./models/lower_cart_empty_loaded.pb --output ./models/lower_cart_empty_loaded.onnx --inputs x:0 --outputs Identity:0
# python -m tf2onnx.convert \
#     --input ./models/lower_cart_empty_loaded.pb \
#     --output ./models/lower_cart_empty_loaded.onnx \
#     --inputs x:0 \
#     --outputs Identity:0 \
#     --inputs-as-nchw x:0\


# output layer name: Identity
#   or myOutput/Softmax -> eher der
# input layer name: x
python -m tf2onnx.convert \
    --input ./models/lower_cart_empty_loaded.pb \
    --output ./models/lower_cart_empty_loaded.onnx \
    --inputs "x:0[1,224,224,3]" \
    --inputs-as-nchw x:0 \
    --outputs Identity:0

# [1,224,224,3]
# worked
# python -m tf2onnx.convert \
#     --saved-model ./models/lower_cart_empty_loaded \
#     --output ./models/lower_cart_empty_loaded.onnx \
#     --inputs "input_1:0[1,224,224,3]" \
#     --inputs-as-nchw input_1:0 \