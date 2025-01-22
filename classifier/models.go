package classifier

import (
	onnxruntime "github.com/yalue/onnxruntime_go"
)

// bounding box
type Box struct {
	X1 float32
	Y1 float32
	X2 float32
	Y2 float32
}

// detection (one box with class and confidence)
type Classification struct {
	Class	string
	Confidence float32
}

type Classifier struct {
	modelPath 	string
	session 	*onnxruntime.Session[float32]
	config		Config
    inputTensor   *onnxruntime.Tensor[float32]
    outputTensor  *onnxruntime.Tensor[float32]
}

type Config struct {
	InputWidth 		int
	InputHeight 	int
	ConfThreshold 	float32
	IOUThreshold 	float32
}

var DefaultConfig = Config{
	InputWidth: 	224,
	InputHeight: 	224,
	ConfThreshold:  0.25,
}