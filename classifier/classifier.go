package classifier

import (
	"fmt"
	"image"
	"os"
	"yolo_detection/imageutils"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// create new classifier
func New(modelPath string) (*Classifier, error) {
	INPUT_LAYER_NAME := "input_1:0"  // "x:0" //"x"
	OUTPUT_LAYER_NAME := "myOutput"// "Identity:0"

	// cmake  libonnxruntime_providers_shared.so  libonnxruntime.so  libonnxruntime.so.1  libonnxruntime.so.1.20.0  pkgconfig
	onnxruntime.SetSharedLibraryPath("detector/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so")
	
	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	// defer onnxruntime.DestroyEnvironment()


	// check if file exists
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model file not found: %v", err)
	}

	// pre-allocate input tensor 1, 3, h, w
	inputShape := onnxruntime.NewShape(1, 3, int64(DefaultConfig.InputHeight), int64(DefaultConfig.InputWidth))
	inputTensor, err := onnxruntime.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}

	// pre-allocate output tensor
	outputShape := []int64{1, 2} // Common YOLOv5 output shape
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	}

	// create ONNX runtime session
	session, err := onnxruntime.NewSession(
		modelPath,
		[]string{INPUT_LAYER_NAME},        // Run options
		[]string{OUTPUT_LAYER_NAME}, // Input names (empty means use default names)
		[]*onnxruntime.Tensor[float32]{inputTensor},  // Input tensors
		[]*onnxruntime.Tensor[float32]{outputTensor}, // Output tensors
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session :%v", err)
	}

	model := &Classifier{
		modelPath:    modelPath,
		session:      session,
		config:       DefaultConfig,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}

	return model, nil
}

// Close releases any resources
func (d *Classifier) Close() error {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.inputTensor != nil {
		d.inputTensor.Destroy()
	}
	if d.outputTensor != nil {
		d.outputTensor.Destroy()
	}
	onnxruntime.DestroyEnvironment()
	return nil
}

// RunInferenceOnly executes just the neural network session.Run() step
func (d *Classifier) RunInferenceOnly() error {
	return d.session.Run()
}

func (d *Classifier) Classify(img image.Image) ([]float32, error) {
	targetSize := imageutils.ImageSize{
		Width:  d.config.InputWidth,
		Height: d.config.InputHeight,
	}
	// tensorData, params := imageutils.PreprocessImage(img, targetSize)
	tensorData, _ := imageutils.PreprocessImage(img, targetSize)

	// verify data is valid
	if !imageutils.VerifyTensorData(tensorData) {
		return nil, fmt.Errorf("invalid tensor data after preprocessing")
	}

	// data to input tensor
	copy(d.inputTensor.GetData(), tensorData)
	// After copy to input tensor

	// run inference
	err := d.session.Run()
	if err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}
	// get output
	outputData := d.outputTensor.GetData()

	return outputData, nil
}
