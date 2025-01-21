package classifier

import (
	"fmt"
	"image"
	"os"
	"yolo_detection/imageutils"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// create new detector
func New(modelPath string) (*Classifier, error) {

	// cmake  libonnxruntime_providers_shared.so  libonnxruntime.so  libonnxruntime.so.1  libonnxruntime.so.1.20.0  pkgconfig
	onnxruntime.SetSharedLibraryPath("detector/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so")

	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer onnxruntime.DestroyEnvironment()

	// classes := []string {
	// 	"cigarettes", "fresh_food_counter", "generic_coffee", "jack_daniels", "redbull", "toffifee",
	// }

	// check if file exists
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model file not found: %v", err)
	}

	// pre-allocate input tensor 1, 3, h, w

	inputShape := onnxruntime.NewShape(1, 3, int64(DefaultConfig.InputHeight), int64(DefaultConfig.InputWidth))
	// inputTensor, err := onnxruntime.NewTensor[float32](inputShape)
	inputTensor, err := onnxruntime.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}

	// pre-allocate output tensor
	// YOLOv5 -> [1, num pred, num cl + 5]
	outputShape := []int64{1, 2} // Common YOLOv5 output shape
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	}

	// create ONNX runtime session
	session, err := onnxruntime.NewSession(
		modelPath,
		[]string{"x"},        // Run options
		[]string{"Identity"}, // Input names (empty means use default names)
		[]*onnxruntime.Tensor[float32]{inputTensor},  // Input tensors
		[]*onnxruntime.Tensor[float32]{outputTensor}, // Output tensors
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session :%v", err)
	}

	// input details
	// inputInfo := session.GetInputs()

	detector := &Classifier{
		modelPath:    modelPath,
		session:      session,
		config:       DefaultConfig,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}

	fmt.Printf("Initialized detector with model: %s\n", modelPath)
	fmt.Printf("Input shape: %v\n", inputShape)

	return detector, nil
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
	return nil
}

// RunInferenceOnly executes just the neural network session.Run() step
func (d *Classifier) RunInferenceOnly() error {
	return d.session.Run()
}

func (d *Classifier) Detect(img image.Image) ([]float32, error) {

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

	// run inference
	err := d.session.Run()
	if err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}

	// get output
	outputData := d.outputTensor.GetData()

	fmt.Println(outputData)

	// Convert back to original image coordinates using existing UnLetterbox

	return outputData, nil
}
