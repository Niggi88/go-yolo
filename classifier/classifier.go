package classifier

import (
	"context"
	"fmt"
	"image"
	"os"
	"yolo_detection/imageutils"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// create new classifier
func New(ctx context.Context, modelPath string) (*Classifier, error) {
	fmt.Println("SCOPE: Classifier.New")
	defer fmt.Println("SCOPE: Classifier.New END")

	INPUT_LAYER_NAME := "input"
	OUTPUT_LAYER_NAME := "empty_loaded"
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
	go func() {
		fmt.Println("Waiting Destroy inputTensor")
		<-ctx.Done()
		inputTensor.Destroy()
		fmt.Println("Destroy inputTensor")
	}()

	// pre-allocate output tensor
	outputShape := []int64{1, 2} // 2 possible classes
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	}
	go func() {
		fmt.Println("Waiting Destroy outputTensor")
		<-ctx.Done()
		outputTensor.Destroy()
		fmt.Println("Destroy outputTensor")
	}()

	// create ONNX runtime session
	session, err := onnxruntime.NewSession(
		modelPath,
		[]string{INPUT_LAYER_NAME},                   // Run options
		[]string{OUTPUT_LAYER_NAME},                  // Input names (empty means use default names)
		[]*onnxruntime.Tensor[float32]{inputTensor},  // Input tensors
		[]*onnxruntime.Tensor[float32]{outputTensor}, // Output tensors
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session :%v", err)
	}
	go func() {
		fmt.Println("Waiting destroy session")
		<-ctx.Done()
		session.Destroy()
		fmt.Println("Destroy session")
	}()

	model := &Classifier{
		modelPath:    modelPath,
		session:      session,
		config:       DefaultConfig,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}

	return model, nil
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
