package detector

import (
	"fmt"
	"image"
	"os"
	"sort"
	"yolo_detection/imageutils"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// create new detector
func New(modelPath string) (*YOLODetector, error){

	// cmake  libonnxruntime_providers_shared.so  libonnxruntime.so  libonnxruntime.so.1  libonnxruntime.so.1.20.0  pkgconfig
	onnxruntime.SetSharedLibraryPath("detector/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so")

	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer onnxruntime.DestroyEnvironment()

	classes := []string {
		"cigarettes", "fresh_food_counter", "generic_coffee", "jack_daniels", "redbull", "toffifee",
	}

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
	outputShape := []int64{1, 10647, int64(len(classes) + 5)}  // Common YOLOv5 output shape
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	}
	
	// create ONNX runtime session
	session, err := onnxruntime.NewSession(
        modelPath,
        []string{"images"},      // Run options
        []string{"output0"},      // Input names (empty means use default names)
        []*onnxruntime.Tensor[float32]{inputTensor},             // Input tensors
        []*onnxruntime.Tensor[float32]{outputTensor},             // Output tensors
    )
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session :%v", err)
	}

	// input details
	// inputInfo := session.GetInputs()


	detector := &YOLODetector{
        modelPath: modelPath,
        classes:   classes,
        session:   session,
        config:    DefaultConfig,
        inputTensor: inputTensor,
        outputTensor: outputTensor,
    }

	fmt.Printf("Initialized detector with model: %s\n", modelPath)
    fmt.Printf("Number of classes: %d\n", len(classes))
    fmt.Printf("Input shape: %v\n", inputShape)

    return detector, nil
}

// Close releases any resources
func (d *YOLODetector) Close() error {
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

func (d *YOLODetector) Detect(img image.Image) ([]Detection, error){

	targetSize := imageutils.ImageSize{
        Width:  d.config.InputWidth,
        Height: d.config.InputHeight,
    }
	// preprocessedImg, scale, padLeft, padTop := letterbox(img, targetHeight, targetWidth)
	// _, letterboxParams := imageutils.Letterbox(img, targetSize)
	tensorData, params := imageutils.PreprocessImage(img, targetSize)

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

	detections := d.processPredictions(outputData, params)
	fmt.Printf("fount %d detections before NMS\n", len(detections))

	detections = d.applyNMS(detections)
	fmt.Printf("fount %d detections before NMS\n", len(detections))
	
	 // Convert back to original image coordinates using existing UnLetterbox
    for i := range detections {
        x1, y1 := imageutils.UnLetterbox(float64(detections[i].Box.X1), float64(detections[i].Box.Y1), params)
        x2, y2 := imageutils.UnLetterbox(float64(detections[i].Box.X2), float64(detections[i].Box.Y2), params)
        
        detections[i].Box = Box{
            X1: float32(x1),
            Y1: float32(y1),
            X2: float32(x2),
            Y2: float32(y2),
        }
    }

    return detections, nil
}


// PROCESSING PREDICTIONS

func (d *YOLODetector) processPredictions(outputData []float32, params imageutils.LetterboxParams) []Detection {
	var detections []Detection

	// calculatte size of prediction
	stride := 5 + len(d.classes)
	numPreds := len(outputData) / stride

	// Process each prediction 

	for i := 0; i < numPreds; i++ {
		baseIdx := i * stride 

		// get box in grid
		x := outputData[baseIdx + 0]
		y := outputData[baseIdx + 1]
		w := outputData[baseIdx + 2]
		h := outputData[baseIdx + 3]

		// objectness
		objectness := outputData[baseIdx + 4]

		// best class
		bestClassScore := float32(-1)
		bestClassIdx := 0
		for j := 0; j < len(d.classes); j++{
			score := outputData[baseIdx + 5 + j]
			if score > bestClassScore {
				bestClassScore = score
				bestClassIdx = j
			}
		}

		confidence := objectness * bestClassScore

		if confidence > d.config.ConfThreshold {
			x1 := float32(x - w/2)
			x2 := float32(x + w/2)
			y1 := float32(y - h/2)
			y2 := float32(y + h/2)

			detection := Detection{
				Box: Box{
					X1: x1,
					Y1: y1,
					X2: x2,
					Y2: y2,
				},
				Class : d.classes[bestClassIdx],
				Confidence: confidence,
			}

			detections = append(detections, detection)
		}
	}

	fmt.Printf("Found %d detections above confidence threshold %.2f\n", 
        len(detections), d.config.ConfThreshold)
    
    return detections
}


// NON MAX SUPPRESSION
func calculateIoU(box1, box2 Box) float32 {
    x1 := max(box1.X1, box2.X1)
    y1 := max(box1.Y1, box2.Y1)
    x2 := min(box1.X2, box2.X2)
    y2 := min(box1.Y2, box2.Y2)

	if x1 > x2 || y1 > y2 {
		return 0
	}

	intersection := (x2 - x1) * (y2 - y1)

	box1Area := (box1.X2 - box1.X1) * (box1.Y2 - box1.Y1)
	box2Area := (box2.X2 - box2.X1) * (box2.Y2 - box2.Y1)
	
	union := box1Area + box2Area - intersection

	return intersection / union
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func (d *YOLODetector) applyNMS(detections []Detection) []Detection {
	// no detection -> do nothing
	if len(detections) == 0 {
		return detections
	}

	// sort by confidence
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Confidence > detections[j].Confidence
	})

	var result []Detection
	selected := make(map[int]bool)

	for i := 0; i < len(detections); i++ {
		if selected[i]{
			continue
		}

		result = append(result, detections[i])
		selected[i] = true

		for j := i + 1; j < len(detections); j++ {
			if selected[j]{
				continue
			}

			if detections[i].Class == detections[j].Class &&
				calculateIoU(detections[i].Box, detections[j].Box) > d.config.ConfThreshold {
					selected[j] = true
				}
		}
	}
	return result
}

