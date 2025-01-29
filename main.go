package main

import (
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	"os"
	"time"
	"yolo_detection/classifier"
	"yolo_detection/detector"
	"yolo_detection/imageutils"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

func main() {
	// START-SCOPE

	// cmake  libonnxruntime_providers_shared.so  libonnxruntime.so  libonnxruntime.so.1  libonnxruntime.so.1.20.0  pkgconfig
	onnxruntime.SetSharedLibraryPath("detector/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so")
	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		panic(err)
	}

	defer func() {
		onnxruntime.DestroyEnvironment()
		fmt.Println("Destroy Environment")
	}()

	ctx, cancel := context.WithCancel(context.Background())
	RunClassifier(ctx)
	cancel()
	ctx, cancel = context.WithCancel(context.Background())
	RunDetector((ctx))
	cancel()

	// RunDetector()

	// END-SCOPE
	// -> cancel() -> inputTensor.Destroy, outputTensor.Destroy(), session.Destroy()
	// -> onnxruntime.DestroyEnvironment()

	time.Sleep(10 * time.Second)
}

func measureImageDetectionTime(yolo *detector.YOLODetector, imagePath string, runs int) (time.Duration, time.Duration, []detector.Detection, error) {
	var totalLoadTime, totalDetectTime time.Duration
	var lastDetections []detector.Detection

	for i := 0; i < runs; i++ {
		// Measure image loading time
		loadStart := time.Now()
		img, err := loadImage(imagePath)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("error loading image: %v", err)
		}
		loadTime := time.Since(loadStart)
		totalLoadTime += loadTime

		// Measure detection time
		detectStart := time.Now()
		detections, err := yolo.Detect(img)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("error running detection: %v", err)
		}
		detectTime := time.Since(detectStart)
		totalDetectTime += detectTime

		// Store last run's detections
		lastDetections = detections

		// fmt.Printf("Run %d - Load: %v, Detect: %v\n", i+1, loadTime, detectTime)
	}

	avgLoadTime := totalLoadTime / time.Duration(runs)
	avgDetectTime := totalDetectTime / time.Duration(runs)
	fmt.Println("Average load time:", avgLoadTime)
	fmt.Println("Average detect time: ", avgDetectTime)
	return avgLoadTime, avgDetectTime, lastDetections, nil
}

func measureImageClassificationTime(model *classifier.Classifier, imagePath string, runs int) (time.Duration, time.Duration, []float32, error) {
	var totalLoadTime, totalClassificationTime time.Duration
	var lastClassifiactions []float32

	for i := 0; i < runs; i++ {
		// Measure image loading time
		loadStart := time.Now()
		img, err := loadImage(imagePath)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("error loading image: %v", err)
		}
		loadTime := time.Since(loadStart)
		totalLoadTime += loadTime

		// Measure detection time
		classifyStart := time.Now()
		classifications, err := model.Classify(img)
		if err != nil {
			return 0, 0, nil, fmt.Errorf("error running detection: %v", err)
		}
		detectTime := time.Since(classifyStart)
		totalClassificationTime += detectTime

		// Store last run's detections
		lastClassifiactions = classifications

		// fmt.Printf("Run %d - Load: %v, Detect: %v\n", i+1, loadTime, detectTime)
	}

	avgLoadTime := totalLoadTime / time.Duration(runs)
	avgDetectTime := totalClassificationTime / time.Duration(runs)
	fmt.Println("Average load time:", avgLoadTime)
	fmt.Println("Average classification time: ", avgDetectTime)
	return avgLoadTime, avgDetectTime, lastClassifiactions, nil
}

func RunDetector(ctx context.Context) {
	imagePath := "examples/images/fresh_food_counter.jpeg"
	modelPath := "examples/models/object_detection1.onnx"
	runs := 10

	// load model
	yolo, err := detector.New(ctx, modelPath)
	if err != nil {
		fmt.Printf("Error initializing detector: %v\n", err)
		return
	}
	defer yolo.Close()
	// *detector.YOLODetector
	measureImageDetectionTime(yolo, imagePath, runs)

	// load image
	img, err := loadImage(imagePath)
	if err != nil {
		fmt.Printf("Error loading image %v\n", err)
		return
	}

	// Run benchmark
	result, err := measureInferenceTimeYolo(yolo, img, 100)
	if err != nil {
		fmt.Printf("Error during benchmark: %v\n", err)
		return
	}

	fmt.Printf("\nBenchmark Results:\n")
	fmt.Printf("Average inference time over %d runs: %v\n",
		result.NumRuns, result.InferenceTime)

	// run detection
	detections, err := yolo.Detect(img)
	if err != nil {
		fmt.Printf("Error running detectioon: %v\n", err)
		return
	}

	// print results
	for _, det := range detections {
		fmt.Printf("Found %s  (confidence %.2f) ad box: %+v\n",
			det.Class, det.Confidence, det.Box)
	}

	if err := DrawDebug(img, detections, "debug_output.jpg"); err != nil {
		fmt.Printf("Failed to save debug image: %v\n", err)
	}
}

func RunClassifier(ctx context.Context) {
	imagePath := "examples/images/bundle.jpeg"
	modelPath := "pbtf2onnx/models/lower_cart_empty_loaded.onnx"
	runs := 10

	model, err := classifier.New(ctx, modelPath)
	if err != nil {
		fmt.Printf("Error initializing detector: %v\n", err)
		return
	}
	measureImageClassificationTime(model, imagePath, runs)

	// load image
	img, err := loadImage(imagePath)
	if err != nil {
		fmt.Printf("Error loading image %v\n", err)
		return
	}

	result, err := measureInferenceTimeBinary(model, img, 100)
	if err != nil {
		fmt.Printf("Error during benchmark: %v\n", err)
		return
	}
	fmt.Printf("\nBenchmark Results:\n")
	fmt.Printf("Average inference time over %d runs: %v\n",
		result.NumRuns, result.InferenceTime)
	// run detection
	classifications, err := model.Classify(img)
	if err != nil {
		fmt.Printf("Error running detectioon: %v\n", err)
		return
	}
	fmt.Println(classifications)

}

func loadImage(filepath string) (image.Image, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	return img, err
}

func DrawDebug(img image.Image, detections []detector.Detection, outputPath string) error {
	debugDetections := make([]imageutils.Detection, len(detections))
	for i, det := range detections {
		debugDetections[i] = imageutils.Detection{
			Box: imageutils.Box{
				X1: det.Box.X1,
				Y1: det.Box.Y1,
				X2: det.Box.X2,
				Y2: det.Box.Y2,
			},
			Class:      det.Class,
			Confidence: det.Confidence,
		}
	}
	return imageutils.DrawResult(img, debugDetections, outputPath)
}
