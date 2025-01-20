package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"os"
	"time"
	"yolo_detection/detector"
	"yolo_detection/imageutils"
)


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

        fmt.Printf("Run %d - Load: %v, Detect: %v\n", i+1, loadTime, detectTime)
    }

    avgLoadTime := totalLoadTime / time.Duration(runs)
    avgDetectTime := totalDetectTime / time.Duration(runs)
	fmt.Println("Average load time:", avgLoadTime)
	fmt.Println("Average detect time: ", avgDetectTime)
    return avgLoadTime, avgDetectTime, lastDetections, nil
}


func main(){
	imagePath := "examples/images/fresh_food_counter.jpeg"
	modelPath := "examples/models/object_detection1.onnx"
	runs := 10

	// load model
	yolo, err := detector.New(modelPath)
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