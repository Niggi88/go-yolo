package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"os"
	"yolo_detection/detector"
	"yolo_detection/imageutils"
)


func main(){

	// load model
	yolo, err := detector.New("examples/models/object_detection1.onnx")
	if err != nil {
		fmt.Printf("Error initializing detector: %v\n", err)
		return
	}
	defer yolo.Close()

	// load image
	img, err := loadImage("examples/images/fresh_food_counter.jpeg")
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