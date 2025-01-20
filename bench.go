package main

import (
	"fmt"
	"image"
	"time"
	"yolo_detection/detector"
	"yolo_detection/imageutils"
)

// BenchmarkResult holds the timing measurements
type BenchmarkResult struct {
    InferenceTime time.Duration
    NumRuns       int
}

func measureInferenceTime(yolo *detector.YOLODetector, img image.Image, runs int) (BenchmarkResult, error) {
    var totalInferenceTime time.Duration

    // Do preprocessing once
    targetSize := imageutils.ImageSize{
        Width:  detector.DefaultConfig.InputWidth,
        Height: detector.DefaultConfig.InputHeight,
    }
    imageutils.PreprocessImage(img, targetSize)
    
    // Do one full detection to set up the tensors
    _, err := yolo.Detect(img)
    if err != nil {
        return BenchmarkResult{}, fmt.Errorf("initial detection failed: %v", err)
    }

    // Run just inference (session.Run) multiple times
    for i := 0; i < runs; i++ {
        inferenceStart := time.Now()
        err := yolo.RunInferenceOnly()  // Only running session.Run()
        if err != nil {
            return BenchmarkResult{}, fmt.Errorf("inference failed: %v", err)
        }
        inferenceTime := time.Since(inferenceStart)
        totalInferenceTime += inferenceTime

        // fmt.Printf("Run %d - Inference: %v\n", i+1, inferenceTime)
    }

    avgInferenceTime := totalInferenceTime / time.Duration(runs)
    
    return BenchmarkResult{
        InferenceTime: avgInferenceTime,
        NumRuns:      runs,
    }, nil
}