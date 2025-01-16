package imageutils

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg" // for saving images
	"os"
	// "yolo_detection/detector"

	// "yolo_detection/detector"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// Just copy these for debugging purposes
type Box struct {
    X1 float32
    Y1 float32
    X2 float32
    Y2 float32
}

type Detection struct {
    Box        Box
    Class      string
    Confidence float32
}




// DrawResult creates a debug image with boxes and labels
func DrawResult(img image.Image, detections []Detection, outputPath string) error {
    // Create new RGBA image
    bounds := img.Bounds()
    rgba := image.NewRGBA(bounds)
    draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)

    // Define colors for different classes (you can customize these)
    colors := map[string]color.RGBA{
        "cigarettes":         {R: 255, G: 0, B: 0, A: 255},    // Red
        "fresh_food_counter": {R: 0, G: 255, B: 0, A: 255},    // Green
        "generic_coffee":     {R: 0, G: 0, B: 255, A: 255},    // Blue
        "jack_daniels":       {R: 255, G: 255, B: 0, A: 255},  // Yellow
        "redbull":           {R: 255, G: 0, B: 255, A: 255},  // Magenta
        "toffifee":          {R: 0, G: 255, B: 255, A: 255},  // Cyan
    }

    // Draw each detection
    for _, det := range detections {
        // Get color for class
        c, ok := colors[det.Class]
        if !ok {
            c = color.RGBA{R: 255, G: 255, B: 255, A: 255} // White for unknown classes
        }

        // Draw rectangle
        drawRect(rgba, det.Box, c)

        // Draw label
        label := fmt.Sprintf("%s %.2f", det.Class, det.Confidence)
        drawLabel(rgba, label, det.Box, c)
    }

    // Save the image
    f, err := os.Create(outputPath)
    if err != nil {
        return fmt.Errorf("failed to create output file: %v", err)
    }
    defer f.Close()

    if err := jpeg.Encode(f, rgba, nil); err != nil {
        return fmt.Errorf("failed to encode image: %v", err)
    }

    fmt.Printf("Debug image saved to: %s\n", outputPath)
    return nil
}

// drawRect draws a rectangle on the image
func drawRect(img *image.RGBA, box Box, c color.RGBA) {
    // Draw horizontal lines
    for x := int(box.X1); x <= int(box.X2); x++ {
        img.Set(x, int(box.Y1), c)
        img.Set(x, int(box.Y2), c)
    }
    // Draw vertical lines
    for y := int(box.Y1); y <= int(box.Y2); y++ {
        img.Set(int(box.X1), y, c)
        img.Set(int(box.X2), y, c)
    }
}

// drawLabel draws text on the image
func drawLabel(img *image.RGBA, label string, box Box, c color.RGBA) {
    point := fixed.Point26_6{
        X: fixed.Int26_6(box.X1 * 64),
        Y: fixed.Int26_6((box.Y1 - 5) * 64), // 5 pixels above the box
    }

    d := &font.Drawer{
        Dst:  img,
        Src:  image.NewUniform(c),
        Face: basicfont.Face7x13,
        Dot:  point,
    }
    d.DrawString(label)
}