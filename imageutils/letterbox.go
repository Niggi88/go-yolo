package imageutils

import (
	"image"
	"image/draw"
	"math"
)


func normalizeColor(color uint32) float32 {
    // First convert from 16-bit (0-65535) to 8-bit (0-255)
    color8bit := color >> 8
    // Then normalize to 0-1 range
    return float32(color8bit) / 255.0
}

func PreprocessImage(img image.Image, targetSize ImageSize) ([]float32, LetterboxParams) {
	// letterbox
	letterboxed, params := Letterbox(img, targetSize)

	// create tensor N C H W
	tensorData := make([]float32, 3*targetSize.Height*targetSize.Width)

	// Convert to float 32
	bounds := letterboxed.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x:= bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := letterboxed.At(x, y).RGBA()

			// in: uint32 [0, 65535]
			// to: float32 [0, 1]
			idx := y*targetSize.Width + x
			tensorData[idx] = normalizeColor(r)
			tensorData[idx + targetSize.Height*targetSize.Width] = normalizeColor(g)
			tensorData[idx + 2*targetSize.Height*targetSize.Width] = normalizeColor(b)
		}
	}

	return tensorData, params
}


func VerifyTensorData(data []float32) bool {
	for _, v := range data {
		if v < 0.0 || v > 1.0 {
			return false
		}
	}
	return true
}

func PreprocessBatch(imgs []image.Image, targetSize ImageSize) ([]float32, map[int]LetterboxParams){
	batchSize := len(imgs)
	params := make(map[int]LetterboxParams)

	// init tensor batch
	tensorData := make([]float32, batchSize*3*targetSize.Height*targetSize.Width)

	// process images
	for i, img := range imgs {
		// single img
		imgData, imgParams := PreprocessImage(img, targetSize)
		params[i] = imgParams

		// 
		offset := i * 3 * targetSize.Height * targetSize.Width
		copy(tensorData[offset:], imgData)
	}

	return tensorData, params
}


func Letterbox(img image.Image, targetSize ImageSize) (image.Image, LetterboxParams) {
	// original
	bounds := img.Bounds()
	originalHeight, originalWidth := bounds.Dy(), bounds.Dx()

	// scale to fit within
	scaleWidth := float64(targetSize.Width) / float64(originalWidth)
	scaleHeight := float64(targetSize.Height) / float64(originalHeight)
	scale := math.Min(scaleWidth, scaleHeight)
	
	// new dims
	newWidth := int(float64(originalWidth) * scale)
	newHeight := int(float64(originalHeight) * scale)

	// padding
	top := (targetSize.Height - newHeight) / 2
	left := (targetSize.Width - newWidth) / 2

	// black background
	dst := image.NewRGBA(image.Rect(0, 0, targetSize.Width, targetSize.Height))

	// temp image for resized content
	resized := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))

	// simple resizing using nearest neighbot (TODO: improve later)
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			
			// map new coordinates to original image
			origX := int(float64(x) / scale)
			origY := int(float64(y) / scale)

			// get color
			c := img.At(origX, origY)
			resized.Set(x, y, c)
		}
	}

	// draw resized on padded background
	draw.Draw(dst, image.Rect(left, top, left+newWidth, top+newHeight),
		resized, image.Point{}, draw.Src)

	params := LetterboxParams{
        Scale: scale,
        Left:  left,
        Top:   top,
    }

	return dst, params
}

func UnLetterbox(x, y float64, params LetterboxParams) (float64, float64){
	// remove padding
	unpadX := x - float64(params.Left)
	unpadY := y - float64(params.Top)

	// undo scale
	if params.Scale > 0 {
		unpadX /= params.Scale
		unpadY /= params.Scale
	}

	return unpadX, unpadY
}