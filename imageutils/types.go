package imageutils

// LetterboxParams stores the transformation parameters
type LetterboxParams struct {
    Scale float64
    Left  int
    Top   int
}

// ImageSize defines target dimensions for processing
type ImageSize struct {
    Width  int
    Height int
}