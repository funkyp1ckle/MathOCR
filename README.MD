# MathOCR
[![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](http://goldsborough.mit-license.org)

Converts PDFs and images into tex

### Dependencies
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [CUDNN](https://developer.nvidia.com/cudnn)
- [Boost](https://www.boost.org/users/download/)
- [Libtorch](https://pytorch.org/get-started/locally/) 
- [Torch-TensorRT](https://github.com/pytorch/TensorRT)
- [OpenCV](https://github.com/opencv/opencv) with [Contrib(for CUDA)](https://github.com/opencv/opencv_contrib)
- [Ghostscript](https://ghostscript.com/releases/gsdnld.html)
- [Katex](https://github.com/KaTeX/KaTeX)
- [v8 JS Engine](https://github.com/v8/v8)
- [wkhtmltox](https://wkhtmltopdf.org/downloads.html)
- [Tesseract](https://github.com/tesseract-ocr/tesseract)

### Models
- [YoloV8](https://github.com/ultralytics/ultralytics) - deskew
- [OCR](https://arxiv.org/ftp/arxiv/papers/1908/1908.11415.pdf)

### Datasets
- [Document Object Classification Dataset](https://universe.roboflow.com/mathocr/classify-e5mwr) - for deskew 
- [Math Image to Latex Dataset](https://zenodo.org/record/56198#.V2px0jXT6eA)

### Usage
```
MathOCR train [dataDirectory] [epochs] [learningRate]
MathOCR preprocess [inputFile] [outputDir] [deskew]
MathOCR [inputFile] [outputDir]
```
