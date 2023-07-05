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
- [Latexpp (Katex)](https://github.com/goldsborough/latexpp)
- [v8 JS Engine](https://github.com/v8/v8)

### Models
- [YoloV8](https://github.com/ultralytics/ultralytics) - deskew
- [OCR](https://arxiv.org/ftp/arxiv/papers/1908/1908.11415.pdf)

### Datasets
- [Document Object Classification Dataset](https://universe.roboflow.com/mathocr/classify-e5mwr) - for deskew 
- [PDF to Latex Dataset] - TODO

### Usage
```
MathOCR train [dataDirectory] [epochs] [learningRate]
MathOCR preprocess [inputFile] [outputDir] [deskew]
MathOCR [inputFile] [outputDir]
```
