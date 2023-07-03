//
// Created by Pramit Govindaraj on 5/19/2023.
//

#ifndef MATHOCR_UTILS_H
#define MATHOCR_UTILS_H

#include "OCREngine.h"

#include <boost/filesystem/path.hpp>
#include <boost/system/error_code.hpp>
#include <iostream>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <utility>

extern const int ALLOC_ERROR;

struct Options {};

extern Options settings;

struct RectComparator {
  bool operator()(const cv::Rect &rect1, const cv::Rect &rect2) const {
    if (rect1.y < rect2.y)
      return true;
    else if (rect1.y > rect2.y)
      return false;
    else
      return rect1.x < rect2.x;
  }
};

class ImageUtils {
public:
  ImageUtils() = default;
  ~ImageUtils();

  static cv::cuda::GpuMat toMat(const torch::Tensor &tensor, bool isNormalized, bool cvFormat);
  static torch::Tensor toTensor(const cv::cuda::GpuMat &matrix, torch::ScalarType size, int channels = 1);
  std::map<cv::Rect, ImageType, RectComparator> getImageBlocks(const cv::cuda::GpuMat &pixels);
  static void addMargin(const cv::cuda::GpuMat &pixels, cv::Rect_<int> &rect, int margin);
  float getSkewAngle(const cv::cuda::GpuMat &pixels, const ImageType &type);
  void rotate(cv::cuda::GpuMat &pixels, float degree);
  void denoise(cv::cuda::GpuMat &pixels);
  void crop(cv::cuda::GpuMat &pixels);
  void threshold(cv::cuda::GpuMat &pixels);
  void equalize(cv::cuda::GpuMat &pixels);

private:
  cv::Ptr<cv::cuda::Filter> thresholdBoxFilter = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(7, 7));
  cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1), cv::Point(-1, -1));
  cv::Ptr<cv::cuda::Filter> denoiseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, morphKernel);
  cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(85, 255);
  cv::Ptr<cv::cuda::HoughSegmentDetector> segmentDetector = cv::cuda::createHoughSegmentDetector(1, CV_PI / 180, 0, 20, 4096, 40);

  cv::cuda::GpuMat cropped;
  cv::cuda::GpuMat mean;
  cv::cuda::GpuMat edges;
  cv::cuda::GpuMat lines;
  cv::cuda::GpuMat rotated;
  cv::cuda::GpuMat resized;

  Classifier imgClassification;
};

void printHelp();

int clamp(int n, int lower, int upper);

bool fileExists(const char *filePath);

bool isDir(const char *path);

void winToNixFilePath(std::string &path);

void getPDFImages(const std::string &inputFilePath, const std::string &outputFilePath, OCREngine &ocr, ImageUtils& imgUtils, const std::function<std::string(OCREngine &, ImageUtils&, cv::cuda::GpuMat &, const std::string &)> &callback);

#endif//MATHOCR_UTILS_H
