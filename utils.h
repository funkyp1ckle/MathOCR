//
// Created by Pramit Govindaraj on 5/19/2023.
//

#ifndef MATHOCR_UTILS_H
#define MATHOCR_UTILS_H

#include "OCREngine.h"

#include <boost/asio/buffer.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/bind/bind.hpp>
#include <boost/process/async_pipe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <boost/process/child.hpp>
#include <iostream>
#include <regex>
#include <string>
#include <variant>

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

enum ImageType {
  TEXT = 0,
  MATH = 1,
  IMAGE = 2,
  TABLE = 3
};

enum CallbackType {
  LATEX,
  PROCESS
};

class GhostscriptHandler {
public:
  GhostscriptHandler(std::string outputFileDirectory, const std::variant<std::function<std::string(cv::cuda::GpuMat &)>, std::function<void(cv::cuda::GpuMat &, const std::string &)>> &callback);
  void processOutput(const boost::system::error_code &ec, std::size_t size);

  void run(const std::string &inputFilePath);

  int done();

private:
  std::variant<std::function<std::string(cv::cuda::GpuMat &)>, std::function<void(cv::cuda::GpuMat &, const std::string &)>> callback;
  CallbackType callbackType;
  boost::asio::io_context ioContext;
  boost::process::async_pipe asyncPipe;
  boost::asio::streambuf buffer;

  boost::process::child process;

  std::string outputFileDirectory;
  std::string outputPrefix;
  std::string fileName;

  std::regex outputFormat;

  std::string latexEncoding;

  cv::cuda::GpuMat curImg;
};

class ImageUtils {
public:
  ImageUtils() = default;
  ~ImageUtils();

  static cv::cuda::GpuMat toMat(const torch::Tensor &tensor, bool isNormalized, bool cvFormat);
  static torch::Tensor toTensor(const cv::cuda::GpuMat &matrix, torch::ScalarType size, int channels = 1);
  static void addMargin(const cv::cuda::GpuMat &pixels, cv::Rect_<int> &rect, int margin);

  static void denoise(cv::cuda::GpuMat &pixels);
  static void crop(cv::cuda::GpuMat &pixels);
  static void threshold(cv::cuda::GpuMat &pixels);
  static void equalize(cv::cuda::GpuMat &pixels);

  static std::map<cv::Rect, ImageType, RectComparator> getImageBlocks(const cv::cuda::GpuMat &pixels);
  static float getSkewAngle(const cv::cuda::GpuMat &pixels, const ImageType &type);
  static void rotate(cv::cuda::GpuMat &pixels, float degree);

private:
  const static inline cv::Ptr<cv::cuda::Filter> thresholdBoxFilter = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(7, 7));
  const static inline cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1), cv::Point(-1, -1));
  const static inline cv::Ptr<cv::cuda::Filter> denoiseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, morphKernel);
  const static inline cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(85, 255);
  const static inline cv::Ptr<cv::cuda::HoughSegmentDetector> segmentDetector = cv::cuda::createHoughSegmentDetector(1, CV_PI / 180, 0, 20, 4096, 40);

  static inline Classifier imgClassification;

  static inline cv::cuda::GpuMat cropped;
  static inline cv::cuda::GpuMat mean;
  static inline cv::cuda::GpuMat edges;
  static inline cv::cuda::GpuMat lines;
  static inline cv::cuda::GpuMat rotated;
  static inline cv::cuda::GpuMat resized;
};

torch::Tensor toTensor(const std::string &strs);

std::vector<std::string> toString(const torch::Tensor &tensor);

void printHelp();

int clamp(int n, int lower, int upper);

bool fileExists(const char *filePath);

bool isDir(const char *path);

void winToNixFilePath(std::string &path);

void getPDFImages(const std::string &inputFilePath, const std::string &outputFileDirectory,
                  const std::variant<std::function<std::string(cv::cuda::GpuMat &)>,
                                     std::function<void(cv::cuda::GpuMat &, const std::string &)>> &callback);
#endif//MATHOCR_UTILS_H
