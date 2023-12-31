//
// Created by Pramit Govindaraj on 5/19/2023.
//

#ifndef MATHOCR_UTILS_H
#define MATHOCR_UTILS_H

#include "OCREngine.h"

#include <wkhtmltox/image.h>

#include <v8.h>
#include <libplatform/libplatform.h>
#include <node.h>

#include <boost/asio/buffer.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/bind/bind.hpp>
#include <boost/process/async_pipe.hpp>
#include <boost/process/child.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <variant>

extern const int ALLOC_ERROR;
extern const int ENVIRONMENT_ERROR;

struct Options {
  bool deskew = false;
};

extern Options settings;

class KatexHandler {
public:
  KatexHandler();

  ~KatexHandler();

  std::string normalize(const std::string &text);

  std::string latexToHTML(std::string latex);

  static void escape(std::string &code);

  static void lineCleanup(std::string &line);

  static void replaceUnsupported(std::string &line);

private:
  v8::Local<v8::Value> run(const std::string &source, const v8::Local<v8::Context> &context) const;

  std::unique_ptr<node::MultiIsolatePlatform> platform;
  std::unique_ptr<node::CommonEnvironmentSetup> setup;

  v8::Isolate *isolate;
  v8::Global<v8::Context> context;
};

class HTMLRenderHandler {
public:
  HTMLRenderHandler();
  ~HTMLRenderHandler();

  cv::Mat renderHTML(const std::string &html);
private:
  wkhtmltoimage_global_settings *settings;
  wkhtmltoimage_converter *converter;
  cv::Size imgSize;
};

class OCRUtils {
public:
  static std::vector<cv::cuda::GpuMat> toMat(const torch::Tensor &tensor, bool isNormalized = false);

  static void normalizeLatex(const std::filesystem::path &inFile, const std::filesystem::path &outFile);

  static void renderLatex(const std::string &latex);

  static std::unordered_map<std::string, int> getVocab(const std::filesystem::path &dataDirectory);

  static torch::Tensor toTensor(const std::string &str);

  static std::vector<std::string> toString(const torch::Tensor &tensor);

private:
  static inline KatexHandler katex;
  static inline HTMLRenderHandler wkhtml;
};

class GhostscriptHandler {
public:
  enum class CallbackType {
    LATEX,
    PROCESS
  };

  GhostscriptHandler(std::filesystem::path outputFileDirectory,
                     const std::variant<std::function<void(cv::cuda::GpuMat &, const std::filesystem::path &)>,
                       std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(
                         cv::cuda::GpuMat &)>> &callback);

  void processOutput(const boost::system::error_code &ec, std::size_t size);

  void processOutput();

  void run(const std::filesystem::path &inputFilePath);

  int done();

private:
  std::variant<std::function<void(cv::cuda::GpuMat &,
                                  const std::filesystem::path &)>, std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(
    cv::cuda::GpuMat &)>> callback;
  CallbackType callbackType;
  boost::asio::io_context ioContext;
  boost::process::async_pipe asyncPipe;
  boost::asio::streambuf buffer;

  boost::process::child process;

  std::filesystem::path outputFileDirectory;
  std::filesystem::path outputPrefix;
  std::filesystem::path fileName;

  int pageNum;

  std::regex outputFormat;

  cv::cuda::GpuMat curImg;
};

class ImageUtils {
public:
  static torch::Tensor toTensor(const cv::cuda::GpuMat &matrix, torch::ScalarType size, int channels = 1);

  static void addMargin(const cv::cuda::GpuMat &pixels, cv::Rect_<int> &rect, int margin);

  static void equalize(cv::cuda::GpuMat &pixels);

  static void denoise(cv::cuda::GpuMat &pixels);

  static void crop(cv::cuda::GpuMat &pixels);

  static void threshold(cv::cuda::GpuMat &pixels);

  static std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>
  getImageBlocks(const cv::cuda::GpuMat &pixels);

  static float getSkewAngle(const cv::cuda::GpuMat &pixels, const Classifier::ImageType &type);

  static void rotate(cv::cuda::GpuMat &pixels, float degree);
};

int clamp(int n, int lower, int upper);

void replaceAll(std::string &str, const std::string &match, const std::string &replacement);

void getPDFImages(const std::filesystem::path &inputFilePath, const std::filesystem::path &outputFileDirectory,
                  const std::variant<std::function<void(cv::cuda::GpuMat &, const std::filesystem::path &)>,
                    std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(
                      cv::cuda::GpuMat &)>> &callback);

#endif//MATHOCR_UTILS_H
