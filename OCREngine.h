//
// Created by Pramit Govindaraj on 5/22/2023.
//

#ifndef MATHOCR_OCRENGINE_H
#define MATHOCR_OCRENGINE_H

#include <opencv2/core/mat.hpp>
#include <string>

#include <torch/torch.h>

extern const int INVALID_PARAMETER;
extern const int READ_ERROR;
extern const int PROCESSING_ERROR;

enum LatexTokens {

};

enum ImageType {
  TEXT = 0,
  MATH = 1,
  IMAGE = 2,
  TABLE = 3
};

class DataSet : public torch::data::datasets::Dataset<DataSet> {
public:
  explicit DataSet(const std::string &inputPath, bool training);
  torch::data::Example<> get(size_t idx) override;//{ return {.first, data[(long long) idx]}; }
  torch::data::Example<> operator[](size_t idx) { return get(idx); }
  torch::optional<size_t> size() const override { return images.size(); }
  bool isTraining() const { return training; }

private:
  std::vector<std::string> images;
  std::vector<std::string> labels;
  bool training;
};

class Classifier {
public:
  Classifier();
  torch::Tensor forward(const torch::Tensor &input);
private:
  torch::jit::script::Module classificationModule;
};

//VGG16 + 2D Positional Encoding
class Encoding : public torch::nn::Module {
public:
  Encoding() = default;
  Encoding(int64_t d_model, int64_t width, int64_t height);
  torch::Tensor forward(torch::Tensor input);

private:
  torch::nn::Conv2d conv1_1{nullptr};
  torch::nn::Conv2d conv1_2{nullptr};
  torch::nn::Conv2d conv2_1{nullptr};
  torch::nn::Conv2d conv2_2{nullptr};
  torch::nn::Conv2d conv3_1{nullptr};
  torch::nn::Conv2d conv3_2{nullptr};
  torch::nn::Conv2d conv3_3{nullptr};
  torch::nn::Conv2d conv4_1{nullptr};
  torch::nn::Conv2d conv4_2{nullptr};
  torch::nn::Conv2d conv4_3{nullptr};
  torch::nn::Conv2d conv5_1{nullptr};
  torch::nn::Conv2d conv5_2{nullptr};
  torch::nn::Conv2d conv5_3{nullptr};
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear fc3{nullptr};

  torch::Tensor positionalEncoding;
};

class OCREngine : public torch::nn::Module {
public:
  OCREngine();
  OCREngine(const OCREngine &ocr) {}
  explicit OCREngine(const std::string &modelPath);
  torch::Tensor forward(torch::Tensor input);

  void train(const std::string &dataDirectory, size_t epoch, float learningRate);
  void test(const std::string &dataDirectory);
  void exportWeights(const std::string &outputPath);

  std::string toLatex(const cv::cuda::GpuMat &pixels, const ImageType &type);

private:
  Encoding encoder;
  //torch::nn::LSTM attention;
  //torch::nn::LSTM decoder;
};

#endif//MATHOCR_OCRENGINE_H
