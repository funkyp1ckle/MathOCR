//
// Created by Pramit Govindaraj on 5/22/2023.
//

#ifndef MATHOCR_OCRENGINE_H
#define MATHOCR_OCRENGINE_H

#include <string>

#include <opencv2/core/cuda.hpp>
#include <torch/torch.h>

extern const int INVALID_PARAMETER;
extern const int READ_ERROR;
extern const int PROCESSING_ERROR;

class Classifier {
public:
  Classifier();
  torch::Tensor forward(const torch::Tensor &input);

private:
  torch::jit::script::Module classificationModule;
};

class DataSet : public torch::data::datasets::Dataset<DataSet> {
public:
  explicit DataSet(const std::string &inputPath);
  torch::data::Example<> get(size_t idx) override;//{ return {.first, data[(long long) idx]}; }
  torch::data::Example<> operator[](size_t idx) { return get(idx); }
  torch::optional<size_t> size() const override { return files.size(); }
private:
  std::vector<std::string> files;
};

//VGG16 + 2D Positional Encoding
class EncoderImpl : public torch::nn::Module {
public:
  EncoderImpl();
  torch::Tensor forward(torch::Tensor input);

private:
  torch::nn::Sequential cnn;
  torch::Tensor positionalEncoding;
};
TORCH_MODULE(Encoder);

class DecoderImpl : public torch::nn::Module {
public:
  DecoderImpl(int64_t inputSize, int64_t hiddenSize, int64_t numLayers, int64_t numClasses);
  torch::Tensor forward(torch::Tensor input);
private:
  torch::nn::LSTM lstm{nullptr};
  torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(Decoder);

class OCREngineImpl : public torch::nn::Module {
public:
  OCREngineImpl();
  explicit OCREngineImpl(const std::string &modelPath);
  torch::Tensor forward(torch::Tensor input);

  void train(const std::string &dataDirectory, size_t epoch, float learningRate);
  void test(const std::string &dataDirectory);
  void exportWeights(const std::string &outputPath);

  std::string toLatex(const cv::cuda::GpuMat &pixels);

private:
  torch::nn::Sequential model;
};
TORCH_MODULE(OCREngine);

#endif//MATHOCR_OCRENGINE_H
