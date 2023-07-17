//
// Created by Pramit Govindaraj on 5/22/2023.
//

#ifndef MATHOCR_OCRENGINE_H
#define MATHOCR_OCRENGINE_H

#include <filesystem>
#include <string>

#include "tesseract/baseapi.h"
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
  enum class OCRMode {
    TRAIN,
    VAL
  };
  explicit DataSet(std::filesystem::path inputPath, OCRMode mode);
  torch::data::Example<> get(size_t idx) override;//{ return {.first, data[(long long) idx]}; }
  torch::data::Example<> operator[](size_t idx) { return get(idx); }
  torch::optional<size_t> size() const override { return files.size(); }

private:
  OCRMode mode;
  std::vector<std::string> files;
};

//VGG16 + 2D Positional Encoding
class EncoderImpl : public torch::nn::Module {
public:
  EncoderImpl();
  torch::Tensor forward(torch::Tensor input);
  torch::Tensor positionalEncoding(torch::Tensor input);

private:
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::MaxPool2d maxPool1{nullptr};

  torch::nn::Conv2d conv2{nullptr};
  torch::nn::MaxPool2d maxPool2{nullptr};

  torch::nn::Conv2d conv3{nullptr};

  torch::nn::Conv2d conv4{nullptr};
  torch::nn::MaxPool2d maxPool3{nullptr};

  torch::nn::Conv2d conv5{nullptr};
  torch::nn::MaxPool2d maxPool4{nullptr};

  torch::nn::Conv2d conv6{nullptr};
};
TORCH_MODULE(Encoder);

class AttentionImpl : public torch::nn::Module {
public:
  AttentionImpl(int64_t encoderDim, int64_t decoderDim, int64_t attentionDim);
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor encoderOut, torch::Tensor decoderHidden);

private:
  torch::nn::Linear encoderAttention{nullptr};
  torch::nn::Linear decoderAttention{nullptr};
  torch::nn::Linear fullAttention{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::Softmax softmax{nullptr};
};
TORCH_MODULE(Attention);

class DecoderImpl : public torch::nn::Module {
public:
  DecoderImpl(int64_t attentionDim, int64_t embedDim, int64_t decoderDim, int64_t vocabSize, int64_t encoderDim, float dropout, double p);
  std::vector<torch::Tensor> forward(torch::Tensor encoderOut, torch::Tensor encodedCaptions, torch::Tensor captionLens, double p);

  friend class LatexOCREngineImpl;

private:
  void initWeights();
  torch::Tensor initHiddenState(torch::Tensor encoderOut);

  Attention attention{nullptr};

  torch::nn::Embedding embedding{nullptr};
  torch::nn::Dropout dropout{nullptr};

  torch::nn::GRUCell decodeStep{nullptr};
  torch::nn::Linear initH{nullptr};
  torch::nn::Linear initC{nullptr};
  torch::nn::Linear fBeta{nullptr};
  torch::nn::Sigmoid sigmoid{nullptr};
  torch::nn::Linear fc{nullptr};

  int64_t encoderDim;
  int64_t attentionDim;
  int64_t decoderDim;
  int64_t vocabSize;
  double p;
};
TORCH_MODULE(Decoder);

class LatexOCREngineImpl : public torch::nn::Module {
public:
  explicit LatexOCREngineImpl();
  explicit LatexOCREngineImpl(const std::string& modelPath);

  torch::Tensor forward(torch::Tensor input, int64_t beamSize);

  void train(DataSet &dataset, size_t epoch, float learningRate);
  void test(const std::filesystem::path &dataDirectory);
  void exportWeights(const std::filesystem::path &outputPath);

private:
  std::unordered_map<std::string, int> vocabMap;

  Encoder encoder{nullptr};
  Decoder decoder{nullptr};
};
TORCH_MODULE(LatexOCREngine);

class TesseractOCREngine {
public:
  TesseractOCREngine();
  ~TesseractOCREngine();

  std::string doOCR(const cv::cuda::GpuMat &pixels);

private:
  static inline tesseract::TessBaseAPI *api;
};

class OCREngine {
public:
  static std::string toLatex(const cv::cuda::GpuMat &pixels);
  static std::string toText(const cv::cuda::GpuMat &pixels);
  static std::string toTable(const std::vector<std::string> &items);
  static std::string toImage(const cv::cuda::GpuMat &pixels);
};

#endif//MATHOCR_OCRENGINE_H
