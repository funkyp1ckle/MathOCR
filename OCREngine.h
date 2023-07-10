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
  torch::Tensor positionalEncoding(torch::Tensor input);

private:
  torch::nn::Conv2d conv1;
  torch::nn::MaxPool2d maxPool1;

  torch::nn::Conv2d conv2;
  torch::nn::MaxPool2d maxPool2;

  torch::nn::Conv2d conv3;

  torch::nn::Conv2d conv4;
  torch::nn::MaxPool2d maxPool3;

  torch::nn::Conv2d conv5;
  torch::nn::MaxPool2d maxPool4;

  torch::nn::Conv2d conv6;
};
TORCH_MODULE(Encoder);

class AttentionImpl : public torch::nn::Module {
public:
  AttentionImpl(int64_t encoderDim, int64_t decoderDim, int64_t attentionDim);
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor encoderOut, torch::Tensor decoderHidden);

private:
  torch::nn::Linear encoderAttention;
  torch::nn::Linear decoderAttention;
  torch::nn::Linear fullAttention;
  torch::nn::ReLU relu;
  torch::nn::Softmax softmax;
};
TORCH_MODULE(Attention);

class DecoderImpl : public torch::nn::Module {
public:
  DecoderImpl(int64_t attentionDim, int64_t embedDim, int64_t decoderDim, int64_t vocabSize, int64_t encoderDim, float dropout, double p);
  std::vector<torch::Tensor> forward(torch::Tensor encoderOut, torch::Tensor encodedCaptions, torch::Tensor captionLens, double p);

private:
  void initWeights();
  torch::Tensor initHiddenState(torch::Tensor encoderOut);

  Attention attention;

  torch::nn::Embedding embedding;
  torch::nn::Dropout dropout;

  torch::nn::GRUCell decodeStep;
  torch::nn::Linear initH;
  torch::nn::Linear initC;
  torch::nn::Linear fBeta;
  torch::nn::Sigmoid sigmoid;
  torch::nn::Linear fc;

  int64_t encoderDim;
  int64_t attentionDim;
  int64_t decoderDim;
  int64_t vocabSize;
  double p;
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
  Encoder encoder;
  Decoder decoder;
};
TORCH_MODULE(OCREngine);

#endif//MATHOCR_OCRENGINE_H
