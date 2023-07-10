//
// Created by Pramit Govindaraj on 5/22/2023.
//

#include "OCREngine.h"
#include "utils.h"

#include <filesystem>
#include <random>

#include <torch/script.h>
#include <torch_tensorrt/logging.h>
#include <torch_tensorrt/torch_tensorrt.h>

DataSet::DataSet(const std::string &inputPath) {
  for (const auto &entry: std::filesystem::directory_iterator(inputPath)) {
    std::string path = entry.path().generic_string();
    files.emplace_back(path.substr(0, path.rfind('.')));
  }
}

torch::data::Example<> DataSet::get(size_t idx) {
  std::string inputDir = files[idx];

  std::ifstream texStream(inputDir + ".tex");
  if (!texStream) {
    std::cerr << "Tex file cannot be read from path(" << inputDir << ".tex"
              << ")" << std::endl;
    exit(READ_ERROR);
  }
  std::stringstream buffer;
  buffer << texStream.rdbuf();
  std::string tex = buffer.str();

  cv::cuda::GpuMat imgMat(cv::imread(inputDir + ".png", cv::IMREAD_GRAYSCALE));
  torch::Tensor imageTensor = ImageUtils::toTensor(imgMat, torch::kByte);

  torch::Tensor labelTensor = OCRUtils::toTensor(tex);
  return {imageTensor, labelTensor};
}

Classifier::Classifier() {
  try {
    torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::kERROR);
    classificationModule = torch::jit::load("classify.torchscript", torch::kCUDA);
    //classificationModule = torch::jit::load("../models/best.torchscript", torch::kCUDA);
    classificationModule.to(torch::kFloat);
    classificationModule.eval();

    //DEBUG GENERATE TORCH-TENSORRT
    /*std::vector<int64_t> dims = {1, 3, 640, 640};
    auto input = torch_tensorrt::Input(dims, torch::kFloat);
    auto compile_settings = torch_tensorrt::ts::CompileSpec({input});
    compile_settings.enabled_precisions = {torch::kFloat};
    compile_settings.truncate_long_and_double = true;
    classificationModule = torch_tensorrt::ts::compile(classificationModule, compile_settings);
    classificationModule.save("../models/classify.torchscript");*/
  } catch (const torch::Error &e) {
    std::cerr << "Error loading classification model" << std::endl;
    exit(READ_ERROR);
  }
}

torch::Tensor Classifier::forward(const torch::Tensor &input) {
  torch::Tensor pT = classificationModule.forward({input}).toTensor();
  torch::Tensor score = std::get<0>(pT.slice(1, 4, -1).max(1, true));
  return torch::cat({pT.slice(1, 0, 4), score, pT.slice(1, 4, -1)}, 1).permute({0, 2, 1});
}

EncoderImpl::EncoderImpl()
    : conv1(torch::nn::Conv2dOptions(1, 64, {3, 3}).stride({1, 1}).padding({1, 1})),
      maxPool1(torch::nn::MaxPool2dOptions({2, 2}).stride({1, 1}).padding({1, 1})),
      conv2(torch::nn::Conv2dOptions(64, 128, {3, 3}).stride({1, 1}).padding({1, 1})),
      maxPool2(torch::nn::MaxPool2dOptions({2, 2}).stride({1, 1}).padding({1, 1})),
      conv3(torch::nn::Conv2dOptions(128, 256, {3, 3}).stride({1, 1}).padding({1, 1})),
      conv4(torch::nn::Conv2dOptions(256, 256, {3, 3}).stride({1, 1}).padding({1, 1})),
      maxPool3(torch::nn::MaxPool2dOptions({2, 1}).stride({2, 1}).padding({1, 0})),
      conv5(torch::nn::Conv2dOptions(256, 512, {3, 3}).stride({1, 1}).padding({1, 1})),
      maxPool4(torch::nn::MaxPool2dOptions({1, 2}).stride({1, 2}).padding({0, 1})),
      conv6(torch::nn::Conv2dOptions(512, 512, {3, 3})) {
  register_module("conv1", conv1);
  register_module("maxPool1", maxPool1);
  register_module("conv2", conv2);
  register_module("maxPool2", maxPool2);
  register_module("conv3", conv3);
  register_module("conv4", conv4);
  register_module("maxPool3", maxPool3);
  register_module("conv5", conv5);
  register_module("maxPool4", maxPool4);
  register_module("conv6", conv6);
}

torch::Tensor EncoderImpl::forward(torch::Tensor input) {
  std::cout << input.sizes() << std::endl;//TODO: REMOVE AFTER DEBUGGING
  input = conv1(input);
  input = maxPool1(input);
  input = torch::relu(input);

  input = conv2(input);
  input = maxPool2(input);
  input = torch::relu(input);

  input = conv3(input);
  input = torch::relu(input);

  input = conv4(input);
  input = maxPool3(input);
  input = torch::relu(input);

  input = conv5(input);
  input = maxPool4(input);
  input = torch::relu(input);

  input = conv6(input);
  input = torch::relu(input);

  input = input.permute({0, 2, 3, 1});
  input = positionalEncoding(input);
  input = input.permute({0, 3, 1, 2});

  input = input.contiguous();
  return input;
}

torch::Tensor EncoderImpl::positionalEncoding(torch::Tensor input) {
  constexpr float minTimeScale = 1.0f;
  constexpr float maxTimeScale = 10000.0f;

  torch::ArrayRef<int64_t> sizes = input.sizes();
  int64_t numDims = static_cast<int64_t>(sizes.size()) - 2;
  int64_t channels = sizes[1];
  int64_t numTimeScales = channels / (numDims * 2);
  float logTimeScaleIncrement = log((float) maxTimeScale / minTimeScale) / (float) (numTimeScales - 1);
  torch::Tensor invTimeScale = minTimeScale * torch::exp(torch::arange(0, numTimeScales).to(torch::kFloat32).multiply(-logTimeScaleIncrement));

  for (int64_t i = 0; i < numDims; ++i) {
    int64_t len = sizes[i + 1];
    torch::Tensor position = torch::arange(len).to(torch::kFloat32);
    torch::Tensor scaledTime = torch::reshape(position, {-1, 1}).multiply(torch::reshape(invTimeScale, {1, -1}));
    torch::Tensor signal = torch::cat({torch::sin(scaledTime), torch::cos(scaledTime)}, 1).to(torch::kCUDA);
    int64_t prepad = i * 2 * numTimeScales;
    int64_t postpad = channels - (i + 1) * 2 * numTimeScales;
    signal = torch::nn::functional::pad(signal, torch::nn::functional::PadFuncOptions({prepad, postpad, 0, 0}));
    for (int64_t j = 0; j < i + 1; ++j)
      signal = signal.unsqueeze(0);
    for (int64_t j = 0; j < numDims - 1 - i; ++j)
      signal = signal.unsqueeze(-2);
    input += signal;
  }
  return input;
}

AttentionImpl::AttentionImpl(int64_t encoderDim, int64_t decoderDim, int64_t attentionDim)
    : encoderAttention(torch::nn::LinearOptions(encoderDim, attentionDim)),
      decoderAttention(torch::nn::LinearOptions(decoderDim, attentionDim)),
      fullAttention(torch::nn::LinearOptions(attentionDim, 1)),
      relu(),
      softmax(torch::nn::SoftmaxOptions(1)) {
  register_module("encoderAttention", encoderAttention);
  register_module("decoderAttention", decoderAttention);
  register_module("fullAttention", fullAttention);
  register_module("relu", relu);
  register_module("softmax", softmax);
}

std::pair<torch::Tensor, torch::Tensor> AttentionImpl::forward(torch::Tensor encoderOut, torch::Tensor decoderHidden) {
  torch::Tensor att1 = encoderAttention->forward(encoderOut);
  torch::Tensor att2 = decoderAttention->forward(decoderHidden);
  torch::Tensor att = fullAttention->forward(relu->forward(att1 + att2.unsqueeze(1))).unsqueeze(2);
  torch::Tensor alpha = softmax->forward(att);
  torch::Tensor attentionWeightedEncoding = (encoderOut * alpha.unsqueeze(2)).sum(1);
  return std::make_pair(attentionWeightedEncoding, alpha);
}

DecoderImpl::DecoderImpl(int64_t attentionDim, int64_t embedDim, int64_t decoderDim, int64_t vocabSize, int64_t encoderDim = 512, float dropout = 0.5f, double p = 0)
    : encoderDim(encoderDim),
      attentionDim(attentionDim),
      decoderDim(decoderDim),
      vocabSize(vocabSize),
      p(p),
      attention(encoderDim, decoderDim, attentionDim),
      embedding(torch::nn::EmbeddingOptions(vocabSize, embedDim)),
      dropout(dropout),
      decodeStep(torch::nn::GRUCellOptions(embedDim + encoderDim, decoderDim).bias(true)),
      initH(torch::nn::LinearOptions(encoderDim, decoderDim)),
      initC(torch::nn::LinearOptions(encoderDim, decoderDim)),
      fBeta(torch::nn::LinearOptions(decoderDim, encoderDim)),
      fc(torch::nn::LinearOptions(decoderDim, vocabSize)) {
  register_module("attention", attention);
  register_module("embebdding", embedding);
  register_module("decodeStep", decodeStep);
  register_module("initH", initH);
  register_module("initC", initC);
  register_module("fBeta", fBeta);
  register_module("sigmoid", sigmoid);
  register_module("fc", fc);

  attention->to(torch::kCUDA);

  initWeights();
}

void DecoderImpl::initWeights() {
  embedding->weight.data().uniform_(-0.1, 0.1);
  fc->bias.data().fill_(0);
  fc->weight.data().uniform_(-0.1, 0.1);
}

torch::Tensor DecoderImpl::initHiddenState(torch::Tensor encoderOut) {
  torch::Tensor meanEncoderOut = encoderOut.mean(1);
  return initH->forward(meanEncoderOut);
}

std::vector<torch::Tensor> DecoderImpl::forward(torch::Tensor encoderOut, torch::Tensor encodedCaptions, torch::Tensor captionLens, double p = 1) {
  this->p = p;
  int64_t batchSize = encoderOut.size(0);
  int64_t encoderSize = encoderOut.size(1);

  encoderOut = encoderOut.view({batchSize, -1, encoderSize});
  int64_t numPixels = encoderOut.size(1);

  std::tuple<torch::Tensor, torch::Tensor> sortedCaptions = captionLens.sort(0, true);
  captionLens = std::get<0>(sortedCaptions);
  torch::Tensor sortIdx = std::get<1>(sortedCaptions);

  encoderOut = encoderOut[sortIdx];
  encodedCaptions = encodedCaptions[sortIdx];

  torch::Tensor embeddings = embedding->forward(encodedCaptions);
  torch::Tensor h = initHiddenState(encodedCaptions);

  torch::Tensor decodeLens = (captionLens - 1);
  auto maxDecodeLen = max(decodeLens).item<int64_t>();

  torch::Tensor predictions = torch::zeros({batchSize, maxDecodeLen, vocabSize}).to(torch::kCUDA);
  torch::Tensor alphas = torch::zeros({batchSize, maxDecodeLen, numPixels}).to(torch::kCUDA);

  static std::random_device randomDevice;
  static std::mt19937 mt(randomDevice());
  static std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int64_t t = 0; t < maxDecodeLen; ++t) {
    auto batchSizeType = torch::sum(decodeLens > t).item<int64_t>();
    std::pair<torch::Tensor, torch::Tensor> attentionWeighted = attention->forward(encoderOut.index({torch::indexing::Slice(torch::indexing::None, batchSizeType)}), h.index({torch::indexing::Slice(torch::indexing::None, batchSizeType)}));
    torch::Tensor gate = sigmoid->forward(fBeta->forward(h.index({torch::indexing::Slice(torch::indexing::None, batchSizeType)})));
    attentionWeighted.first *= gate;
    if(t == 1 || dist(mt) < p)
      h = decodeStep->forward(torch::cat({embeddings.index({torch::indexing::Slice(torch::indexing::None, batchSizeType), torch::indexing::Slice(t)}), captionLens}, 1), h.index({torch::indexing::Slice(torch::indexing::None, batchSizeType)}));
    else
      h = decodeStep->forward(torch::cat({embedding->forward(torch::argmax(predictions.index({torch::indexing::Slice(torch::indexing::None, batchSizeType), torch::indexing::Slice(t)}), 1)), attentionWeighted.first}, 1), h.index({torch::indexing::Slice(torch::indexing::None, batchSizeType)}));
    torch::Tensor preds = fc->forward(dropout->forward(h));
    predictions.index({torch::indexing::Slice(torch::indexing::None, batchSizeType), torch::indexing::Slice(t)}) = preds;
    alphas.index({torch::indexing::Slice(torch::indexing::None, batchSizeType), torch::indexing::Slice(t)}) = alphas;
  }
  return {predictions, encodedCaptions, decodeLens, alphas, sortIdx};
}

OCREngineImpl::OCREngineImpl()
    : encoder(),
      decoder(512, 32, 512, 483, 0.5) {
  register_module("OCREncoder", encoder);
  register_module("OCRDecoder", decoder);
}

OCREngineImpl::OCREngineImpl(const std::string &modelPath) {
  std::shared_ptr<OCREngineImpl> ptr(this);
  torch::load(ptr, modelPath);
}

torch::Tensor OCREngineImpl::forward(torch::Tensor input) {
  return model->forward(input);
}

void OCREngineImpl::train(const std::string &dataDirectory, size_t epoch, float learningRate) {
  model->train();
  DataSet dataset(dataDirectory);
  namespace data = torch::data;
  auto dataLoader = data::make_data_loader<data::samplers::SequentialSampler>(std::move(dataset).map(data::transforms::Stack<>()), 64);

  auto start = std::chrono::high_resolution_clock::now();
  torch::optim::Adam optimizer(this->parameters(), learningRate);
  torch::nn::CTCLoss criterion;
  criterion->to(torch::kCUDA);
  for (size_t i = 1; i <= epoch; ++i) {
    for (auto &batch: *dataLoader) {
      torch::Tensor data = batch.data.to(torch::kCUDA);
      data = data.squeeze(1);
      optimizer.zero_grad();
      torch::Tensor prediction = forward(data);
      torch::Tensor logProbs = torch::nn::functional::log_softmax(prediction, torch::nn::functional::LogSoftmaxFuncOptions(2));
      torch::Tensor inputLengths = torch::full({prediction.size(0)}, prediction.size(0), torch::TensorOptions(torch::kLong)).to(torch::kCUDA);
      torch::Tensor loss = criterion->forward(logProbs, batch.target, inputLengths, torch::tensor(batch.target.size(0)));
      loss.backward();
      optimizer.step();
      std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << " | Duration: " << (std::chrono::high_resolution_clock::now() - start).count() << std::endl;
    }
  }
}

void OCREngineImpl::test(const std::string &dataDirectory) {
  DataSet dataset(dataDirectory);
  namespace data = torch::data;
  auto data_loader = data::make_data_loader<data::samplers::SequentialSampler>(std::move(dataset)
                                                                                   .map(data::transforms::Stack<>()),
                                                                               64);
  size_t total = 0;
  size_t counter = 0;
  torch::NoGradGuard no_grad;
  for (auto &batch: *data_loader) {
    batch.data.to(torch::kCUDA);
    torch::Tensor output = forward(batch.data);
    torch::Tensor prediction = std::get<1>(output.max(2)).transpose(1, 0).contiguous().view(-1).to(torch::kCPU);
    torch::Tensor inputLens = torch::full(batch.data.size(0), output.size(0), torch::TensorOptions(torch::kLong));
    std::vector<std::string> simPreds = OCRUtils::toString(prediction);
    std::vector<std::string> targets = OCRUtils::toString(batch.target);
    size_t outputLens = simPreds.size();
    for (int i = 0; i < outputLens; ++i) {
      total += targets[i].size();
      size_t len = std::min(simPreds[i].size(), targets[i].size());
      for (size_t j = 0; j < len; ++j)
        if (simPreds[i][j] == targets[i][j])
          ++counter;
    }
  }
  std::cout << counter << "/" << total << " characters correct" << std::endl;
}

void OCREngineImpl::exportWeights(const std::string &outputPath) {
  std::shared_ptr<OCREngineImpl> ptr(this);
  torch::save(ptr, outputPath + "/weights.pt");
}

std::string OCREngineImpl::toLatex(const cv::cuda::GpuMat &pixels) {
  torch::Tensor imageTensor = ImageUtils::toTensor(pixels, torch::kByte).transpose(1, 3).to(torch::kFloat32).sub(128).div(128);
  torch::Tensor prediction = forward(imageTensor);
  return OCRUtils::toString(prediction)[0];
}
