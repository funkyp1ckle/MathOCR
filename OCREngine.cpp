//
// Created by Pramit Govindaraj on 5/22/2023.
//

#include "OCREngine.h"
#include "utils.h"
#include <filesystem>
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

  torch::Tensor labelTensor = toTensor(tex);
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
    : cnn(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).padding({1, 1})),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding({1, 1})),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding({1, 1})),
          torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding({1, 1})),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({1, 2})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding({1, 1})),
          torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding({1, 1})),
          torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 1})),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding({1, 1})),
          torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))) {
  register_module("cnn", cnn);

  constexpr int width = 640;
  constexpr int height = 640;
  positionalEncoding = torch::zeros({512, width, height});
  torch::Tensor divTerm = torch::exp(torch::arange(0, 256, 2) * -log(10000 / 256));
  torch::Tensor posW = torch::arange(0, width).unsqueeze(1);
  torch::Tensor posH = torch::arange(0, height).unsqueeze(1);
  positionalEncoding.slice(0, 256, 2) = torch::sin(posW * divTerm).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
  positionalEncoding.slice(1, 256, 2) = torch::cos(posW * divTerm).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
  positionalEncoding.slice(256, torch::indexing::None, 2) = torch::sin(posH * divTerm).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
  positionalEncoding.slice(257, torch::indexing::None, 2) = torch::sin(posH * divTerm).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
}

torch::Tensor EncoderImpl::forward(torch::Tensor input) {
  std::cout << input.sizes() << std::endl; //TODO: REMOVE AFTER DEBUGGING
  return cnn->forward(input).squeeze(2).permute({2, 0, 1});
}

DecoderImpl::DecoderImpl(int64_t inputSize, int64_t hiddenSize, int64_t numLayers, int64_t numClasses)
    : lstm(torch::nn::LSTMOptions(inputSize, hiddenSize)
               .num_layers(numLayers)
               .bidirectional(true)),
      fc(torch::nn::LinearOptions(hiddenSize * 2, numClasses)) {
  register_module("lstm", lstm);
  register_module("fc", fc);
}

torch::Tensor DecoderImpl::forward(torch::Tensor input) {
  torch::Tensor h0 = torch::zeros({lstm->options.num_layers() * 2, input.size(1), lstm->options.hidden_size()}).to(torch::kCUDA);
  torch::Tensor c0 = h0.clone();

  lstm->flatten_parameters();
  std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstmOut = lstm->forward(input, std::tuple<torch::Tensor, torch::Tensor>(h0, c0));
  torch::Tensor out = std::get<0>(lstmOut);
  torch::IntArrayRef sizes = out.sizes();
  out = fc->forward(out.view({sizes[0] * sizes[1], sizes[2]}));
  return out.view({sizes[0], sizes[1], -1});
}

OCREngineImpl::OCREngineImpl() : model(Encoder(), Decoder(512, 256, 2, 64)) {
  register_module("OCRModel", model);
}

OCREngineImpl::OCREngineImpl(const std::string &modelPath) {
  //std::shared_ptr<OCREngineImpl> ptr(this);
  //torch::load(ptr, modelPath);
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
    std::vector<std::string> simPreds = toString(prediction);
    std::vector<std::string> targets = toString(batch.target);
    size_t outputLens = simPreds.size();
    for(int i = 0; i < outputLens; ++i) {
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
  torch::Tensor imageTensor = ImageUtils::toTensor(pixels, torch::kByte);
  torch::Tensor prediction = forward(imageTensor);
  return toString(prediction)[0];
}
