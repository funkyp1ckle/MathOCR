//
// Created by Pramit Govindaraj on 5/22/2023.
//

#include "OCREngine.h"
#include "utils.h"

DataSet::DataSet(const std::string &inputPath, bool training) : training(training) {
  std::string inputFilePath = training ? inputPath + "/train.txt" : inputPath + "/test.txt";
  std::ifstream inStream(inputFilePath);
  if (!inStream) {
    std::cerr << "Training file cannot be read from path(" << inputPath << ")" << std::endl;
    exit(READ_ERROR);
  }
  long long len = std::count(std::istreambuf_iterator<char>(inStream), std::istreambuf_iterator<char>(), '\n');
  images.reserve(len);
  labels.reserve(len);
  std::string label, imagePath;
  while (inStream >> label >> imagePath) {
    images.emplace_back(imagePath);
    labels.emplace_back(label);
  }
}

torch::data::Example<> DataSet::get(size_t idx) {
  auto index = static_cast<long long>(idx);
  std::string imagePath = images[index];
  std::string label = labels[index];

  torch::nn::MaxPool2d maxPool(torch::nn::MaxPoolOptions<2>({3, 3}).stride({1, 1}));//probably gonna break
  cv::cuda::GpuMat imgMat(cv::imread(imagePath, cv::IMREAD_GRAYSCALE));
  torch::Tensor imageTensor = maxPool(ImageUtils::toTensor(imgMat, torch::kByte));

  torch::Tensor labelTensor = torch::from_blob(label.data(), {1}, torch::kUInt8);
  return {labelTensor, imageTensor};
}

Encoder::Encoder(int d_model, int width, int height)
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
  d_model /= 2;
  torch::Tensor div_term = torch::exp(torch::arange(0, d_model, 2) * -(std::log(10000.0) / (double) d_model));
  torch::Tensor pos_w = torch::arange(0, width).unsqueeze(1);
  torch::Tensor pos_h = torch::arange(0., height).unsqueeze(1);
  positionalEncoding.slice(0, d_model, 2) = torch::sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
  positionalEncoding.slice(1, d_model, 2) = torch::cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
  positionalEncoding.index({d_model, nullptr, 2}) = torch::sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
  positionalEncoding.index({d_model + 1, nullptr, 2}) = torch::cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
}

torch::Tensor Encoder::forward(torch::Tensor input) {
  return cnn->forward(input).squeeze(2).permute({2, 0, 1});
}

Decoder::Decoder(int64_t inputSize, int64_t hiddenSize, int64_t numLayers, int64_t numClasses)
    : lstm(torch::nn::LSTMOptions(inputSize, hiddenSize).num_layers(numLayers).bidirectional(true)),
      fc(torch::nn::LinearOptions(hiddenSize * 2, numClasses)) { }

torch::Tensor Decoder::forward(torch::Tensor input) {
  torch::Tensor h0 = torch::zeros({lstm->options.num_layers() * 2, input.size(1), lstm->options.hidden_size()}).to(torch::kCUDA);
  torch::Tensor c0 = h0.clone();

  lstm->flatten_parameters();
  std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstmOut
      = lstm->forward(input, std::tuple<torch::Tensor, torch::Tensor>(h0, c0));
  torch::Tensor out = std::get<0>(lstmOut);
  torch::IntArrayRef sizes = out.sizes();
  out = fc->forward(out.view({sizes[0] * sizes[1], sizes[2]}));
  return out.view({sizes[0], sizes[1], -1});
}

OCREngine::OCREngine() : model(Encoder(512, 640, 640), Decoder(512, 512, 512, 512)) {}

OCREngine::OCREngine(const std::string &modelPath) {
  //std::shared_ptr<OCREngine> ptr(this);
  //torch::load(ptr, modelPath);
}

torch::Tensor OCREngine::forward(torch::Tensor input) {
  return model->forward(input);
}

void OCREngine::train(const std::string &dataDirectory, size_t epoch, float learningRate) {
  DataSet dataset(dataDirectory, true);
  namespace data = torch::data;
  auto data_loader = data::make_data_loader<data::samplers::SequentialSampler>(std::move(dataset).map(data::transforms::Stack<>()), 64);
  torch::optim::SGD optimizer(this->parameters(), learningRate);
  for (size_t i = 1; i <= epoch; ++i) {
    for (auto &batch: *data_loader) {
      optimizer.zero_grad();
      torch::Tensor prediction = forward(batch.data);
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      loss.backward();
      optimizer.step();
      std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << std::endl;
    }
  }
}

void OCREngine::test(const std::string &dataDirectory) {
  DataSet dataset(dataDirectory, false);
  torch::optional<size_t> totalDataCount = dataset.size();
  namespace data = torch::data;
  auto data_loader = data::make_data_loader<data::samplers::SequentialSampler>(std::move(dataset)
                                                                                   .map(data::transforms::Stack<>()),
                                                                               64);
  size_t counter = 0;
  for (auto &batch: *data_loader) {
    torch::Tensor prediction = forward(batch.data);
    if (prediction.equal(batch.target))
      ++counter;
  }
  std::cout << counter << "/" << totalDataCount << " correct" << std::endl;
}

void OCREngine::exportWeights(const std::string &outputPath) {
  torch::save(std::shared_ptr<OCREngine>(this), outputPath + "/weights.pt");
}

std::string OCREngine::toLatex(const cv::cuda::GpuMat &pixels) {
  torch::Tensor imageTensor = ImageUtils::toTensor(pixels, torch::kByte);
  torch::Tensor prediction = forward(imageTensor);
  std::ostringstream stream;
  stream << prediction;
  return stream.str();
}
