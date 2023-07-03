//
// Created by Pramit Govindaraj on 5/22/2023.
//

#include "OCREngine.h"
#include "utils.h"
#include <opencv2/imgcodecs.hpp>
#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <torch_tensorrt/logging.h>


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

  torch::nn::MaxPool2d maxPool(torch::nn::MaxPoolOptions<2>({3, 3}).stride({1, 1})); //probably gonna break
  cv::cuda::GpuMat imgMat(cv::imread(imagePath, cv::IMREAD_GRAYSCALE));
  torch::Tensor imageTensor = maxPool(ImageUtils::toTensor(imgMat, torch::kByte));

  torch::Tensor labelTensor = torch::from_blob(label.data(), {1}, torch::kUInt8);
  return {labelTensor, imageTensor};
}

Classifier::Classifier() {
  try {
    classificationModule = torch::jit::load("classify.torchscript", torch::kCUDA);
    //classificationModule = torch::jit::load("../models/best.torchscript", torch::kCUDA);
    classificationModule.to(torch::kFloat);
    classificationModule.eval();

    //TODO: DEBUG GENERATE TORCH-TENSORRT
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

Encoding::Encoding(int64_t d_model, int64_t width, int64_t height) : conv1_1(register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 3).padding(1)))),
                                                                     conv1_2(register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).padding(1)))),
                                                                     conv2_1(register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 30, 3).padding(1)))),
                                                                     conv2_2(register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(30, 40, 3).padding(1)))),
                                                                     conv3_1(register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(40, 50, 3).padding(1)))),
                                                                     conv3_2(register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(50, 60, 3).padding(1)))),
                                                                     conv3_3(register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(60, 70, 3).padding(1)))),
                                                                     conv4_1(register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(70, 80, 3).padding(1)))),
                                                                     conv4_2(register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(80, 90, 3).padding(1)))),
                                                                     conv4_3(register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(90, 100, 3).padding(1)))),
                                                                     conv5_1(register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(100, 110, 3).padding(1)))),
                                                                     conv5_2(register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(110, 120, 3).padding(1)))),
                                                                     conv5_3(register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(120, 130, 3).padding(1)))),
                                                                     fc1(register_module("fc1", torch::nn::Linear(130, 50))),
                                                                     fc2(register_module("fc2", torch::nn::Linear(50, 20))),
                                                                     fc3(register_module("fc3", torch::nn::Linear(20, 10))) {
  d_model /= 2;
  torch::Tensor div_term = torch::exp(torch::arange(0, d_model, 2) * -(std::log(10000.0) / (double)d_model));
  torch::Tensor pos_w = torch::arange(0, width).unsqueeze(1);
  torch::Tensor pos_h = torch::arange(0., height).unsqueeze(1);
  positionalEncoding.slice(0, d_model, 2) = torch::sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
  positionalEncoding.slice(1, d_model, 2) = torch::cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
  positionalEncoding.index({d_model, nullptr, 2}) = torch::sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
  positionalEncoding.index({d_model + 1, nullptr, 2}) = torch::cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
}

torch::Tensor Encoding::forward(torch::Tensor x) {
  x = torch::relu(conv1_1->forward(x));
  x = torch::relu(conv1_2->forward(x));
  x = torch::max_pool2d(x, 2);

  x = torch::relu(conv2_1->forward(x));
  x = torch::relu(conv2_2->forward(x));
  x = torch::max_pool2d(x, 2);

  x = torch::relu(conv3_1->forward(x));
  x = torch::relu(conv3_2->forward(x));
  x = torch::relu(conv3_3->forward(x));
  x = torch::max_pool2d(x, 2);

  x = torch::relu(conv4_1->forward(x));
  x = torch::relu(conv4_2->forward(x));
  x = torch::relu(conv4_3->forward(x));
  x = torch::max_pool2d(x, 2);

  x = torch::relu(conv5_1->forward(x));
  x = torch::relu(conv5_2->forward(x));
  x = torch::relu(conv5_3->forward(x));

  x = x.view({-1, 130});

  x = torch::relu(fc1->forward(x));
  x = torch::relu(fc2->forward(x));
  x = fc3->forward(x);

  return torch::log_softmax(x, 1);
}

OCREngine::OCREngine() : encoder(4, 640, 640) {
  std::cout << "test" << std::endl;
}

OCREngine::OCREngine(const std::string &modelPath) {
  //std::shared_ptr<OCREngine> ptr(this);
  //torch::load(ptr, modelPath);
}

torch::Tensor OCREngine::forward(torch::Tensor input) {
  return torch::Tensor();
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

std::string OCREngine::toLatex(const cv::cuda::GpuMat &pixels, const ImageType &type) {
  torch::Tensor imageTensor = ImageUtils::toTensor(pixels, torch::kByte);

  if (type == TEXT) {
  } else if (type == MATH) {
  } else if (type == IMAGE) {
  } else if (type == TABLE) {
  } else {
    std::cerr << "Model was not able to detect the type of blob(Should never happen not valid enum int)" << std::endl;
    exit(PROCESSING_ERROR);
  }

  torch::Tensor prediction = forward(imageTensor);
  std::ostringstream stream;
  stream << prediction;
  return stream.str();
}
