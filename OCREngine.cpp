//
// Created by Pramit Govindaraj on 5/22/2023.
//

#include "OCREngine.h"
#include "utils.h"

#include <filesystem>
#include <random>
#include <utility>

#include <opencv2/cudawarping.hpp>
#include <torch/script.h>
#include <torch_tensorrt/logging.h>
#include <torch_tensorrt/torch_tensorrt.h>

LatexOCREngineImpl::DataSet::DataSet(std::filesystem::path inputPath, OCRMode mode) : mode(mode), formulasFile(inputPath / "im2latex_formulas.lst"), formulasFolder(inputPath / "formula_images") {
  std::filesystem::path trainingFile(std::move(inputPath));
  switch (mode) {
    case OCRMode::TRAIN:
      trainingFile /= "im2latex_train.lst";
      break;
    case OCRMode::VAL:
      trainingFile /= "im2latex_validate.lst";
      break;
    case OCRMode::TEST:
      trainingFile /= "im2latex_test.lst";
      break;
    default:
      std::cerr << "Invalid Mode" << std::endl;
      exit(INVALID_PARAMETER);
  }
  std::ifstream trainStream(trainingFile);
  if (!trainStream) {
    std::cerr << "Training file " << trainingFile.generic_string() << " does not exist" << std::endl;
    exit(READ_ERROR);
  }
  int lineNum;
  std::string fileName;
  while (trainStream >> lineNum >> fileName)
    itemLocations.emplace_back(lineNum, formulasFolder / fileName += ".png");
}

void LatexOCREngineImpl::DataSet::resize(cv::cuda::GpuMat &pixels) {
  cv::cuda::resize(pixels, pixels, cv::Size(EncoderImpl::IMG_SIZE, EncoderImpl::IMG_SIZE), cv::INTER_CUBIC);
}

torch::data::Example<> LatexOCREngineImpl::DataSet::Collate::apply_batch(std::vector<torch::data::Example<>> data) {
  std::vector<torch::Tensor> imgs, lbls;
  imgs.reserve(data.size());
  lbls.reserve(data.size());
  for (torch::data::Example<> &item: data) {
    torch::nn::ConstantPad1d lblPadFunc(torch::nn::ConstantPad1dOptions({0, MAX_LABEL_LEN - item.target.size(0)}, 0));

    item.target = lblPadFunc(item.target);
    item.data = item.data.squeeze(0);

    imgs.emplace_back(item.data);
    lbls.emplace_back(item.target);
  }
  return {torch::stack(imgs), torch::stack(lbls)};
}

torch::data::Example<> LatexOCREngineImpl::DataSet::get(size_t idx) {
  std::pair<int, std::filesystem::path> itemLocation = itemLocations[idx];

  std::ifstream texStream(formulasFile);
  int len = itemLocation.first - 1;
  for (int i = 0; i < len; ++i)
    texStream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string tex;
  std::getline(texStream, tex);

  cv::cuda::GpuMat imgMat(cv::imread(itemLocation.second.generic_string(), cv::IMREAD_GRAYSCALE));
  resize(imgMat);
  torch::Tensor imageTensor = ImageUtils::toTensor(imgMat, torch::kByte);

  torch::Tensor labelTensor = OCRUtils::toTensor(tex);
  return {imageTensor, labelTensor};
}

void createTorchTensorRT(torch::jit::Module &jitModule, const std::vector<int64_t> &dims, const std::filesystem::path &outputFile) {
  auto input = torch_tensorrt::Input(dims, torch::kFloat);
  auto compile_settings = torch_tensorrt::ts::CompileSpec({input});
  compile_settings.enabled_precisions = {torch::kFloat};
  compile_settings.truncate_long_and_double = true;
  jitModule = torch_tensorrt::ts::compile(jitModule, compile_settings);
  jitModule.save(outputFile.generic_string());
}

Classifier::Classifier() {
  try {
    torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::kERROR);
    classificationModule = torch::jit::load("classify.torchscript", torch::kCUDA);
    //classificationModule = torch::jit::load("../models/classifyBest.torchscript", torch::kCUDA);
    classificationModule.to(torch::kFloat);
    classificationModule.eval();

    //DEBUG GENERATE TORCH-TENSORRT
    /*
     std::vector<int64_t> dims = {1, 3, 640, 640};
     createTorchTensorRT(classificationModule, dims, "../models/classify.torchscript");
     */
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

EncoderImpl::FeedForwardImpl::FeedForwardImpl(int64_t dim, int64_t hiddenDim) {
  net = register_module("feedForwardNet", torch::nn::Sequential(torch::nn::LayerNorm(std::vector<int64_t>({dim})),
                                                                torch::nn::Linear(dim, hiddenDim),
                                                                torch::nn::GELU(),
                                                                torch::nn::Linear(hiddenDim, dim)));
}

torch::Tensor EncoderImpl::FeedForwardImpl::forward(const torch::Tensor &input) {
  return net->forward(input);
}

EncoderImpl::AttentionImpl::AttentionImpl(int64_t dim, int64_t heads = 8, int64_t dimHead = 64) : heads(heads),
                                                                                                  scale((float) 1 / (float) (dimHead * dimHead)) {
  norm = register_module("norm", torch::nn::LayerNorm(std::vector<int64_t>({dim})));
  attend = register_module("attend", torch::nn::Softmax(-1));
  toQkv = register_module("toQkv", torch::nn::Linear(torch::nn::LinearOptions(dim, dimHead * heads * 3).bias(false)));
  toOut = register_module("toOut", torch::nn::Linear(torch::nn::LinearOptions(dimHead * heads * 3, dim).bias(false)));
}

torch::Tensor EncoderImpl::AttentionImpl::forward(torch::Tensor input) {
  input = norm->forward(input);

  std::vector<torch::Tensor> qkv = toQkv->forward(input).chunk(3, 1);
  torch::Tensor q = qkv[0].unflatten(2, {heads, qkv[0].size(2) / heads}).permute({0, 2, 1, 3});
  torch::Tensor k = qkv[1].unflatten(2, {heads, qkv[1].size(2) / heads}).permute({0, 2, 1, 3});
  torch::Tensor v = qkv[2].unflatten(2, {heads, qkv[2].size(2) / heads}).permute({0, 2, 1, 3});

  torch::Tensor dots = torch::matmul(q, k.transpose(-1, -2)) * scale;
  torch::Tensor attn = attend->forward(dots);
  torch::Tensor out = torch::matmul(attn, v).permute({0, 2, 1, 3}).flatten(2);
  return toOut->forward(out);
}

EncoderImpl::TransformerImpl::TransformerImpl(int64_t dim, int64_t depth, int64_t heads, int64_t dimHeads, int64_t mlpDim) {
  for (int64_t i = 0; i < depth; ++i)
    layers->push_back(torch::nn::ModuleList(AttentionImpl(dim, heads, dimHeads),
                                            FeedForwardImpl(dim, mlpDim)));
  layers = register_module("transformerLayers", layers);
}

torch::Tensor EncoderImpl::TransformerImpl::forward(torch::Tensor input) {
  size_t len = layers->size();
  for (size_t i = 0; i < len; ++i) {
    torch::nn::ModuleList transformerItems = static_cast<torch::nn::ModuleList>(layers[i]);
    input = transformerItems->at<AttentionImpl>(0).forward(input) + input;
    input = transformerItems->at<FeedForwardImpl>(1).forward(input) + input;
  }
  return input;
}

torch::Tensor EncoderImpl::positionalEncoding(const torch::Tensor &patches) {
  int64_t height = patches.size(1);
  int64_t width = patches.size(2);
  int64_t dim = patches.size(3);

  std::vector<torch::Tensor> xy = torch::meshgrid({torch::arange(height).to(torch::kCUDA),
                                                   torch::arange(width).to(torch::kCUDA)},
                                                  "ij");
  torch::Tensor x = xy[0];
  torch::Tensor y = xy[1];

  torch::Tensor omega = torch::arange(dim / 4).to(torch::kCUDA) / ((dim / 4) - 1);
  omega = 1.0 / (torch::pow(TEMPERATURE, omega));

  y = y.flatten().index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None)}) *
      omega.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice()});
  x = x.flatten().index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None)}) *
      omega.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice()});

  return torch::cat({x.sin(), x.cos(), y.sin(), y.cos()}, 1);
}

EncoderImpl::EncoderImpl(int64_t numClasses) {
  toPatchEmbedding->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
  toPatchEmbedding->push_back(torch::nn::Linear(256, 512));
  toPatchEmbedding->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({512})));
  toPatchEmbedding = register_module("toPatchEmbedding", toPatchEmbedding);
  transformer = register_module("encodingTransformer", Transformer(512, 6, 16, 64, 2048));
  toLatent = register_module("toLatent", torch::nn::Identity());
  linearHead = register_module("linearHead", torch::nn::Sequential(torch::nn::LayerNorm(std::vector<int64_t>({512})),
                                                                   torch::nn::Linear(512, numClasses)));
}

torch::Tensor EncoderImpl::forward(torch::Tensor input) {
  int64_t h = input.size(2);
  int64_t w = input.size(3);

  input = input.permute({0, 2, 3, 1}).unflatten(1, {h / PATCH_SIZE, PATCH_SIZE}).unflatten(3, {w / PATCH_SIZE, PATCH_SIZE}).permute({0, 1, 3, 2, 4, 5}).flatten(3);

  std::cout << input.sizes() << std::endl;

  torch::Tensor x = toPatchEmbedding->forward(input);
  torch::Tensor pe = positionalEncoding(x);
  x = x.flatten(1, (int64_t)x.sizes().size() - 1) + pe;

  x = transformer->forward(x);
  x = x.mean(1);

  x = toLatent->forward(x);
  x = linearHead->forward(x);

  return x;
}

DecoderImpl::DecoderImpl() {
}

torch::Tensor DecoderImpl::forward(torch::Tensor input) {
  return torch::Tensor();
}

LatexOCREngineImpl::LatexOCREngineImpl() : vocabMap() {
  encoder = register_module("OCREncoder", Encoder(400));
  decoder = register_module("OCRDecoder", Decoder());
}

LatexOCREngineImpl::LatexOCREngineImpl(const std::string &modelPath) {
  std::shared_ptr<LatexOCREngineImpl> ptr(this);
  torch::load(ptr, modelPath);
}

void LatexOCREngineImpl::train(DataSet dataset, int batchSize, size_t epoch, float learningRate) {
  decoder->train();
  encoder->train();

  namespace data = torch::data;
  auto dataLoader = data::make_data_loader<data::samplers::RandomSampler>(std::move(dataset.map(DataSet::Collate())), data::DataLoaderOptions(16).workers(2));

  auto start = std::chrono::high_resolution_clock::now();
  torch::optim::Adam optimizer(this->parameters(), learningRate);
  torch::nn::CTCLoss criterion;
  criterion->to(torch::kCUDA);
  for (size_t i = 1; i <= epoch; ++i) {
    for (auto &batch: *dataLoader) {
      torch::Tensor data = batch.data;
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

void LatexOCREngineImpl::test(const std::filesystem::path &dataDirectory) {
  decoder->eval();
  encoder->eval();

  DataSet dataset(dataDirectory, DataSet::OCRMode::VAL);
  namespace data = torch::data;
  auto dataLoader = data::make_data_loader<data::samplers::RandomSampler>(std::move(dataset.map(DataSet::Collate())), data::DataLoaderOptions(16).workers(2));

  size_t total = 0;
  size_t counter = 0;
  torch::NoGradGuard no_grad;
  for (auto &batch: *dataLoader) {
    torch::Tensor data = batch.data;
    torch::Tensor output = forward(data);
    torch::Tensor prediction = std::get<1>(output.max(2)).transpose(1, 0).contiguous().view(-1).to(torch::kCPU);
    torch::Tensor inputLens = torch::full(data.size(0), output.size(0), torch::TensorOptions(torch::kLong));
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

void LatexOCREngineImpl::exportWeights(const std::filesystem::path &outputPath) {
  std::shared_ptr<LatexOCREngineImpl> ptr(this);
  torch::save(ptr, (outputPath / "weights.pt").generic_string());
}

torch::Tensor LatexOCREngineImpl::forward(torch::Tensor input) {
  input = encoder->forward(input);
  input = decoder->forward(input);
  return input;
}

TesseractOCREngine::TesseractOCREngine() {
  api = new tesseract::TessBaseAPI();
  if (api->Init(nullptr, "eng")) {
    std::cerr << "Unable to create Tesseract object" << std::endl;
    exit(PROCESSING_ERROR);
  }
  api->SetPageSegMode(tesseract::PageSegMode::PSM_RAW_LINE);
}

TesseractOCREngine::~TesseractOCREngine() {
  api->End();
  delete api;
}

std::string TesseractOCREngine::doOCR(const cv::cuda::GpuMat &pixels) {
  cv::Mat img(pixels);
  api->SetImage((uchar *) img.data, img.cols, img.rows, img.channels(), (int)img.step1());
  return api->GetUTF8Text();
}

std::string OCREngine::toLatex(const cv::cuda::GpuMat &pixels) {
  cv::cuda::GpuMat pixelsCopy = pixels;
  LatexOCREngineImpl::DataSet::resize(pixelsCopy);
  torch::Tensor imageTensor = ImageUtils::toTensor(pixelsCopy, torch::kByte).transpose(1, 3).to(torch::kFloat32).sub(128).div(128);
  static LatexOCREngine latexOCR("weights.pt");
  torch::Tensor prediction = latexOCR->forward(imageTensor);
  return OCRUtils::toString(prediction)[0];
}

std::string OCREngine::toText(const cv::cuda::GpuMat &pixels) {
  static TesseractOCREngine tesseract;
  return tesseract.doOCR(pixels);
}

std::string OCREngine::toTable(const std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> &items, const std::filesystem::path &path) {
  return std::string();//TODO: LOOK AT TABLE SYNTAX
}

std::string OCREngine::toImage(const cv::cuda::GpuMat &pixels) {
  return std::string();//TODO: LOOK AT IMAGE SYNTAX PROBABLY ALSO REQUIRES INDEX COUNTER FOR SAVING IMAGE
}
