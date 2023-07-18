//
// Created by Pramit Govindaraj on 5/22/2023.
//

#include "OCREngine.h"
#include "utils.h"

#include <filesystem>
#include <random>
#include <utility>

#include <torch/script.h>
#include <torch_tensorrt/logging.h>
#include <torch_tensorrt/torch_tensorrt.h>

DataSet::DataSet(std::filesystem::path inputPath, OCRMode mode) : mode(mode), formulasFile(inputPath / "im2latex_formulas.lst"), formulasFolder(inputPath / "formula_images") {
  std::filesystem::path trainingFile(std::move(inputPath));
  switch(mode) {
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
  if(!trainStream) {
    std::cerr << "Training file " << trainingFile.generic_string() << " does not exist" << std::endl;
    exit(READ_ERROR);
  }
  int lineNum;
  std::string fileName;
  while(trainStream >> lineNum >> fileName)
    itemLocations.emplace_back(lineNum, formulasFolder / fileName += ".png");
}

torch::data::Example<> DataSet::Collate::apply_batch(std::vector<torch::data::Example<>> data) {
  int64_t maxW = 0;
  int64_t maxH = 0;
  for(torch::data::Example<>& item : data) {
    if(item.data.size(2) > maxH)
      maxH = item.data.size(2);
    if(item.data.size(3) > maxW)
      maxW = item.data.size(3);
  }

  std::vector<torch::Tensor> imgs, lbls;
  imgs.reserve(data.size());
  lbls.reserve(data.size());
  for(torch::data::Example<>& item : data) {
    torch::nn::ConstantPad1d lblPadFunc(torch::nn::ConstantPad1dOptions({0, MAX_LABEL_LEN - item.target.size(0)}, 0));
    torch::nn::ConstantPad2d imgPadFunc(
        torch::nn::ConstantPad2dOptions({0, maxW - item.data.size(3), 0, maxH - item.data.size(2)}, 0));

    item.target = lblPadFunc(item.target);
    item.data = imgPadFunc(item.data.squeeze(0));

    imgs.emplace_back(item.data);
    lbls.emplace_back(item.target);
  }
  return {torch::stack(imgs), torch::stack(lbls)};
}

torch::data::Example<> DataSet::get(size_t idx) {
  std::pair<int, std::filesystem::path> itemLocation = itemLocations[idx];

  std::ifstream texStream(formulasFile);
  int len = itemLocation.first - 1;
  for(int i = 0; i < len; ++i)
    texStream.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
  std::string tex;
  std::getline(texStream, tex);

  cv::cuda::GpuMat imgMat(cv::imread(itemLocation.second.generic_string(), cv::IMREAD_GRAYSCALE));
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
      sigmoid(),
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

LatexOCREngineImpl::LatexOCREngineImpl()
    : vocabMap(OCRUtils::getVocabMap("")),
      encoder(),
      decoder(512, 32, 512, (int64_t)vocabMap.size()) {
  register_module("OCREncoder", encoder);
  register_module("OCRDecoder", decoder);

  encoder->to(torch::kCUDA);
  decoder->to(torch::kCUDA);
}

LatexOCREngineImpl::LatexOCREngineImpl(const std::string &modelPath) {
  std::shared_ptr<LatexOCREngineImpl> ptr(this);
  torch::load(ptr, modelPath);
}

torch::Tensor LatexOCREngineImpl::forward(torch::Tensor input, int64_t beamSize) {
  int64_t vocabSize = decoder->vocabSize;
  torch::NoGradGuard no_grad;
  torch::Tensor encoderOut = encoder->forward(std::move(input));
  int64_t encW = encoderOut.size(2);
  int64_t encH = encoderOut.size(3);
  int64_t encoderDim = encoderOut.size(1);

  encoderOut = encoderOut.view({1, -1, encoderDim});
  int64_t numPixels = encoderOut.size(1);

  encoderOut = encoderOut.expand({beamSize, numPixels, encoderDim});

  torch::Tensor kPrevWords = torch::full(beamSize, 0).to(torch::kCUDA); //0 is the idx of <START>
  torch::Tensor seqs = kPrevWords;
  torch::Tensor topKScores = torch::zeros({beamSize, 1}).to(torch::kCUDA);
  torch::Tensor seqsAlpha = torch::ones({beamSize, 1, encW, encH}).to(torch::kCUDA);

  std::vector<int64_t> completedSeqs;
  std::vector<float> completedSeqsAlpha;
  std::vector<float> completedSeqsScores;

  int step = 1;
  torch::Tensor h = decoder->initHiddenState(encoderOut);

  torch::Tensor seq;
  torch::Tensor alphas;
  while(true) {
    torch::Tensor embeddings = decoder->embedding->forward(kPrevWords).squeeze(1);
    std::pair<torch::Tensor, torch::Tensor> attentionOut = decoder->attention->forward(encoderOut, h);

    attentionOut.second = attentionOut.second.view({-1, encW, encH});
    torch::Tensor gate = decoder->sigmoid->forward(decoder->fBeta->forward(h));
    attentionOut.first *= gate;

    h = decoder->decodeStep->forward(torch::cat({embeddings, attentionOut.first}, 1), h);
    torch::Tensor scores = decoder->fc->forward(h);
    scores = torch::log_softmax(scores, 1);

    scores = topKScores.expand_as(scores) + scores;

    if(step == 1) {
      std::tuple<torch::Tensor, torch::Tensor> kTensors = scores[0].topk(beamSize, 0, true, true);
      topKScores = std::get<0>(kTensors);
      torch::Tensor topKWords = std::get<1>(kTensors);

      torch::Tensor prevWordIdx = topKWords / vocabSize;
      torch::Tensor nextWordIdx = topKWords % vocabSize;

      seqs = torch::cat({seqs[prevWordIdx], nextWordIdx.unsqueeze(1)}, 1);
      seqsAlpha = torch::cat({seqsAlpha[prevWordIdx], attentionOut.second[prevWordIdx].unsqueeze(1)}, 1);

      torch::Tensor incompleteIdx = nextWordIdx.not_equal(decoder->vocabSize - 1); //decoder->vocabSize - 1 idx of <END>
      torch::Tensor completeIdxTensor = std::get<0>(at::_unique(torch::cat({torch::arange(1, nextWordIdx.size(0)), incompleteIdx}))).contiguous();
      std::vector<int64_t> completeIdx(completeIdxTensor.data_ptr<int64_t>(), completeIdxTensor.data_ptr<int64_t>() + completeIdxTensor.numel());

      if(!completeIdx.empty()) {
        size_t len = completeIdx.size();
        for(size_t i = 0; i < len; ++i) {
          completedSeqs.push_back(seqs[completeIdx[i]].item<int64_t>());
          completedSeqsAlpha.push_back(seqsAlpha[completeIdx[i]].item<float>());
          completedSeqsScores.push_back(topKScores[completeIdx[i]].item<float>());
        }
      }
      beamSize -= static_cast<int64_t>(completeIdx.size());

      if(beamSize == 0)
        break;
      seqs = seqs[incompleteIdx];
      seqsAlpha = seqsAlpha[incompleteIdx];
      h = h[prevWordIdx[incompleteIdx]];

      encoderOut = encoderOut[prevWordIdx[incompleteIdx]];
      topKScores = topKScores[incompleteIdx].unsqueeze(1);
      kPrevWords = nextWordIdx[incompleteIdx].unsqueeze(1);

      if(step > 160)
        break;
      ++step;

      float i = *std::max(completedSeqsScores.begin(), completedSeqsScores.end());
      //TODO: FIX
      //seq = completedSeqs[i];
      //alphas = completedSeqsAlpha[i];
    }
  }

  return seq;
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
      torch::Tensor data = batch.data.to(torch::kCUDA);
      data = data.squeeze(1);
      optimizer.zero_grad();
      torch::Tensor prediction = forward(data, 5);
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
    batch.data.to(torch::kCUDA);
    torch::Tensor output = forward(batch.data, 5);
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

void LatexOCREngineImpl::exportWeights(const std::filesystem::path &outputPath) {
  std::shared_ptr<LatexOCREngineImpl> ptr(this);
  torch::save(ptr, (outputPath / "weights.pt").generic_string());
}

TesseractOCREngine::TesseractOCREngine() {
  if(!api->Init("", "eng")) {
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
  api->SetImage((uchar*)img.data, img.cols, img.rows, img.channels(), img.step1());
  return api->GetUTF8Text();
}

std::string OCREngine::toLatex(const cv::cuda::GpuMat &pixels) {
  torch::Tensor imageTensor = ImageUtils::toTensor(pixels, torch::kByte).transpose(1, 3).to(torch::kFloat32).sub(128).div(128);
  static LatexOCREngine latexOCR("weights.pt");
  torch::Tensor prediction = latexOCR->forward(imageTensor, 5);
  return OCRUtils::toString(prediction)[0];
}

std::string OCREngine::toText(const cv::cuda::GpuMat &pixels) {
  static TesseractOCREngine tesseract;
  return tesseract.doOCR(pixels);
}

std::string OCREngine::toTable(const std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> &items, const std::filesystem::path &path) {
  return std::string(); //TODO: LOOK AT TABLE SYNTAX
}

std::string OCREngine::toImage(const cv::cuda::GpuMat &pixels) {
  return std::string(); //TODO: LOOK AT IMAGE SYNTAX PROBABLY ALSO REQUIRES INDEX COUNTER FOR SAVING IMAGE
}
