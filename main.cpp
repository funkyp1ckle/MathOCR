#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"

const int INVALID_PARAMETER_COUNT = 1;
const int INVALID_PARAMETER = 2;
const int ALLOC_ERROR = 3;
const int READ_ERROR = 4;
const int PROCESSING_ERROR = 5;

const std::string VERSION = "0.1";

Options settings;

void printHelp() {
  std::cout << "usage: MathOCR [-v | --version]\n[-h | --help]\n[PDF or IMAGE PATH] [OUTPUT DIRECTORY]\npreprocess [PDF OR IMAGE PATH]\ntrain [dataFolder][epochs][learningRate]" << std::endl;
}

std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> preprocess(cv::cuda::GpuMat &pixels) {
  ImageUtils::equalize(pixels);
  ImageUtils::threshold(pixels);
  ImageUtils::crop(pixels);

  std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> imageBlocks = ImageUtils::getImageBlocks(pixels);
  if (settings.deskew) {
    float skewSum = 0;
    for (auto itr = imageBlocks.begin(); itr != imageBlocks.end(); ++itr)
      skewSum += ImageUtils::getSkewAngle(pixels(itr->first), itr->second);
    ImageUtils::rotate(pixels, skewSum / (float) imageBlocks.size());
  }
  return imageBlocks;
}

void imgToLatex(cv::cuda::GpuMat &pixels, const std::filesystem::path &outputPrefix) {
  std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> imageBlocks = preprocess(pixels);
  std::vector<std::string> latexStrs;
  //TODO: ADD LOGIC TO MERGE TEXT INTO ONE STR UNTIL IT HITS ANOTHER TYPE
  for(auto itr = imageBlocks.begin(); itr != imageBlocks.end(); ++itr) {
    cv::cuda::GpuMat roi = pixels(itr->first);
    if(itr->second == Classifier::ImageType::MATH) {
      latexStrs.emplace_back(OCREngine::toLatex(roi));
    } else if(itr->second == Classifier::ImageType::TEXT) {
      latexStrs.emplace_back(OCREngine::toText(roi));
    } else if(itr->second == Classifier::ImageType::TABLE) {
      std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> tableBlocks = ImageUtils::getImageBlocks(roi);
      latexStrs.emplace_back(OCREngine::toTable(tableBlocks, outputPrefix));
    } else {
      latexStrs.emplace_back(OCREngine::toImage(roi));
    }
  }
  //TODO: ADD OUTPUT HANDLING
}

int main(int argc, char **argv) {
  if (argc == 1) {
    std::cerr << "did not supply parameters, use /help for usage" << std::endl;
    exit(INVALID_PARAMETER_COUNT);
  }

  std::string argOne(argv[1]);
  size_t argOneLen = argOne.size();
  int i;
  for (i = 0; i < argOneLen; ++i) {
    if (!isalpha(argOne[i]))
      continue;
    break;
  }
  argOne = argOne.substr(i);
  if (argOne == "h" || argOne == "help") {
    printHelp();
  } else if (argOne == "v" || argOne == "version") {
    std::cout << "MathOCR version: " << VERSION << std::endl;
  } else if (argOne == "train") {
    std::filesystem::path dataDirectory(argv[2]);
    LatexOCREngine latexOCR;
    latexOCR->to(torch::kCUDA);
    int batchSize = std::stoi(argv[3]);
    unsigned long epoch = std::stoul(argv[4]);
    float learningRate = std::stof(argv[5]);
    LatexOCREngineImpl::DataSet dataset(dataDirectory, LatexOCREngineImpl::DataSet::OCRMode::TRAIN);
    latexOCR->train(dataset, batchSize, epoch, learningRate);
    latexOCR->exportWeights("../models/ocr.pt");
  } else if (argOne == "preprocess") {
    std::filesystem::path inputFilePath(argv[2]);
    std::filesystem::path outputDirectory(argv[3]);
    settings.deskew = std::stoi(argv[4]) != 0;
    std::filesystem::path fileType = inputFilePath.extension();
    if(fileType == ".pdf") {
      std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(cv::cuda::GpuMat &)> bindedCallback = std::bind(static_cast<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(&)(cv::cuda::GpuMat &)>(preprocess), std::placeholders::_1);
      getPDFImages(inputFilePath, outputDirectory, bindedCallback);
    } else {
      std::cerr << "invalid input format" << std::endl;
      exit(INVALID_PARAMETER);
    }
  } else {
    if (argc < 3) {
      std::cerr << "There are not enough parameters supplied, use -help for usage" << std::endl;
      exit(INVALID_PARAMETER_COUNT);
    }
    std::filesystem::path inputFilePath(argv[1]);
    std::filesystem::path outputDirectory(argv[2]);
    std::filesystem::path inputFileEnding = inputFilePath.extension();
    if (inputFileEnding == ".pdf") {
      std::function<void(cv::cuda::GpuMat &, const std::filesystem::path &)> bindedCallback = std::bind(static_cast<void(&)(cv::cuda::GpuMat &, const std::filesystem::path &)>(imgToLatex), std::placeholders::_1, std::placeholders::_2);
      getPDFImages(inputFilePath, outputDirectory, bindedCallback);
    } else {
      std::cerr << "invalid input format" << std::endl;
      exit(INVALID_PARAMETER);
    }
  }
  return 0;
}
