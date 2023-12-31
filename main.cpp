#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"

const int INVALID_PARAMETER_COUNT = 1;
const int INVALID_PARAMETER = 2;
const int ALLOC_ERROR = 3;
const int READ_ERROR = 4;
const int PROCESSING_ERROR = 5;
const int ENVIRONMENT_ERROR = 6;

const std::string VERSION = "0.1";

Options settings;

void printHelp() {
  std::cout << "usage: MathOCR [-v | --version]\n[-h | --help]\n[PDF or IMAGE PATH] [OUTPUT DIRECTORY]\nnormalize [PDF OR IMAGE PATH]\ntrain [dataFolder][epochs][learningRate]" << std::endl;
}

std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> preprocess(cv::cuda::GpuMat &pixels) {
  ImageUtils::equalize(pixels);
  ImageUtils::threshold(pixels);
  ImageUtils::crop(pixels);

  std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> imageBlocks = ImageUtils::getImageBlocks(pixels);
  if (settings.deskew) {
    float skewSum = 0;
    for (auto & imageBlock : imageBlocks)
      skewSum += ImageUtils::getSkewAngle(pixels(imageBlock.first), imageBlock.second);
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
  if(!cv::cuda::getCudaEnabledDeviceCount()) {
    std::cerr << "Program requires CUDA enabled device" << std::endl;
    std::cerr << cv::getBuildInformation() << std::endl;
    exit(ENVIRONMENT_ERROR);
  }

  if (argc == 1) {
    std::cerr << "did not supply parameters, use /help for usage" << std::endl;
    exit(INVALID_PARAMETER_COUNT);
  }

  std::string argOne(argv[1]);
  if (argOne == "-h" || argOne == "--help") {
    printHelp();
  } else if (argOne == "-v" || argOne == "--version") {
    std::cout << "MathOCR version: " << VERSION << std::endl;
  } else if (argOne == "train") {
    std::filesystem::path dataDirectory(argv[2]);
    LatexOCR::LatexOCREngine latexOCR;
    int batchSize = std::stoi(argv[3]);
    unsigned long epoch = std::stoul(argv[4]);
    float learningRate = std::stof(argv[5]);
    LatexOCR::DataSet dataset(dataDirectory, LatexOCR::DataSet::OCRMode::TRAIN);
    latexOCR->train(dataset, batchSize, epoch, learningRate);
    latexOCR->exportWeights("../models/ocr.pt");
  } else if (argOne == "normalize") {
    std::filesystem::path inputFilePath(argv[2]);
    std::string mode(argv[3]);

    if(mode != "images" && mode != "vocab" && mode != "both") {
      std::cerr << "Invalid normalization mode: " << mode << std::endl;
      exit(INVALID_PARAMETER);
    }

    if(mode == "images" || mode == "both") {
      std::filesystem::path outputDirectory = inputFilePath / "formula_images";
      for(const auto& item : std::filesystem::directory_iterator(outputDirectory)) {
        if(item.is_regular_file()) {
          std::filesystem::path fileType = inputFilePath.extension();
          if(fileType == ".jpg" || fileType == ".png" || fileType == ".tiff") {
            cv::cuda::GpuMat pixels(cv::imread(item.path().generic_string()));
            ImageUtils::equalize(pixels);
            ImageUtils::threshold(pixels);
            ImageUtils::crop(pixels);
            cv::Mat out;
            pixels.download(out);
            cv::imwrite(item.path().generic_string(), out);
          } else {
            std::cerr << "invalid input format: " << fileType << std::endl;
            exit(INVALID_PARAMETER);
          }
        }
      }
    }

    if(mode == "vocab" || mode == "both") {
      std::filesystem::path formulaFile = inputFilePath / "im2latex_formulas.lst";
      std::filesystem::path formulaNormFile = inputFilePath / "im2latex_formulas.norm.lst";
      std::filesystem::path vocabFile = inputFilePath / "vocab.txt";
      OCRUtils::normalizeLatex(formulaFile, formulaNormFile);

      std::unordered_map<std::string, int> freqMap;
      std::string curToken;
      std::ifstream formulaStream(formulaNormFile);
      while (formulaStream >> curToken)
        ++freqMap[curToken];

      std::vector<std::pair<std::string, int>> vocab(freqMap.begin(), freqMap.end());
      std::sort(vocab.begin(), vocab.end(), [&freqMap](const std::pair<std::string, int>& p1, const std::pair<std::string, int>& p2) -> bool {
        return freqMap[p1.first] > freqMap[p2.first];
      });

      std::ofstream vocabStream(vocabFile);
      size_t len = vocab.size();
      for (size_t i = 0; i < len; ++i)
        if(vocab[i].second >= ((float)len * 0.01)) // CAN CHANGE THRESHOLD
          vocabStream << vocab[i].first << " " << vocab[i].second << std::endl;
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
