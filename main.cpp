#include <filesystem>
#include <iostream>

#include "utils.h"

const int INVALID_PARAMETER_COUNT = 1;
const int INVALID_PARAMETER = 2;
const int ALLOC_ERROR = 3;
const int READ_ERROR = 4;
const int PROCESSING_ERROR = 5;

const std::string VERSION = "0.1";

Options settings;

void preprocess(cv::cuda::GpuMat &pixels, bool deskew) {
  ImageUtils::equalize(pixels);
  ImageUtils::threshold(pixels);
  ImageUtils::denoise(pixels);
  ImageUtils::crop(pixels);
  if (deskew) {
    std::map<cv::Rect, ImageType, RectComparator> imageBlocks = ImageUtils::getImageBlocks(pixels);
    float skewSum = 0;
    for (auto itr = imageBlocks.begin(); itr != imageBlocks.end(); ++itr)
      skewSum += ImageUtils::getSkewAngle(pixels(itr->first), itr->second);
    ImageUtils::rotate(pixels, skewSum / (float) imageBlocks.size());
  }
}

void preprocess(cv::cuda::GpuMat &pixels, bool deskew, const std::string &outputPath) {
  preprocess(pixels, deskew);
  cv::Mat out(pixels);
  cv::imwrite(outputPath, out);
}

std::string imgToLatex(cv::cuda::GpuMat &pixels, OCREngine &ocr) {
  preprocess(pixels, true);
  return ocr.toLatex(pixels);
}

void imgToLatex(cv::cuda::GpuMat &pixels, OCREngine &ocr, const std::string &outputPrefix) {
  std::string output = imgToLatex(pixels, ocr);
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
    OCREngine ocr;
    ocr.to(torch::kCUDA);
    std::string dataDirectory(argv[2]);
    winToNixFilePath(dataDirectory);
    unsigned long epoch = std::stoul(argv[3]);
    float learningRate = std::stof(argv[4]);
    ocr.train(dataDirectory, epoch, learningRate);
  } else if (argOne == "preprocess") {
    std::string inputFilePath(argv[2]);
    std::string outputDirectory(argv[3]);
    bool deskew = std::stoi(argv[4]) != 0;
    std::string fileType = inputFilePath.substr(inputFilePath.rfind(".") + 1);
    if(fileType == "pdf") {
      std::function<void(cv::cuda::GpuMat &, const std::string &)> bindedCallback = std::bind(static_cast<void(&)(cv::cuda::GpuMat &, bool, const std::string &)>(preprocess), std::placeholders::_1, deskew, std::placeholders::_2);
      getPDFImages(inputFilePath, outputDirectory, bindedCallback);
    } else if(fileType != "jpeg" && fileType != "jpg" && fileType != "png") {
      std::cerr << "invalid input format" << std::endl;
      exit(INVALID_PARAMETER);
    } else {
      cv::cuda::GpuMat pixels(cv::imread(inputFilePath, cv::IMREAD_GRAYSCALE));
      preprocess(pixels, deskew, outputDirectory + "/img.png");
    }
  } else {
    if (argc < 3) {
      std::cerr << "There are not enough parameters supplied, use -help for usage" << std::endl;
      exit(INVALID_PARAMETER_COUNT);
    }
    std::string inputFilePath(argv[1]);
    winToNixFilePath(inputFilePath);
    std::string outputDirectory(argv[2]);
    winToNixFilePath(outputDirectory);
    size_t slashIdx = inputFilePath.rfind('/');
    size_t dotIdx = inputFilePath.rfind('.');
    std::string inputFileEnding = inputFilePath.substr(dotIdx + 1);
    OCREngine ocr("weights.pth");
    ocr.to(torch::kCUDA);
    ImageUtils imgUtils;
    if (inputFileEnding == "pdf") {
      std::function<std::string(cv::cuda::GpuMat &)> bindedCallback = std::bind(static_cast<std::string(&)(cv::cuda::GpuMat &, OCREngine&)>(imgToLatex), std::placeholders::_1, ocr);
      getPDFImages(inputFilePath, outputDirectory, bindedCallback);
    } else if (inputFileEnding != "jpeg" && inputFileEnding != "jpg" && inputFileEnding != "png") {
      std::cerr << "invalid input format" << std::endl;
      exit(INVALID_PARAMETER);
    } else {
      std::filesystem::copy_file(inputFilePath, outputDirectory, std::filesystem::copy_options::overwrite_existing);
      outputDirectory += "/";
      std::string imgPath = outputDirectory;
      imgPath += inputFilePath.substr(slashIdx + 1);
      cv::Mat imgCPU = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
      cv::cuda::GpuMat img(imgCPU);
      if (img.empty()) {
        std::cerr << "Filed to read image file(" << imgPath << ") into cv::Mat" << std::endl;
        exit(ALLOC_ERROR);
      }
      std::string outputPrefix = outputDirectory;
      outputPrefix += inputFilePath.substr(slashIdx + 1, dotIdx - slashIdx - 1);
      imgToLatex(img, ocr, outputPrefix);
    }
  }
  return 0;
}
