#include <boost/filesystem/convenience.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "opencv2/cudawarping.hpp"
#include "utils.h"

const int INVALID_PARAMETER_COUNT = 1;
const int INVALID_PARAMETER = 2;
const int ALLOC_ERROR = 3;
const int READ_ERROR = 4;
const int PROCESSING_ERROR = 5;

const std::string VERSION = "0.1";

Options settings;

std::string imgToLatex(OCREngine &ocr, ImageUtils& imgUtils, cv::cuda::GpuMat &pixels, const std::string &outputPrefix) {
  imgUtils.equalize(pixels);
  imgUtils.threshold(pixels);
  imgUtils.denoise(pixels);
  imgUtils.crop(pixels);
  std::map<cv::Rect, ImageType, RectComparator> imageBlocks = imgUtils.getImageBlocks(pixels);
  float skewSum = 0;
  for (auto itr = imageBlocks.begin(); itr != imageBlocks.end(); ++itr)
    skewSum += imgUtils.getSkewAngle(pixels(itr->first), itr->second);
  imgUtils.rotate(pixels, skewSum / (float) imageBlocks.size());
  std::string latexStr;
  for(std::pair<cv::Rect, ImageType> img : imageBlocks) {
    std::string latexLine = ocr.toLatex(pixels(img.first), img.second);
    //TODO: ADD POSITIONING INFO
    latexStr += latexLine;
  }
  //TODO: ADD OUTPUT HANDLING
  return latexStr;
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
      getPDFImages(inputFilePath, outputDirectory, ocr, imgUtils, imgToLatex);
    } else if (inputFileEnding != "jpeg" && inputFileEnding != "jpg" && inputFileEnding != "png") {
      std::cerr << "invalid input format" << std::endl;
      exit(INVALID_PARAMETER);
    } else {
      boost::filesystem::copy_file(inputFilePath, outputDirectory, boost::filesystem::copy_options::overwrite_existing);
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
      std::string latexEncoding = imgToLatex(ocr, imgUtils, img, outputPrefix);
    }
  }
  return 0;
}
