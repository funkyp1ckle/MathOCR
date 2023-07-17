//
// Created by Pramit Govindaraj on 5/19/2023.
//

#include "utils.h"

#include <boost/process.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/highgui.hpp>

#include <utility>

KatexHandler::KatexHandler() : platform(v8::platform::NewDefaultPlatform()) {
  v8::V8::InitializeICU();
  v8::V8::InitializePlatform(platform.get());
  v8::V8::Initialize();

  parameters.array_buffer_allocator = v8::ArrayBuffer::Allocator::NewDefaultAllocator();
  _isolate = v8::Isolate::New(parameters);

  v8::Isolate::Scope isolate_scope(_isolate);
  v8::HandleScope handle_scope(_isolate);

  auto context = v8::Context::New(_isolate);
  v8::Context::Scope context_scope(context);

  std::ifstream file("../katex/katex.min.js");
  std::stringstream stream;
  stream << file.rdbuf();
  std::string source = stream.str();

  _run(source, context);

  _persistent_context = v8::UniquePersistent<v8::Context>(_isolate, context);
}

KatexHandler::~KatexHandler() {
  _persistent_context.Reset();
  _isolate->Dispose();
  v8::V8::Dispose();
  v8::V8::DisposePlatform();
  delete parameters.array_buffer_allocator;
}

v8::Local<v8::Value> KatexHandler::_run(const std::string &source,
                                        const v8::Local<v8::Context> &context) const {
  v8::EscapableHandleScope handle_scope(_isolate);
  v8::Local<v8::String> checked = v8::String::NewFromUtf8(_isolate, source.c_str()).ToLocalChecked();
  v8::Local<v8::Script> script = v8::Script::Compile(context, checked).ToLocalChecked();
  v8::TryCatch try_catch(_isolate);
  auto result = script->Run(context);
  if (result.IsEmpty()) {
    v8::String::Utf8Value exception(_isolate, try_catch.Exception());
    std::string what(*exception);
    std::cerr << what << std::endl;
    exit(PROCESSING_ERROR);
  }
  return handle_scope.Escape(result.ToLocalChecked());
}

void KatexHandler::_escape(std::string &code) {
  static std::vector<std::string> escapeReplacements = {R"(\\)", R"(\')"};
  static std::vector<std::regex> escapeFilters = {std::regex("\\\\"), std::regex("\\'")};
  size_t len = escapeFilters.size();
  for (size_t i = 0; i < len; ++i)
    code = std::regex_replace(code, escapeFilters[i], escapeReplacements[i]);
}

std::string KatexHandler::preprocess(const std::string &text) {
  v8::Isolate::Scope isolate_scope(_isolate);
  v8::HandleScope handle_scope(_isolate);
  auto context = v8::Local<v8::Context>::New(_isolate,
                                             _persistent_context);
  v8::Context::Scope context_scope(context);
  std::string code = "var text = '" + text + "'";
  code += KATEX_NORMALIZE;//TODO: ADD NORMALIZE STRING AFTER TESTING
  v8::String::Utf8Value value(_isolate, _run(code, context));
  return {*value};
}

void OCRUtils::normalizeLatex(const std::filesystem::path &file) {
  static KatexHandler katex;
  std::ifstream texInStream(file.generic_string());
  std::string texContent;
  std::string line;
  static std::regex formatting(R"((hskip|hspace)(.*?)(cm|in|pt|mm|em))");
  while (std::getline(texInStream, line)) {
    texContent += std::regex_replace(line, formatting, "");
    texContent += ' ';
  }
  katex._escape(texContent);
  texContent = "<START>" + katex.preprocess(texContent) + "<END>";
  std::ofstream texOutStream(file.generic_string());
  texOutStream << texContent;
}

std::unordered_map<std::string, int> OCRUtils::getVocabMap(const std::filesystem::path &dataDirectory) {
  std::filesystem::path vocabFile("../models/vocab.txt");
  std::unordered_map<std::string, int> vocab;
  std::string token;
  int tokenId = -1;
  if (std::filesystem::exists(vocabFile)) {
    std::ifstream vocabInStream(vocabFile.generic_string());
    while (vocabInStream >> token) {
      if (vocab.find(token) == vocab.end())
        vocab.insert(std::make_pair(token, ++tokenId));
    }
  }
  if (dataDirectory.empty())
    return vocab.empty() ? std::unordered_map<std::string, int>({{"<START>", 0}, {"<END>", 0}}) : vocab;
  for (const auto &entry: std::filesystem::directory_iterator(dataDirectory)) {
    std::string fileStr = entry.path().generic_string();
    fileStr = fileStr.substr(0, fileStr.rfind('.'));
    fileStr += ".tex";
    std::filesystem::path file(fileStr);
    normalizeLatex(file);
    std::ifstream curTexStream(file.generic_string());
    while (curTexStream >> token) {
      if (vocab.find(token) == vocab.end())
        vocab.insert(std::make_pair(token, ++tokenId));
    }
  }
  std::ofstream vocabOutStream(vocabFile.generic_string());
  std::unordered_map<std::string, int>::const_iterator itr;
  for (itr = vocab.begin(); itr != vocab.end(); ++itr)
    vocabOutStream << itr->first << std::endl;
  return vocab;
}

torch::Tensor OCRUtils::toTensor(const std::string &str) {
  size_t len = str.size();
  torch::Tensor tensor = torch::empty(static_cast<int64_t>(len + 1), torch::TensorOptions(torch::kInt8));
  int i;
  for (i = 0; i < len; ++i)
    tensor[i] = str[i];
  tensor[i] = -1;
  return tensor;
}

std::vector<std::string> OCRUtils::toString(const torch::Tensor &tensor) {
  int64_t batchSize = tensor.size(0);
  std::vector<std::string> answer;
  answer.reserve(batchSize);
  for (int64_t i = 0; i < batchSize; ++i) {
    std::string item;
    signed char curChar = (signed char) tensor[i][0].item<schar>();
    for (int j = 0; curChar != -1; ++j, curChar = (signed char) tensor[i][j].item<schar>())
      item += curChar;
    answer.emplace_back(item);
  }
  return answer;
}

cv::cuda::GpuMat ImageUtils::toMat(const torch::Tensor &tensor, bool isNormalized, bool cvFormat) {
  torch::Tensor imageTensor;
  if (cvFormat)
    imageTensor = tensor.permute({0, 2, 3, 1}).squeeze_(0);
  else
    imageTensor = tensor.unsqueeze_(3);
  if (isNormalized)
    imageTensor = imageTensor.mul_(255);
  imageTensor = imageTensor.to(torch::kByte);
  torch::IntArrayRef dimensions = imageTensor.sizes();
  if (dimensions.size() != 3) {
    std::cerr << "Invalid dimension" << std::endl;
    exit(INVALID_PARAMETER);
  }
  return {cv::Size((int) dimensions[1], (int) dimensions[0]), CV_8UC1, imageTensor.data_ptr<uchar>(), static_cast<size_t>(tensor.stride(2))};
}

torch::Tensor ImageUtils::toTensor(const cv::cuda::GpuMat &matrix, torch::ScalarType size, int channels) {
  if (matrix.channels() != 1) {
    std::cerr << "Invalid number of channels" << std::endl;
    exit(INVALID_PARAMETER);
  }
  auto options = torch::TensorOptions().dtype(size).device(torch::kCUDA);
  return torch::from_blob(matrix.data, {1, static_cast<int64_t>(channels), static_cast<int64_t>(matrix.rows), static_cast<int64_t>(matrix.cols)},
                          {1, 1, (long long) (matrix.step / sizeof(size)), static_cast<int64_t>(channels)},
                          torch::Deleter(), options);
}

void ImageUtils::addMargin(const cv::cuda::GpuMat &pixels, cv::Rect_<int> &rect, int margin) {
  rect.x = rect.x - margin < 0 ? 0 : rect.x - margin;
  rect.y = rect.y - margin < 0 ? 0 : rect.y - margin;
  rect.width = rect.x + rect.width + (2 * margin) >= pixels.cols ? pixels.cols - rect.x : rect.width + (2 * margin);
  rect.height = rect.y + rect.height + (2 * margin) >= pixels.rows ? pixels.rows - rect.y : rect.height + (2 * margin);
}

std::map<cv::Rect, ImageType, RectComparator> ImageUtils::getImageBlocks(const cv::cuda::GpuMat &pixels) {
  static cv::cuda::GpuMat resized;
  constexpr float confThres = 0.25f;
  constexpr float iouThres = 0.5f;
  constexpr int maxWh = 4096;
  constexpr int maxNms = 30000;
  float scaleX = (float) pixels.cols / 640;
  float scaleY = (float) pixels.rows / 640;
  cv::cuda::resize(pixels, resized, cv::Size(640, 640), scaleX, scaleY, cv::INTER_AREA);
  torch::NoGradGuard no_grad;
  torch::Tensor imgTensor = toTensor(resized, torch::kByte).contiguous().to(torch::kFloat).div(255).expand({1, 3, -1, -1});
  static Classifier imgClassification;
  torch::Tensor prediction = imgClassification.forward(imgTensor).to(torch::kCUDA);
  std::map<cv::Rect, ImageType, RectComparator> imageBlocks;

  auto conf_mask = prediction.select(2, 4) > confThres;

  torch::Tensor output = torch::zeros({0, 6});
  prediction = prediction[0];
  prediction = prediction.index_select(0, torch::nonzero(conf_mask[0]).select(1, 0));
  if (prediction.size(0) == 0)
    return imageBlocks;

  prediction.slice(1, 5, prediction.size(1)).mul_(prediction.slice(1, 4, 5));
  torch::Tensor boxIn = prediction.slice(1, 0, 4);
  torch::Tensor boxOut = boxIn.clone();
  boxOut.select(1, 0) = boxIn.select(1, 0) - boxIn.select(1, 2).div(2);
  boxOut.select(1, 1) = boxIn.select(1, 1) - boxIn.select(1, 3).div(2);
  boxOut.select(1, 2) = boxIn.select(1, 0) + boxIn.select(1, 2).div(2);
  boxOut.select(1, 3) = boxIn.select(1, 1) + boxIn.select(1, 3).div(2);

  std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(prediction.slice(1, 5, prediction.size(1)), 1, true);
  prediction = torch::cat({boxOut, std::get<0>(max_tuple), std::get<1>(max_tuple)}, 1);
  prediction = prediction.index_select(0, torch::nonzero(std::get<0>(max_tuple) > confThres).select(1, 0));
  int64_t n = prediction.size(0);
  if (n == 0)
    return imageBlocks;

  if (n > maxNms)
    prediction = prediction.index_select(0, prediction.select(1, 4).argsort(0, true).slice(0, 0, maxNms));

  torch::Tensor c = prediction.slice(1, 5, 6) * maxWh;
  torch::Tensor bboxes = prediction.slice(1, 0, 4) + c, scores = prediction.select(1, 4);

  auto x1 = bboxes.select(1, 0);
  auto y1 = bboxes.select(1, 1);
  auto x2 = bboxes.select(1, 2);
  auto y2 = bboxes.select(1, 3);
  auto areas = (x2 - x1) * (y2 - y1);
  auto tuple_sorted = scores.sort(0, true);
  auto order = std::get<1>(tuple_sorted);

  std::vector<int> keep;
  while (order.numel() > 0) {
    if (order.numel() == 1) {
      auto i = order.item();
      keep.push_back(i.toInt());
      break;
    } else {
      auto i = order[0].item();
      keep.push_back(i.toInt());
    }

    auto order_mask = order.narrow(0, 1, order.size(-1) - 1);

    auto xx1 = x1.index({order_mask}).clamp(x1[keep.back()].item().toFloat(), 1e10);
    auto yy1 = y1.index({order_mask}).clamp(y1[keep.back()].item().toFloat(), 1e10);
    auto xx2 = x2.index({order_mask}).clamp(0, x2[keep.back()].item().toFloat());
    auto yy2 = y2.index({order_mask}).clamp(0, y2[keep.back()].item().toFloat());
    auto inter = (xx2 - xx1).clamp(0, 1e10) * (yy2 - yy1).clamp(0, 1e10);

    auto iou = inter / (areas[keep.back()] + areas.index({order.narrow(0, 1, order.size(-1) - 1)}) - inter);
    auto idx = (iou <= iouThres).nonzero().squeeze();
    if (idx.numel() == 0) {
      break;
    }
    order = order.index({idx + 1});
  }
  torch::Tensor ix = torch::tensor(keep).to(torch::kCUDA);
  output = prediction.index_select(0, ix).cpu();
  //(left, top, right, bottom, confidence, type)

  //cv::Mat test(pixels);

  for (int i = 0; i < output.size(0); ++i) {
    int left = clamp((int) round(output[i][0].item<float>() * scaleX), 0, pixels.cols);
    int top = clamp((int) round(output[i][1].item<float>() * scaleY), 0, pixels.rows);
    int right = clamp((int) round(output[i][2].item<float>() * scaleX), 0, pixels.cols);
    int bottom = clamp((int) round(output[i][3].item<float>() * scaleY), 0, pixels.rows);
    cv::Rect rect(cv::Point(left, top), cv::Point(right, bottom));

    //cv::rectangle(test, rect, cv::Scalar(255, 255, 255), 3);

    std::pair<cv::Rect, ImageType> pair = std::make_pair(rect, static_cast<ImageType>(output[i][5].item<int>()));
    if (rect.area())
      imageBlocks.emplace(pair);
  }
  return imageBlocks;
}

float ImageUtils::getSkewAngle(const cv::cuda::GpuMat &pixels, const ImageType &type) {
  if (type == TABLE || type == IMAGE) {
    cv::Mat img(pixels);
    std::vector<cv::Point> points;
    cv::Mat_<uchar>::iterator it = img.begin<uchar>();
    cv::Mat_<uchar>::iterator end = img.end<uchar>();
    for (; it != end; ++it)
      if (*it)
        points.push_back(it.pos());

    cv::RotatedRect bbox = cv::minAreaRect(cv::Mat(points));
    return bbox.angle;
  } else {
    static cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(85, 255);
    static cv::Ptr<cv::cuda::HoughSegmentDetector> segmentDetector = cv::cuda::createHoughSegmentDetector(1, CV_PI / 180, 0, 20, 4096, 40);
    static cv::cuda::GpuMat edges;
    static cv::cuda::GpuMat lines;
    cannyDetector->detect(pixels, edges);
    segmentDetector->setMinLineLength((int) (pixels.cols * 0.33));
    segmentDetector->detect(edges, lines);
    std::vector<cv::Vec4i> linesVec;
    if (lines.cols == 0) {
      return 0.f;
    }
    linesVec.resize(lines.cols);
    cv::Mat linesCopyMat(1, lines.cols, CV_32SC4, (void *) &linesVec[0]);
    lines.download(linesCopyMat);
    float rotationAngle = 0;
    cv::Mat test(edges);
    for (unsigned i = 0; i < lines.cols; ++i) {
      line(test, cv::Point(linesVec[i][0], linesVec[i][1]), cv::Point(linesVec[i][2], linesVec[i][3]), cv::Scalar(255, 255, 255));
      rotationAngle += (float) (atan2((double) linesVec[i][3] - linesVec[i][1], (double) linesVec[i][2] - linesVec[i][0]));
    }
    return (float) ((rotationAngle / (float) lines.cols) * 180. / CV_PI);
  }
}

void ImageUtils::rotate(cv::cuda::GpuMat &pixels, float degree) {
  static cv::cuda::GpuMat rotated;
  cv::cuda::warpAffine(pixels, rotated, cv::getRotationMatrix2D(cv::Point2f((float) ((pixels.cols - 1) / 2.0), (float) ((pixels.rows - 1) / 2.0)), degree, 1.0),
                       pixels.size());
  pixels = rotated;
}

void ImageUtils::denoise(cv::cuda::GpuMat &pixels) {
  static cv::Ptr<cv::cuda::Filter> denoiseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1)));
  denoiseFilter->apply(pixels, pixels);
}

void ImageUtils::crop(cv::cuda::GpuMat &pixels) {
  cv::Mat pixelsMat(pixels);
  cv::Mat nonZeroMat;
  cv::findNonZero(pixelsMat, nonZeroMat);
  cv::Rect bbox = cv::boundingRect(nonZeroMat);
  addMargin(pixels, bbox, 3);
  pixels = pixels(bbox);
}

void ImageUtils::threshold(cv::cuda::GpuMat &pixels) {
  static cv::Ptr<cv::cuda::Filter> thresholdFilter = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
  static cv::cuda::GpuMat mean;
  thresholdFilter->apply(pixels, mean);
  cv::cuda::subtract(mean, 8, mean);
  cv::cuda::compare(pixels, mean, pixels, cv::CMP_LE);
}

GhostscriptHandler::GhostscriptHandler(std::filesystem::path outputFileDirectory,
                                       const std::variant<std::function<std::vector<std::string>(cv::cuda::GpuMat &)>,
                                                          std::function<std::map<cv::Rect, ImageType, RectComparator>(cv::cuda::GpuMat &)>> &callback) : callback(callback), ioContext(), asyncPipe(ioContext), outputFileDirectory(std::move(outputFileDirectory)), outputFormat("^Page [0-9]+\n$"), pageNum(0) {
  if (std::holds_alternative<std::function<std::vector<std::string>(cv::cuda::GpuMat &)>>(callback))
    callbackType = LATEX;
  else
    callbackType = PROCESS;
}

void GhostscriptHandler::run(const std::filesystem::path &inputFilePath) {
  if (!std::filesystem::exists(inputFilePath)) {
    std::cerr << "Input PDF does not exist" << std::endl;
    exit(INVALID_PARAMETER);
  }

  if (!std::filesystem::exists(outputFileDirectory))
    std::filesystem::create_directory(outputFileDirectory);
  outputPrefix = outputFileDirectory;
  outputPrefix /= "pageImgs";
  if (!std::filesystem::exists(outputPrefix))
    std::filesystem::create_directory(outputPrefix);
  fileName = inputFilePath.stem();
  outputPrefix /= fileName;

  process = boost::process::child("ghostscript -sDEVICE=tifflzw -sOutputFile=" + outputPrefix.generic_string() + "_page%d.tiff -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -dUseTrimBox -dBATCH -dNOPAUSE " + inputFilePath.generic_string(),
                                  boost::process::std_in.close(), boost::process::std_out > asyncPipe, boost::process::std_err > stderr, ioContext);
  boost::asio::async_read_until(asyncPipe, buffer, '\n', boost::bind(&GhostscriptHandler::processOutput, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
  ioContext.run();
}

void GhostscriptHandler::processOutput(const boost::system::error_code &ec, std::size_t size) {
  if (ec) {
    if (ec == boost::asio::error::broken_pipe)
      return processOutput();
    std::cerr << ec.message() << std::endl;
    exit(PROCESSING_ERROR);
  }
  std::string line((char *) buffer.data().data(), size);
  buffer.consume(size);
  if (std::regex_match(line, outputFormat))
    processOutput();
  boost::asio::async_read_until(asyncPipe, buffer, '\n', boost::bind(&GhostscriptHandler::processOutput, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
}

void GhostscriptHandler::processOutput() {
  if (pageNum != 0) {
    std::filesystem::path imgPath = outputPrefix;
    imgPath += "_page";
    imgPath += std::to_string(pageNum);
    imgPath += ".tiff";
    while (!std::filesystem::exists(imgPath))
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    curImg.upload(cv::imread(imgPath.generic_string(), cv::IMREAD_GRAYSCALE));
    if (curImg.empty()) {
      std::cerr << "Failed to read image file(" << imgPath << ") into cv::Mat" << std::endl;
      exit(ALLOC_ERROR);
    }
    if (callbackType == LATEX) {
      //TODO:ADD NEWPAGE COMMAND AND HANDLE OUTPUT
      std::vector<std::string> latex = std::get<std::function<std::vector<std::string>(cv::cuda::GpuMat &)>>(callback)(curImg);
    } else {
      std::filesystem::path outputFilePath = outputFileDirectory;
      outputFilePath /= fileName;
      outputFilePath += "_page";
      outputFilePath += std::to_string(pageNum);
      outputFilePath += ".png";
      std::get<std::function<std::map<cv::Rect, ImageType, RectComparator>(cv::cuda::GpuMat &)>>(callback)(curImg);
      cv::cuda::resize(curImg, curImg, cv::Size(640, 640), cv::INTER_AREA);
      cv::Mat out(curImg);
      cv::imwrite(outputFilePath.generic_string(), out);
    }
  }
  ++pageNum;
}

int GhostscriptHandler::done() {
  std::filesystem::remove_all(outputFileDirectory / "pageImgs");
  process.wait();
  return process.exit_code();
}

void getPDFImages(const std::filesystem::path &inputFilePath, const std::filesystem::path &outputFileDirectory, const std::variant<std::function<std::vector<std::string>(cv::cuda::GpuMat &)>, std::function<std::map<cv::Rect, ImageType, RectComparator>(cv::cuda::GpuMat &)>> &callback) {
  GhostscriptHandler ghostscriptHandler(outputFileDirectory, callback);
  //BENCHMARK
  //auto startTime = std::chrono::high_resolution_clock::now();
  ghostscriptHandler.run(inputFilePath);
  //BENCHMARK
  //std::cout << (std::chrono::high_resolution_clock::now() - startTime).count() << std::endl;
  int returnCode = ghostscriptHandler.done();
  if (returnCode != 0) {
    std::cerr << "Ghostscript was unable to parse images from the pdf" << std::endl;
    exit(returnCode);
  }
}

int clamp(int n, int lower, int upper) {
  return std::max(lower, std::min(n, upper));
}
