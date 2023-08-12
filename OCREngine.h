//
// Created by Pramit Govindaraj on 5/22/2023.
//

#ifndef MATHOCR_OCRENGINE_H
#define MATHOCR_OCRENGINE_H

#include <filesystem>
#include <string>

#include "tesseract/baseapi.h"
#include <opencv2/core/cuda.hpp>
#include <torch/torch.h>

extern const int INVALID_PARAMETER;
extern const int READ_ERROR;
extern const int PROCESSING_ERROR;

static void createTorchTensorRT(torch::jit::Module& model, const std::vector<int64_t>& dims, const std::filesystem::path& outputFile);

class Classifier {
public:
  enum class ImageType {
    TEXT = 0,
    MATH = 1,
    IMAGE = 2,
    TABLE = 3
  };

  struct RectComparator {
    bool operator()(const cv::Rect &rect1, const cv::Rect &rect2) const {
      if (rect1.y < rect2.y)
        return true;
      else if (rect1.y > rect2.y)
        return false;
      else
        return rect1.x < rect2.x;
    }
  };

  Classifier();
  torch::Tensor forward(const torch::Tensor &input);

private:
  torch::jit::script::Module classificationModule;
};

class EncoderImpl : public torch::nn::Module {
public:
  class FeedForwardImpl : public torch::nn::Module {
  public:
    FeedForwardImpl(int64_t dim, int64_t hiddenDim);
    torch::Tensor forward(const torch::Tensor& input);
  private:
    torch::nn::Sequential net;
  };
  TORCH_MODULE(FeedForward);

  class AttentionImpl : public torch::nn::Module {
  public:
    AttentionImpl(int64_t dim, int64_t heads, int64_t dimHead);
    torch::Tensor forward(torch::Tensor input);
  private:
    int64_t heads;
    float scale;
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Softmax attend{nullptr};
    torch::nn::Linear toQkv{nullptr};
    torch::nn::Linear toOut{nullptr};
  };
  TORCH_MODULE(Attention);

  class TransformerImpl : public torch::nn::Module {
  public:
    TransformerImpl(int64_t dim, int64_t depth, int64_t heads, int64_t dimHeads, int64_t mlpDim);
    torch::Tensor forward(torch::Tensor input);
  private:
    torch::nn::ModuleList layers;
  };
  TORCH_MODULE(Transformer);

  explicit EncoderImpl(int64_t numClasses);
  torch::Tensor forward(torch::Tensor input);

  static torch::Tensor positionalEncoding(const torch::Tensor &input);

  const static int IMG_SIZE = 224;
  const static int PATCH_SIZE = 16;
private:
  const static int64_t TEMPERATURE = 10000;

  torch::nn::Sequential toPatchEmbedding;
  Transformer transformer{nullptr};
  torch::nn::Identity toLatent;
  torch::nn::Sequential linearHead;
};
TORCH_MODULE(Encoder);

class DecoderImpl : public torch::nn::Module {
public:
  DecoderImpl();
  torch::Tensor forward(torch::Tensor input);
};
TORCH_MODULE(Decoder);

class LatexOCREngineImpl : public torch::nn::Module {
public:
  class DataSet : public torch::data::datasets::Dataset<DataSet> {
  public:
    const static int MAX_LABEL_LEN = 999;

    enum class OCRMode {
      TRAIN,
      VAL,
      TEST
    };

    struct Collate : public torch::data::transforms::Collation<torch::data::Example<torch::Tensor, torch::Tensor>,
                                                               std::vector<torch::data::Example<torch::Tensor, torch::Tensor>>> {
      torch::data::Example<torch::Tensor, torch::Tensor> apply_batch(std::vector<torch::data::Example<torch::Tensor, torch::Tensor>> data) override;
    };

    explicit DataSet(std::filesystem::path inputPath, OCRMode mode);
    torch::data::Example<> get(size_t idx) override;
    torch::data::Example<> operator[](size_t idx) { return get(idx); }
    torch::optional<size_t> size() const override { return itemLocations.size(); }

    static void resize(cv::cuda::GpuMat& pixels);

  private:
    OCRMode mode;
    std::filesystem::path formulasFile;
    std::filesystem::path formulasFolder;
    std::vector<std::pair<int, std::filesystem::path>> itemLocations;
  };

  explicit LatexOCREngineImpl();
  explicit LatexOCREngineImpl(const std::string& modelPath);

  torch::Tensor forward(torch::Tensor input);

  void train(DataSet dataset, int batchSize, size_t epoch, float learningRate);
  void test(const std::filesystem::path &dataDirectory);
  void exportWeights(const std::filesystem::path &outputPath);

private:
  std::unordered_map<std::string, int> vocabMap;

  Encoder encoder{nullptr};
  Decoder decoder{nullptr};
};
TORCH_MODULE(LatexOCREngine);

class TesseractOCREngine {
public:
  TesseractOCREngine();
  ~TesseractOCREngine();

  static std::string doOCR(const cv::cuda::GpuMat &pixels);

private:
  static inline tesseract::TessBaseAPI *api;
};

class OCREngine {
public:
  static std::string toLatex(const cv::cuda::GpuMat &pixels);
  static std::string toText(const cv::cuda::GpuMat &pixels);
  static std::string toTable(const std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> &items, const std::filesystem::path &path);
  static std::string toImage(const cv::cuda::GpuMat &pixels);
};

#endif//MATHOCR_OCRENGINE_H
