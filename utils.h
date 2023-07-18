//
// Created by Pramit Govindaraj on 5/19/2023.
//

#ifndef MATHOCR_UTILS_H
#define MATHOCR_UTILS_H

#include "OCREngine.h"

#include <v8.h>
#include <libplatform/libplatform.h>

#include <boost/asio/buffer.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/bind/bind.hpp>
#include <boost/process/async_pipe.hpp>
#include <boost/process/child.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <variant>

extern const int ALLOC_ERROR;

struct Options {
  bool deskew = false;
};

extern Options settings;

class KatexHandler {
public:
  KatexHandler();
  ~KatexHandler();
  std::string preprocess(const std::string &code);
  void _escape(std::string& code);

private:
  v8::Local<v8::Value> _run(const std::string &source, const v8::Local<v8::Context> &context) const;

  std::unique_ptr<v8::Platform> platform;
  v8::Isolate *_isolate;
  v8::UniquePersistent<v8::Context> _persistent_context;
  v8::Isolate::CreateParams parameters;

  const static inline std::string KATEX_NORMALIZE = R"(var global_str = ''var norm_str = ''// This is a LaTeX AST to LaTeX Renderer (modified version of KaTeX AST-> MathML).var groupTypes = {};groupTypes.mathord = function(group, options) {    if (options.font == "mathrm"){        for (i = 0; i < group.value.length; ++i ) {            if (group.value[i] == " ") {                norm_str = norm_str + group.value[i] + "\; ";            } else {                norm_str = norm_str + group.value[i] + " ";            }        }    } else {        norm_str = norm_str + group.value + " ";    }};groupTypes.textord = function(group, options) {    norm_str = norm_str + group.value + " ";};groupTypes.bin = function(group) {    norm_str = norm_str + group.value + " ";};groupTypes.rel = function(group) {    norm_str = norm_str + group.value + " ";};groupTypes.open = function(group) {    norm_str = norm_str + group.value + " ";};groupTypes.close = function(group) {    norm_str = norm_str + group.value + " ";};groupTypes.inner = function(group) {    norm_str = norm_str + group.value + " ";};groupTypes.punct = function(group) {    norm_str = norm_str + group.value + " ";};groupTypes.ordgroup = function(group, options) {    norm_str = norm_str + "{ ";    buildExpression(group.value, options);    norm_str = norm_str +  "} ";};groupTypes.text = function(group, options) {    norm_str = norm_str + "\\mathrm { ";    buildExpression(group.value.body, options);    norm_str = norm_str + "} ";};groupTypes.color = function(group, options) {    var inner = buildExpression(group.value.value, options);    var node = new mathMLTree.MathNode("mstyle", inner);    node.setAttribute("mathcolor", group.value.color);    return node;};groupTypes.supsub = function(group, options) {    buildGroup(group.value.base, options);    if (group.value.sub) {        norm_str = norm_str + "_ ";        if (group.value.sub.type != 'ordgroup') {            norm_str = norm_str + " { ";            buildGroup(group.value.sub, options);            norm_str = norm_str + "} ";        } else {            buildGroup(group.value.sub, options);        }    }    if (group.value.sup) {        norm_str = norm_str + "^ ";        if (group.value.sup.type != 'ordgroup') {            norm_str = norm_str + " { ";            buildGroup(group.value.sup, options);            norm_str = norm_str + "} ";        } else {            buildGroup(group.value.sup, options);        }    }};groupTypes.genfrac = function(group, options) {    if (!group.value.hasBartext) {        norm_str = norm_str + "\\binom ";    } else {        norm_str = norm_str + "\\frac ";    }    buildGroup(group.value.numer, options);    buildGroup(group.value.denom, options);};groupTypes.array = function(group, options) {    norm_str = norm_str + "\\begin{" + group.value.style + "} ";    if (group.value.style == "array" || group.value.style == "tabular") {        norm_str = norm_str + "{ ";        if (group.value.cols) {            group.value.cols.map(function(start) {                if (start) {                    if (start.type == "align") {                        norm_str = norm_str + start.align + " ";                    } else if (start.type == "separator") {                        norm_str = norm_str + start.separator + " ";                    }                }            });        } else {            group.value.body[0].map(function(start) {                norm_str = norm_str + "c ";            } );        }        norm_str = norm_str + "} ";    }    group.value.body.map(function(row) {        if (row.length > 1 || row[0].value.length > 0) {            if (row[0].value[0] && row[0].value[0].value == "\\htext") {                norm_str = norm_str + "\\htext ";                row[0].value = row[0].value.slice(1);            }            out = row.map(function(cell) {                buildGroup(cell, options);                norm_str = norm_str + "& ";            });            norm_str = norm_str.substring(0, norm_str.length-2) + "\\\\ ";        }    });     norm_str = norm_str + "\\end{" + group.value.style + "} ";};groupTypes.sqrt = function(group, options) {    var node;    if (group.value.index) {        norm_str = norm_str + "\\sqrt [ " + group.value.index + " ] ";        buildGroup(group.value.body, options);    } else {        norm_str = norm_str + "\\sqrt ";        buildGroup(group.value.body, options);    }};groupTypes.leftright = function(group, options) {    norm_str = norm_str + "\\left" + group.value.left + " ";    buildExpression(group.value.body, options);    norm_str = norm_str + "\\right" + group.value.right + " ";};groupTypes.accent = function(group, options) {    if (group.value.base.type != 'ordgroup') {        norm_str = norm_str + group.value.accent + " { ";        buildGroup(group.value.base, options);        norm_str = norm_str + "} ";    } else {        norm_str = norm_str + group.value.accent + " ";        buildGroup(group.value.base, options);    }};groupTypes.spacing = function(group) {    var node;    if (group.value == " ") {        norm_str = norm_str + "~ ";    } else {        norm_str = norm_str + group.value + " ";    }    return node;};groupTypes.op = function(group) {    var node;    // TODO(emily): handle big operators using the `largeop` attribute    if (group.value.symbol) {        // This is a symbol. Just add the symbol.        norm_str = norm_str + group.value.body + " ";    } else {        if (group.value.limits == false) {            norm_str = norm_str + "\\\operatorname { ";        } else {            norm_str = norm_str + "\\\operatorname* { ";        }        for (i = 1; i < group.value.body.length; ++i ) {            norm_str = norm_str + group.value.body[i] + " ";        }        norm_str = norm_str + "} ";    }};groupTypes.katex = function(group) {    var node = new mathMLTree.MathNode(        "mtext", [new mathMLTree.TextNode("KaTeX")]);    return node;};groupTypes.font = function(group, options) {    var font = group.value.font;    if (font == "mbox" || font == "hbox") {        font = "mathrm";    }    norm_str = norm_str + "\\" + font + " ";    buildGroup(group.value.body, options.withFont(font));    };groupTypes.delimsizing = function(group) {    var children = [];    norm_str = norm_str + group.value.funcName + " " + group.value.value + " ";};groupTypes.styling = function(group, options) {    norm_str = norm_str + " " + group.value.original + " ";    buildExpression(group.value.value, options);};groupTypes.sizing = function(group, options) {    if (group.value.original == "\\rm") {        norm_str = norm_str + "\\mathrm { ";         buildExpression(group.value.value, options.withFont("mathrm"));        norm_str = norm_str + "} ";    } else {        norm_str = norm_str + " " + group.value.original + " ";        buildExpression(group.value.value, options);    }};groupTypes.overtext = function(group, options) {    norm_str = norm_str + "\\overtext { ";    buildGroup(group.value.body, options);    norm_str = norm_str + "} ";    norm_str = norm_str;};groupTypes.undertext = function(group, options) {    norm_str = norm_str + "\\undertext { ";    buildGroup(group.value.body, options);    norm_str = norm_str + "} ";    norm_str = norm_str;};groupTypes.rule = function(group) {    norm_str = norm_str + "\\rule { "+group.value.width.number+" "+group.value.width.unit+"  } { "+group.value.height.number+" "+group.value.height.unit+ " } ";};groupTypes.llap = function(group, options) {    norm_str = norm_str + "\\llap ";    buildGroup(group.value.body, options);};groupTypes.rlap = function(group, options) {    norm_str = norm_str + "\\rlap ";    buildGroup(group.value.body, options);};groupTypes.phantom = function(group, options, prev) {    norm_str = norm_str + "\\phantom { ";    buildExpression(group.value.value, options);    norm_str = norm_str + "} ";};/*** Takes a list of nodes, builds them, and returns a list of the generated* MathML nodes. A little simpler than the HTML version because we don't do any* previous-node handling.*/var buildExpression = function(expression, options) {    var groups = [];    for (var i = 0; i < expression.length; i++) {        var group = expression[i];        buildGroup(group, options);    }    // console.log(norm_str);    // return groups;};/*** Takes a group from the parser and calls the appropriate groupTypes function* on it to produce a MathML node.*/var buildGroup = function(group, options) {    if (groupTypes[group.type]) {        groupTypes[group.type](group, options);    } else {        throw new katex.ParseError(            "Got group of unknown type: '" + group.type + "'");    }};function main(text) {    if (text.indexOf("matrix") == -1 && text.indexOf("cases")==-1 &&    text.indexOf("array")==-1 && text.indexOf("begin")==-1)  {        for (var i = 0; i < 300; i++) {            text = text.replace(/\\\\/, "\\,");        }    }    // global_str is tokenized version (build in parser.js)    // norm_str is normalized version build by renderer below.    try {        var tree = katex.__parse(text, {});        norm_str = global_str.replace(/\\label { .*? }/, "");        for (var i = 0; i < 300; ++i) {            text = text.replace(/{\\rm/, "\\mathrm{");            text = text.replace(/{ \\rm/, "\\mathrm{");            text = text.replace(/\\rm{/, "\\mathrm{");        }        buildExpression(tree, {});        for (var i = 0; i < 300; ++i) {            norm_str = norm_str.replace('SSSSSS', '$');            norm_str = norm_str.replace(' S S S S S S', '$');        }        return norm_str.replace(/\\label { .*? }/, "");    } catch (e) {        return e;    }})";
};

class OCRUtils {
public:
  static void normalizeLatex(const std::filesystem::path &file);
  static std::unordered_map<std::string, int> getVocabMap(const std::filesystem::path &dataDirectory);

  static torch::Tensor toTensor(const std::string &str);
  static std::vector<std::string> toString(const torch::Tensor &tensor);
};

class GhostscriptHandler {
public:
  enum class CallbackType {
    LATEX,
    PROCESS
  };

  GhostscriptHandler(std::filesystem::path outputFileDirectory,
                     const std::variant<std::function<void(cv::cuda::GpuMat &, const std::filesystem::path &)>,
                     std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(cv::cuda::GpuMat &)>> &callback);
  void processOutput(const boost::system::error_code &ec, std::size_t size);
  void processOutput();

  void run(const std::filesystem::path &inputFilePath);

  int done();

private:
  std::variant<std::function<void(cv::cuda::GpuMat &, const std::filesystem::path &)>, std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(cv::cuda::GpuMat &)>> callback;
  CallbackType callbackType;
  boost::asio::io_context ioContext;
  boost::process::async_pipe asyncPipe;
  boost::asio::streambuf buffer;

  boost::process::child process;

  std::filesystem::path outputFileDirectory;
  std::filesystem::path outputPrefix;
  std::filesystem::path fileName;

  int pageNum;

  std::regex outputFormat;

  cv::cuda::GpuMat curImg;
};

class ImageUtils {
public:
  static cv::cuda::GpuMat toMat(const torch::Tensor &tensor, bool isNormalized, bool cvFormat);
  static torch::Tensor toTensor(const cv::cuda::GpuMat &matrix, torch::ScalarType size, int channels = 1);
  static void addMargin(const cv::cuda::GpuMat &pixels, cv::Rect_<int> &rect, int margin);

  static void denoise(cv::cuda::GpuMat &pixels);
  static void crop(cv::cuda::GpuMat &pixels);
  static void threshold(cv::cuda::GpuMat &pixels);

  static std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator> getImageBlocks(const cv::cuda::GpuMat &pixels);
  static float getSkewAngle(const cv::cuda::GpuMat &pixels, const Classifier::ImageType &type);
  static void rotate(cv::cuda::GpuMat &pixels, float degree);
};

int clamp(int n, int lower, int upper);

void getPDFImages(const std::filesystem::path &inputFilePath, const std::filesystem::path &outputFileDirectory,
                  const std::variant<std::function<void(cv::cuda::GpuMat &, const std::filesystem::path &)>,
                 std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(cv::cuda::GpuMat &)>> &callback);
#endif//MATHOCR_UTILS_H
