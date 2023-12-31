//
// Created by Pramit Govindaraj on 5/19/2023.
//

#include "utils.h"

#include <boost/process.hpp>
#include <boost/regex.hpp>

#include <uv.h>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/highgui.hpp>

#include <utility>

KatexHandler::KatexHandler() {
  char *nodeArgsA[] = {(char *) "node"};
  char **nodeArgsP = nodeArgsA;
  nodeArgsP = uv_setup_args(1, nodeArgsP);
  std::vector <std::string> nodeArgs(nodeArgsP, nodeArgsP + 1);

  std::unique_ptr <node::InitializationResult> result = node::InitializeOncePerProcess(nodeArgs, {
    node::ProcessInitializationFlags::kNoInitializeV8,
    node::ProcessInitializationFlags::kNoInitializeNodeV8Platform
  });

  std::string errorCodes;
  for (const std::string &error: result->errors())
    std::cerr << "NodeJS Init Error: " << error << std::endl;

  if (!result->errors().empty() || result->early_return() != 0)
    exit(INVALID_PARAMETER);

  platform = node::MultiIsolatePlatform::Create(4);
  v8::V8::InitializePlatform(platform.get());
  v8::V8::Initialize();

  std::vector <std::string> errors;
  std::vector <std::string> execArgs;

  setup = node::CommonEnvironmentSetup::Create(platform.get(), &errors, nodeArgs, execArgs);
  if (!setup) {
    for (const std::string &error: errors)
      std::cerr << "NodeJS Environment Error: " << error << std::endl;
    exit(INVALID_PARAMETER);
  }
  v8::Isolate *setupIsolate = setup->isolate();
  node::Environment *env = setup->env();

  v8::Locker locker(setupIsolate);
  v8::Isolate::Scope isolateScope(setupIsolate);
  v8::HandleScope handleScope(setupIsolate);
  v8::Context::Scope contextScope(setup->context());

  v8::MaybeLocal <v8::Value> scriptReturn = node::LoadEnvironment(env, [&](
    const node::StartExecutionCallbackInfo &info) -> v8::MaybeLocal <v8::Value> {
    auto loadKatex = "const publicRequire = require('module').createRequire(process.cwd() + '/');"
                     "globalThis.require = publicRequire;"
                     "const katex = require('katex');"
                     "var global_str = '';"
                     "var norm_str = '';"
                     "var groupTypes = {};"
                     "const delimiterSizes = {"
                     "    \"mopen1\" : \"\\\\bigl\","
                     "    \"mopen2\" : \"\\\\Bigl\","
                     "    \"mopen3\" : \"\\\\biggl\","
                     "    \"mopen4\" : \"\\\\Biggl\","
                     "    \"mclose1\": \"\\\\bigr\","
                     "    \"mclose2\": \"\\\\Bigr\","
                     "    \"mclose3\": \"\\\\biggr\","
                     "    \"mclose4\": \"\\\\Biggr\","
                     "    \"mrel1\"  : \"\\\\bigm\","
                     "    \"mrel2\"  : \"\\\\Bigm\","
                     "    \"mrel3\"  : \"\\\\biggm\","
                     "    \"mrel4\"  : \"\\\\Biggm\","
                     "    \"mord1\"  : \"\\\\big\","
                     "    \"mord2\"  : \"\\\\Big\","
                     "    \"mord3\"  : \"\\\\bigg\","
                     "    \"mord4\"  : \"\\\\Bigg\""
                     "};"
                     "groupTypes.hbox = function (group, options) {"
                     "    norm_str = norm_str + \" \\\\mathrm{ \";"
                     "    buildExpression(group.body, options);"
                     "    norm_str = norm_str + \"} \";"
                     "};"
                     "groupTypes.cr = function (group, options) {"
                     "    norm_str = norm_str + \" \\\\\\\\\";"
                     "};"
                     "groupTypes.atom = function (group, options) {"
                     "    groupTypes[group.family](group, options);"
                     "};"
                     "groupTypes.kern = function (group, options) {"
                     "    switch (group.dimension.number) {"
                     "        case -3:"
                     "            norm_str = norm_str + \"\\\\! \";"
                     "            break;"
                     "        case 1:"
                     "            norm_str = group.dimension.unit === \"em\" ? norm_str + \"\\\\quad \" : norm_str + \"\\\\ \";"
                     "            break;"
                     "        case 2:"
                     "            norm_str = group.dimension.unit === \"em\" ? norm_str + \"\\\\qquad \" : norm_str + \"\\\\;\\\\! \";"
                     "            break;"
                     "        case 3:"
                     "            norm_str = norm_str + \"\\\\, \";"
                     "            break;"
                     "        case 4:"
                     "            norm_str = norm_str + \"\\\\: \";"
                     "            break;"
                     "        case 5:"
                     "            norm_str = norm_str + \"\\\\; \";"
                     "            break;"
                     "        case 18:"
                     "            norm_str = norm_str + \"\\\\quad \";"
                     "            break;"
                     "        case 36:"
                     "            norm_str = norm_str + \"\\\\qquad \";"
                     "            break;"
                     "        default:"
                     "            throw new katex.ParseError(\"Got unknown kern size: '\" + group.dimension.number + \"'\");"
                     "    }"
                     "};"
                     "groupTypes.internal = function (group, options) {"
                     "};"
                     "groupTypes.mathord = function (group, options) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.textord = function (group, options) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.bin = function (group) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.rel = function (group) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.open = function (group) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.close = function (group) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.inner = function (group) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.punct = function (group) {"
                     "    norm_str = norm_str + group.text + \" \";"
                     "};"
                     "groupTypes.ordgroup = function (group, options) {"
                     "    norm_str = norm_str + \"{ \";"
                     "    buildExpression(group.body, options);"
                     "    norm_str = norm_str + \"} \";"
                     "};"
                     "groupTypes.text = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\mathrm { \";"
                     "    buildExpression(group.body, options);"
                     "    norm_str = norm_str + \"} \";"
                     "};"
                     "groupTypes.color = function (group, options) {"
                     "    var inner = buildExpression(group.body[0], options);"
                     "    var node = new mathMLTree.MathNode(\"mstyle\", inner);"
                     "    node.setAttribute(\"mathcolor\", group.color);"
                     "    return node;"
                     "};"
                     "groupTypes.supsub = function (group, options) {"
                     "    buildGroup(group.base, options);"
                     "    if (group.sub) {"
                     "        norm_str = norm_str + \"_ \";"
                     "        if (group.sub.type !== 'ordgroup') {"
                     "            norm_str = norm_str + \"{ \";"
                     "            buildGroup(group.sub, options);"
                     "            norm_str = norm_str + \"} \";"
                     "        } else {"
                     "            buildGroup(group.sub, options);"
                     "        }"
                     "    }"
                     "    if (group.sup) {"
                     "        norm_str = norm_str + \"^ \";"
                     "        if (group.sup.type !== 'ordgroup') {"
                     "            norm_str = norm_str + \"{ \";"
                     "            buildGroup(group.sup, options);"
                     "            norm_str = norm_str + \"} \";"
                     "        } else {"
                     "            buildGroup(group.sup, options);"
                     "        }"
                     "    }"
                     "};"
                     "groupTypes.genfrac = function (group, options) {"
                     "    if (!group.hasBarLine) {"
                     "        norm_str = norm_str + \"\\\\binom \";"
                     "    } else {"
                     "        norm_str = norm_str + \"\\\\frac \";"
                     "    }"
                     "    buildGroup(group.numer, options);"
                     "    buildGroup(group.denom, options);"
                     "};"
                     "groupTypes.array = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\begin{array} { \";"
                     "    if (group.cols) {"
                     "        group.cols.map(function (start) {"
                     "            if (start && start.align) {"
                     "                norm_str = norm_str + start.align + \" \";"
                     "            }"
                     "        });"
                     "    } else {"
                     "        group.body[0].map(function (start) {"
                     "            norm_str = norm_str + \"l \";"
                     "        });"
                     "    }"
                     "    norm_str = norm_str + \"} \";"
                     "    group.body.map(function (row) {"
                     "        if (row.length > 0) {"
                     "            out = row.map(function (cell) {"
                     "                buildGroup(cell, options);"
                     "                norm_str = norm_str + \"& \";"
                     "            });"
                     "            norm_str = norm_str.substring(0, norm_str.length - 2) + \"\\\\\\\\ \";"
                     "        }"
                     "    });"
                     "    norm_str = norm_str + \"\\\\end{array} \";"
                     "};"
                     "groupTypes.sqrt = function (group, options) {"
                     "    if (group.index) {"
                     "        norm_str = norm_str + \"\\\\sqrt [ \";"
                     "        buildExpression(group.index.value, options);"
                     "        norm_str = norm_str + \"] \";"
                     "        buildGroup(group.body, options);"
                     "    } else {"
                     "        norm_str = norm_str + \"\\\\sqrt \";"
                     "        buildGroup(group.body, options);"
                     "    }"
                     "};"
                     "groupTypes.leftright = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\left\" + group.left + \" \";"
                     "    buildExpression(group.body, options);"
                     "    norm_str = norm_str + \"\\\\right\" + group.right + \" \";"
                     "};"
                     "groupTypes.accent = function (group, options) {"
                     "    if (group.base.type !== 'ordgroup') {"
                     "        norm_str = norm_str + group.label + \" { \";"
                     "        buildGroup(group.base, options);"
                     "        norm_str = norm_str + \"} \";"
                     "    } else {"
                     "        norm_str = norm_str + group.label + \" \";"
                     "        buildGroup(group.base, options);"
                     "    }"
                     "};"
                     "groupTypes.spacing = function (group) {"
                     "    if (group.text === \" \") {"
                     "        norm_str = norm_str + \"~ \";"
                     "    } else {"
                     "        norm_str = norm_str + group.text + \" \";"
                     "    }"
                     "};"
                     "groupTypes.op = function (group) {"
                     "    if (group.symbol) {"
                     "        norm_str = norm_str + group.name + \" \";"
                     "    } else {"
                     "        if (group.limits === false) {"
                     "            norm_str = norm_str + \"\\\\\\operatorname { \";"
                     "        } else {"
                     "            norm_str = norm_str + \"\\\\\\operatorname* { \";"
                     "        }"
                     "        for (let i = 1; i < group.name.length; ++i) {"
                     "            norm_str = norm_str + group.name[i] + \" \";"
                     "        }"
                     "        norm_str = norm_str + \"} \";"
                     "    }"
                     "};"
                     "groupTypes.katex = function (group) {"
                     "    return new mathMLTree.MathNode("
                     "        \"mtext\", [new mathMLTree.TextNode(\"KaTeX\")]);"
                     "};"
                     "groupTypes.font = function (group, options) {"
                     "    var font = group.font;"
                     "    if (font === \"mbox\" || font === \"hbox\") {"
                     "        font = \"mathrm\";"
                     "    }"
                     "    norm_str = norm_str + \"\\\\\" + font + \" \";"
                     "    let newOptions = options;"
                     "    newOptions[\"font\"] = font;"
                     "    buildGroup(group.body, newOptions);"
                     "};"
                     "groupTypes.delimsizing = function (group) {"
                     "    if(group.delim !== \".\")"
                     "        norm_str = norm_str + delimiterSizes[group.mclass + group.size] + \" \" + group.delim + \" \";"
                     "};"
                     "groupTypes.styling = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\\" + group.style + \"style \";"
                     "    buildExpression(group.body, options);"
                     "};"
                     "groupTypes.sizing = function (group, options) {"
                     "    if (group.value.original === \"\\\\rm\") {"
                     "        norm_str = norm_str + \"\\\\mathrm { \";"
                     "        let newOptions = options;"
                     "        newOptions[\"font\"] = \"mathrm\";"
                     "        buildExpression(group.body.type, newOptions);"
                     "        norm_str = norm_str + \"} \";"
                     "    } else {"
                     "        norm_str = norm_str + \" \" + group.value.original + \" \";"
                     "        buildExpression(group.body.type, options);"
                     "    }"
                     "};"
                     "groupTypes.overline = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\overline { \";"
                     "    buildGroup(group.body, options);"
                     "    norm_str = norm_str + \"} \";"
                     "};"
                     "groupTypes.underline = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\underline { \";"
                     "    buildGroup(group.body, options);"
                     "    norm_str = norm_str + \"} \";"
                     "};"
                     "groupTypes.rule = function (group) {"
                     "    norm_str = norm_str + \"\\\\rule { \" + group.value.width.number + \" \" + group.value.width.unit + \"  } { \" + group.value.height.number + \" \" + group.value.height.unit + \" } \";"
                     "};"
                     "groupTypes.llap = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\llap \";"
                     "    buildGroup(group.value.body, options);"
                     "};"
                     "groupTypes.rlap = function (group, options) {"
                     "    norm_str = norm_str + \"\\\\rlap \";"
                     "    buildGroup(group.value.body, options);"
                     "};"
                     "groupTypes.phantom = function (group, options, prev) {"
                     "    norm_str = norm_str + \"\\\\phantom { \";"
                     "    buildExpression(group.value.value, options);"
                     "    norm_str = norm_str + \"} \";"
                     "};"
                     "var buildExpression = function (expression, options) {"
                     "    for (let i = 0; i < expression.length; i++) {"
                     "        var group = expression[i];"
                     "        buildGroup(group, options);"
                     "    }"
                     "};"
                     "var buildGroup = function (group, options) {"
                     "    if (groupTypes[group.type]) {"
                     "        groupTypes[group.type](group, options);"
                     "    } else {"
                     "        throw new katex.ParseError(\"Got group of unknown type: '\" + group.type + \"'\");"
                     "    }"
                     "};"
                     ""
                     "function normalize(text) {"
                     "    try {"
                     "        global_str = '';"
                     "        norm_str = '';"
                     "        var tree = katex.__parse(text, {"
                     "            macros: {"
                     "                \"\\\\sp\": \"^\","
                     "                \"\\\\mbox\": \"\\\\mathrm\""
                     "            }"
                     "        });"
                     "        buildExpression(tree, {});"
                     "        for (let i = 0; i < 300; ++i) {"
                     "            norm_str = norm_str.replace('SSSSSS', '$');"
                     "            norm_str = norm_str.replace(' S S S S S S', '$');"
                     "        }"
                     "        return norm_str.replace(/\\\\label { .*? }/, \"\");"
                     "    } catch (e) {"
                     "        return e;"
                     "    }"
                     "}";
    v8::HandleScope scope(setupIsolate);
    v8::Local <v8::Context> setupContext = setupIsolate->GetCurrentContext();
    v8::Local <v8::Object> globalObject = setupContext->Global();

    v8::Local <v8::String> requireStr = v8::String::NewFromUtf8(setupIsolate, "require").ToLocalChecked();
    globalObject->Set(setupContext, requireStr, info.native_require).ToChecked();
    v8::Local <v8::String> processStr = v8::String::NewFromUtf8(setupIsolate, "process").ToLocalChecked();
    globalObject->Set(setupContext, processStr, info.process_object).ToChecked();
    v8::Local <v8::String> source = v8::String::NewFromUtf8(setupIsolate, loadKatex,
                                                            v8::NewStringType::kNormal).ToLocalChecked();
    v8::Local <v8::Script> depScript = v8::Script::Compile(setupContext, source).ToLocalChecked();
    v8::Local <v8::Value> result = depScript->Run(setupContext).ToLocalChecked();
    return v8::Null(setupIsolate);
  });

  if (scriptReturn.IsEmpty()) {
    std::cerr << "Loading Katex has unhandled exception" << std::endl;
    exit(PROCESSING_ERROR);
  }

  int exit_code = node::SpinEventLoop(env).FromMaybe(1);

  context = v8::Global<v8::Context>(setupIsolate, setup->context());
  isolate = setupIsolate;
}

KatexHandler::~KatexHandler() {
  node::TearDownOncePerProcess();
}

v8::Local <v8::Value> KatexHandler::run(const std::string &source, const v8::Local <v8::Context> &localContext) const {
  v8::EscapableHandleScope escapeHandleScope(isolate);
  auto checkedCode = v8::String::NewFromUtf8(isolate, source.c_str()).ToLocalChecked();
  auto script = v8::Script::Compile(localContext, checkedCode).ToLocalChecked();
  v8::TryCatch tryCatch(isolate);
  auto result = script->Run(localContext);
  if (result.IsEmpty()) {
    std::cerr << "Running Script has unhandled exception" << std::endl;
    std::cerr << *v8::String::Utf8Value(isolate, tryCatch.Exception()) << std::endl;
    exit(PROCESSING_ERROR);
  }
  return escapeHandleScope.Escape(result.ToLocalChecked());
}

void KatexHandler::escape(std::string &code) {
  static std::vector <std::string> escapeReplacements = {R"(\\)", R"(\')"};
  static std::vector <std::regex> escapeFilters = {std::regex("\\\\"), std::regex("'")};
  size_t len = escapeFilters.size();
  for (size_t i = 0; i < len; ++i)
    code = std::regex_replace(code, escapeFilters[i], escapeReplacements[i]);
}

std::string KatexHandler::normalize(const std::string &text) {
  v8::Locker locker(isolate);
  v8::Isolate::Scope isolateScope(isolate);
  v8::HandleScope handleScope(isolate);
  v8::Local <v8::Context> localContext = v8::Local<v8::Context>::New(isolate, context);
  v8::Context::Scope contextScope(localContext);
  std::string code = "normalize('" + text + "');";
  v8::Local <v8::Value> returnVal = run(code, localContext);
  v8::String::Utf8Value value(isolate, returnVal);
  std::string strVal{*value};
  return strVal.find("Error:") == std::string::npos ? strVal : "";
}

std::string KatexHandler::latexToHTML(std::string latex) {
  v8::Locker locker(isolate);
  v8::Isolate::Scope isolateScope(isolate);
  v8::HandleScope handleScope(isolate);
  v8::Local <v8::Context> localContext = v8::Local<v8::Context>::New(isolate, context);
  v8::Context::Scope contextScope(localContext);
  escape(latex);
  std::string code = "katex.renderToString('" + latex + "', {'displayMode': true});";
  v8::Local <v8::Value> returnVal = run(code, localContext);
  v8::String::Utf8Value value(isolate, returnVal);
  std::string answer = "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'/>\n<link rel='stylesheet' type='text/css' href='./katex/dist/katex.css' />\n</head>\n<body>\n";
  answer += *value;
  answer += "</body>\n</html>";
  return answer;
}

void replaceAll(std::string &str, const std::string &match, const std::string &replacement) {
  size_t start_pos = 0;
  while ((start_pos = str.find(match, start_pos)) != std::string::npos) {
    str.replace(start_pos, match.size(), replacement);
    start_pos += replacement.size();
  }
}

void KatexHandler::replaceUnsupported(std::string &str) {
  static std::vector <boost::regex> regex = {boost::regex("boldmath"),
                                             boost::regex(R"(\\(p?matrix)([{]((?>[^{}]+|(?2))*)[}]))")};
  static std::vector <std::string> replacement = {"bold", R"(\\begin\{$1\}$3\\end{$1})"};

  size_t numRegex = regex.size();
  for (size_t i = 0; i < numRegex; ++i)
    str = boost::regex_replace(str, regex[i], replacement[i]);
}

void KatexHandler::lineCleanup(std::string &line) {
  replaceAll(line, "\r", "");
  if (line[0] == '%')
    line = line.substr(1, line.size() - 2);
  line = line.substr(0, line.find('%'));
  replaceAll(line, "\\~", " ");
  for (int i = 0; i < 300; ++i) {
    size_t idx = line.find("\\>");
    if (idx != std::string::npos)
      line.replace(idx, 2, " ");
    idx = line.find('$');
    if (idx != std::string::npos)
      line.replace(idx, 1, " ");
    static std::regex labelRegex(R"(\\label\{.*?\})");
    line = std::regex_replace(line, labelRegex, "", std::regex_constants::format_first_only);
  }

  if (line.find("matrix") != std::string::npos && line.find("cases") != std::string::npos &&
      line.find("array") != std::string::npos && line.find("begin") != std::string::npos) {
    for (int i = 0; i < 300; ++i)
      line.replace(line.find("\\\\"), 2, "\\,");
  }
  line += " ";
  for (int i = 0; i < 300; ++i) {
    size_t idx = line.find("{\\rm");
    if (idx != std::string::npos)
      line.replace(idx, 4, "\\mathrm{");
    idx = line.find("{ \\rm");
    if (idx != std::string::npos)
      line.replace(idx, 4, "\\mathrm{");
    idx = line.find("\\rm{");
    if (idx != std::string::npos)
      line.replace(idx, 4, "\\mathrm{");
  }
}

HTMLRenderHandler::HTMLRenderHandler() {
  if (!wkhtmltoimage_init(true)) {
    std::cerr << "HTML Renderer Initialization Failed" << std::endl;
    exit(ENVIRONMENT_ERROR);
  }
  settings = wkhtmltoimage_create_global_settings();
  wkhtmltoimage_set_global_setting(settings, "in", "./temp.html");
  wkhtmltoimage_set_global_setting(settings, "fmt", "jpeg");
  char *width = nullptr;
  char *height = nullptr;
  wkhtmltoimage_get_global_setting(settings, "crop.width", width, 0);
  wkhtmltoimage_get_global_setting(settings, "crop.height", height, 0);
  imgSize.width = std::stoi(width);
  imgSize.height = std::stoi(height);
}

HTMLRenderHandler::~HTMLRenderHandler() {
  wkhtmltoimage_deinit();
}

cv::Mat HTMLRenderHandler::renderHTML(const std::string &html) {
  std::ofstream htmlStream("temp.html");
  htmlStream << html;
  converter = wkhtmltoimage_create_converter(settings, nullptr);
  if (!wkhtmltoimage_convert(converter)) {
    std::cerr << "Error Converting HTML to Image" << std::endl;
    exit(PROCESSING_ERROR);
  }
  const unsigned char *data = nullptr;
  unsigned long len = wkhtmltoimage_get_output(converter, &data);
  wkhtmltoimage_destroy_converter(converter);
  return {imgSize, CV_8UC3, (void *) data};
}

void OCRUtils::normalizeLatex(const std::filesystem::path &inFile, const std::filesystem::path &outFile) {
  std::ifstream texInStream(inFile);
  std::ofstream texOutStream(outFile);
  std::string line;
  static std::regex formatting(R"((hskip|hspace)(.*?)(cm|in|pt|mm|em))");
  static std::regex tabCleanup("\\t");
  static std::regex spaceCleanup(" {2,}");
  static std::regex startEndCleanup("^ +| +$");
  for (int lineCounter = 0; std::getline(texInStream, line); ++lineCounter) {
    line = std::regex_replace(line, formatting, "");
    line = std::regex_replace(line, tabCleanup, " ");
    KatexHandler::lineCleanup(line);
    KatexHandler::replaceUnsupported(line);
    KatexHandler::escape(line);
    line = std::regex_replace(line, spaceCleanup, " ");
    line = std::regex_replace(line, startEndCleanup, "");
    line = katex.normalize(line);
    texOutStream << line << std::endl;
    //std::cout << "Completed line #" << lineCounter << std::endl;
  }
}

void OCRUtils::renderLatex(const std::string &latex) {
  std::string html = katex.latexToHTML(latex);
  wkhtml.renderHTML(html);
}

std::unordered_map<std::string, int> OCRUtils::getVocab(const std::filesystem::path &dataDirectory) {
  std::filesystem::path vocabFile(dataDirectory / "vocab.txt");
  std::unordered_map<std::string, int> vocab;
  if (!exists(vocabFile)) {
    std::cerr << "Missing vocab file. Run preprocess first" << std::endl;
    exit(READ_ERROR);
  }
  std::ifstream vocabStream(vocabFile);
  std::string word;
  int count;
  while (vocabStream >> word >> count)
    vocab[word] = count;
  return vocab;
}

torch::Tensor OCRUtils::toTensor(const std::string &str) {
  size_t len = str.size();
  torch::Tensor tensor = torch::empty(static_cast<int64_t>(len + 1),
                                      torch::TensorOptions(torch::kInt8).device(torch::kCUDA));
  int i;
  for (i = 0; i < len; ++i)
    tensor[i] = str[i];
  tensor[i] = -1;
  return tensor;
}

std::vector <std::string> OCRUtils::toString(const torch::Tensor &tensor) {
  int64_t batchSize = tensor.size(0);
  std::vector <std::string> answer;
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

std::vector <cv::cuda::GpuMat> OCRUtils::toMat(const torch::Tensor &tensor, bool isNormalized) {
  std::vector <cv::cuda::GpuMat> mats;
  int64_t batchSize = tensor.size(0);
  for (int64_t i = 0; i < batchSize; ++i) {
    torch::Tensor imageTensor = tensor[i].permute({1, 2, 0});
    if (isNormalized)
      imageTensor = imageTensor.mul_(255);
    imageTensor = imageTensor.to(torch::kByte);
    torch::IntArrayRef dimensions = imageTensor.sizes();
    mats.emplace_back(cv::Size((int) dimensions[1], (int) dimensions[0]), CV_8UC1, imageTensor.data_ptr<uchar>());
  }
  return mats;
}

torch::Tensor ImageUtils::toTensor(const cv::cuda::GpuMat &matrix, torch::ScalarType size, int channels) {
  if (matrix.channels() != 1) {
    std::cerr << "Invalid number of channels" << std::endl;
    exit(INVALID_PARAMETER);
  }
  auto options = torch::TensorOptions().dtype(size).device(torch::kCUDA);
  return torch::from_blob(matrix.data, {1, static_cast<int64_t>(channels), static_cast<int64_t>(matrix.rows),
                                        static_cast<int64_t>(matrix.cols)},
                          {1, 1, (long long) (matrix.step / sizeof(size)), static_cast<int64_t>(channels)},
                          torch::Deleter(), options).contiguous();
}

void ImageUtils::addMargin(const cv::cuda::GpuMat &pixels, cv::Rect_<int> &rect, int margin) {
  rect.x = rect.x - margin < 0 ? 0 : rect.x - margin;
  rect.y = rect.y - margin < 0 ? 0 : rect.y - margin;
  rect.width = rect.x + rect.width + (2 * margin) >= pixels.cols ? pixels.cols - rect.x : rect.width + (2 * margin);
  rect.height = rect.y + rect.height + (2 * margin) >= pixels.rows ? pixels.rows - rect.y : rect.height + (2 * margin);
}

std::map <cv::Rect, Classifier::ImageType, Classifier::RectComparator>
ImageUtils::getImageBlocks(const cv::cuda::GpuMat &pixels) {
  static cv::cuda::GpuMat resized;
  constexpr float confThres = 0.25f;
  constexpr float iouThres = 0.5f;
  constexpr int maxWh = 4096;
  constexpr int maxNms = 30000;
  float scaleX = (float) pixels.cols / 640;
  float scaleY = (float) pixels.rows / 640;
  cv::cuda::resize(pixels, resized, cv::Size(640, 640), scaleX, scaleY, cv::INTER_CUBIC);
  torch::NoGradGuard no_grad;
  torch::Tensor imgTensor = toTensor(resized, torch::kByte).contiguous().to(torch::kFloat).div(255).expand(
    {1, 3, -1, -1});
  static Classifier imgClassification;
  torch::Tensor prediction = imgClassification.forward(imgTensor).to(torch::kCUDA);
  std::map <cv::Rect, Classifier::ImageType, Classifier::RectComparator> imageBlocks;

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

  std::tuple <torch::Tensor, torch::Tensor> max_tuple = torch::max(prediction.slice(1, 5, prediction.size(1)), 1, true);
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

    std::pair <cv::Rect, Classifier::ImageType> pair = std::make_pair(rect,
                                                                      static_cast<Classifier::ImageType>(output[i][5].item<int>()));
    if (rect.area())
      imageBlocks.emplace(pair);
  }
  return imageBlocks;
}

float ImageUtils::getSkewAngle(const cv::cuda::GpuMat &pixels, const Classifier::ImageType &type) {
  if (type == Classifier::ImageType::TABLE || type == Classifier::ImageType::IMAGE) {
    cv::Mat img(pixels);
    std::vector <cv::Point> points;
    cv::Mat_<uchar>::iterator it = img.begin<uchar>();
    cv::Mat_<uchar>::iterator end = img.end<uchar>();
    for (; it != end; ++it)
      if (*it)
        points.push_back(it.pos());

    cv::RotatedRect bbox = cv::minAreaRect(cv::Mat(points));
    return bbox.angle;
  } else {
    static cv::Ptr <cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(85, 255);
    static cv::Ptr <cv::cuda::HoughSegmentDetector> segmentDetector = cv::cuda::createHoughSegmentDetector(1,
                                                                                                           CV_PI / 180,
                                                                                                           0, 20, 4096,
                                                                                                           40);
    static cv::cuda::GpuMat edges;
    static cv::cuda::GpuMat lines;
    cannyDetector->detect(pixels, edges);
    segmentDetector->setMinLineLength((int) (pixels.cols * 0.33));
    segmentDetector->detect(edges, lines);
    std::vector <cv::Vec4i> linesVec;
    if (lines.cols == 0) {
      return 0.f;
    }
    linesVec.resize(lines.cols);
    cv::Mat linesCopyMat(1, lines.cols, CV_32SC4, (void *) &linesVec[0]);
    lines.download(linesCopyMat);
    float rotationAngle = 0;
    cv::Mat test(edges);
    for (unsigned i = 0; i < lines.cols; ++i) {
      line(test, cv::Point(linesVec[i][0], linesVec[i][1]), cv::Point(linesVec[i][2], linesVec[i][3]),
           cv::Scalar(255, 255, 255));
      rotationAngle += (float) (atan2((double) linesVec[i][3] - linesVec[i][1],
                                      (double) linesVec[i][2] - linesVec[i][0]));
    }
    return (float) ((rotationAngle / (float) lines.cols) * 180. / CV_PI);
  }
}

void ImageUtils::rotate(cv::cuda::GpuMat &pixels, float degree) {
  static cv::cuda::GpuMat rotated;
  cv::cuda::warpAffine(pixels, rotated, cv::getRotationMatrix2D(
                         cv::Point2f((float) ((pixels.cols - 1) / 2.0), (float) ((pixels.rows - 1) / 2.0)), degree, 1.0),
                       pixels.size());
  pixels = rotated;
}

void ImageUtils::equalize(cv::cuda::GpuMat &pixels) {
  cv::cuda::equalizeHist(pixels, pixels);
}

void ImageUtils::denoise(cv::cuda::GpuMat &pixels) {
  static cv::Ptr <cv::cuda::Filter> denoiseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1,
                                                                                     cv::getStructuringElement(
                                                                                       cv::MORPH_RECT, cv::Size(2, 1)));
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
  static cv::Ptr <cv::cuda::Filter> thresholdFilter = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
  static cv::cuda::GpuMat mean;
  thresholdFilter->apply(pixels, mean);
  cv::cuda::subtract(mean, 8, mean);
  cv::cuda::compare(pixels, mean, pixels, cv::CMP_LE);
}

GhostscriptHandler::GhostscriptHandler(std::filesystem::path outputFileDirectory,
                                       const std::variant< std

::function<
void(cv::cuda::GpuMat
&,
const std::filesystem::path &)>,
std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(
  cv::cuda::GpuMat & )>> &callback) :

callback (callback), ioContext(),
asyncPipe(ioContext), outputFileDirectory(
  std::move(outputFileDirectory)), outputFormat("^Page [0-9]+\n$"), pageNum(0) {
  if (std::holds_alternative < std::function < void(cv::cuda::GpuMat & , const std::filesystem::path &)>>(callback))
  callbackType = CallbackType::LATEX;
  else
  callbackType = CallbackType::PROCESS;
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

  process = boost::process::child("ghostscript -sDEVICE=tifflzw -sOutputFile=" + outputPrefix.generic_string() +
                                  "_page%d.tiff -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -dDITHER=300 -dUseTrimBox -dBATCH -dSAFER -dNOPAUSE " +
                                  inputFilePath.generic_string(),
                                  boost::process::std_in.close(), boost::process::std_out > asyncPipe,
                                  boost::process::std_err > stderr, ioContext);
  boost::asio::async_read_until(asyncPipe, buffer, '\n',
                                boost::bind(&GhostscriptHandler::processOutput, this, boost::asio::placeholders::error,
                                            boost::asio::placeholders::bytes_transferred));
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
  boost::asio::async_read_until(asyncPipe, buffer, '\n',
                                boost::bind(&GhostscriptHandler::processOutput, this, boost::asio::placeholders::error,
                                            boost::asio::placeholders::bytes_transferred));
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
    if (callbackType == CallbackType::LATEX) {
      std::get < std::function < void(cv::cuda::GpuMat & ,
      const std::filesystem::path &)>>(callback)(curImg,
                                                 outputFileDirectory);
    } else {
      std::filesystem::path outputFilePath = outputFileDirectory;
      outputFilePath /= fileName;
      outputFilePath += "_page";
      outputFilePath += std::to_string(pageNum);
      outputFilePath += ".png";
      std::get < std::function < std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(
        cv::cuda::GpuMat & ) >> (callback)(curImg);
      cv::cuda::resize(curImg, curImg, cv::Size(640, 640), cv::INTER_CUBIC);
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

void getPDFImages(const std::filesystem::path &inputFilePath, const std::filesystem::path &outputFileDirectory,
                  const std::variant< std

::function<
void(cv::cuda::GpuMat
&,
const std::filesystem::path &)>, std::function<std::map<cv::Rect, Classifier::ImageType, Classifier::RectComparator>(
  cv::cuda::GpuMat & )>> &callback) {
GhostscriptHandler ghostscriptHandler(outputFileDirectory, callback);
//BENCHMARK
//auto startTime = std::chrono::high_resolution_clock::now();
ghostscriptHandler.
run(inputFilePath);
//BENCHMARK
//std::cout << (std::chrono::high_resolution_clock::now() - startTime).count() << std::endl;
int returnCode = ghostscriptHandler.done();
if (returnCode != 0) {
std::cerr << "Ghostscript was unable to parse images from the pdf" <<
std::endl;
exit(returnCode);
}
}

int clamp(int n, int lower, int upper) {
  return std::max(lower, std::min(n, upper));
}
