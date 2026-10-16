// Copyright 2026 Adobe. Licensed under the MIT license.

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

cv::Mat load_image(const fs::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty() || image.depth() != CV_8U || (image.channels() != 3 && image.channels() != 4)) {
        throw std::runtime_error("expected an 8-bit RGB or RGBA image: " + path.string());
    }
    return image;
}

cv::Mat crop_image(const cv::Mat& image, bool square) {
    if (!square && image.cols <= 2 * image.rows && image.rows <= 2 * image.cols) {
        return image;
    }
    int edge = std::min(image.cols, image.rows);
    return image(cv::Rect((image.cols - edge) / 2, (image.rows - edge) / 2, edge, edge));
}

Tensor image_tensor(const cv::Mat& original, int size, bool square) {
    cv::Mat rgb;
    auto cropped = crop_image(original, square);
    cv::cvtColor(cropped, rgb, cropped.channels() == 4 ? cv::COLOR_BGRA2RGB : cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
    Tensor tensor{{1, 3, size, size}, std::vector<float>(3 * size * size)};
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            auto pixel = resized.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                tensor.data[c * size * size + y * size + x] = pixel[c] / 127.5f - 1.0f;
            }
        }
    }
    return tensor;
}

void embed_image(const cv::Mat& original, const Tensor& input, const Tensor& encoded,
                 const fs::path& path, float strength, bool square, bool soften) {
    cv::Mat residual(256, 256, CV_32FC3);
    std::array<double, 3> means{};
    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 256; ++x) {
            auto& pixel = residual.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) {
                int index = c * 256 * 256 + y * 256 + x;
                pixel[c] = std::clamp(encoded.data[index] - input.data[index], -0.2f, 0.2f)
                           * strength / 2.0f;
                means[c] += pixel[c];
            }
        }
    }
    if (soften) {
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                if (x < 2 || y < 2 || x >= 254 || y >= 254) {
                    auto& pixel = residual.at<cv::Vec3f>(y, x);
                    for (int c = 0; c < 3; ++c) pixel[c] = static_cast<float>(means[c] / 65536);
                }
            }
        }
    }
    cv::Mat full(original.rows, original.cols, CV_32FC3);
    if (square) {
        full.setTo(cv::Scalar(means[0] / 65536, means[1] / 65536, means[2] / 65536));
        int edge = std::min(original.cols, original.rows);
        cv::Mat scaled;
        cv::resize(residual, scaled, cv::Size(edge, edge), 0, 0, cv::INTER_LINEAR);
        scaled.copyTo(full(cv::Rect((original.cols - edge) / 2, (original.rows - edge) / 2, edge, edge)));
    } else {
        cv::resize(residual, full, original.size(), 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat output = original.clone();
    for (int y = 0; y < original.rows; ++y) {
        for (int x = 0; x < original.cols; ++x) {
            const auto delta = full.at<cv::Vec3f>(y, x);
            auto* pixel = output.ptr<uint8_t>(y) + x * output.channels();
            for (int c = 0; c < 3; ++c) {
                pixel[2 - c] = cv::saturate_cast<uint8_t>(pixel[2 - c] + delta[c] * 255.0f);
            }
        }
    }
    if (!cv::imwrite(path.string(), output)) throw std::runtime_error("failed to save image: " + path.string());
}

std::string name_at(const Ort::Session& session, size_t index, bool input) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = input ? session.GetInputNameAllocated(index, allocator)
                      : session.GetOutputNameAllocated(index, allocator);
    return name.get();
}

std::vector<int64_t> shape_at(const Ort::Session& session, size_t index, bool input) {
    auto info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    if (info.GetONNXType() != ONNX_TYPE_TENSOR ||
        info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("only float32 tensor inputs and outputs are supported");
    }
    return info.GetTensorTypeAndShapeInfo().GetShape();
}

void check_shape(const Ort::Session& session, const std::string& name,
                 const std::vector<int64_t>& actual, bool input) {
    const size_t count = input ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < count; ++i) {
        if (name_at(session, i, input) != name) {
            continue;
        }
        const auto expected = shape_at(session, i, input);
        if (expected.size() != actual.size()) {
            throw std::runtime_error(name + " has an unexpected number of dimensions");
        }
        for (size_t j = 0; j < actual.size(); ++j) {
            if (expected[j] > 0 && expected[j] != actual[j]) {
                throw std::runtime_error(name + " has an unexpected dimension at index " + std::to_string(j));
            }
        }
        return;
    }
    throw std::runtime_error("model has no tensor named " + name);
}

void inspect(const Ort::Session& session) {
    for (bool input : {true, false}) {
        const size_t count = input ? session.GetInputCount() : session.GetOutputCount();
        for (size_t i = 0; i < count; ++i) {
            std::cout << (input ? "input " : "output ") << name_at(session, i, input) << " float32 [";
            auto shape = shape_at(session, i, input);
            for (size_t j = 0; j < shape.size(); ++j) {
                if (j) {
                    std::cout << ", ";
                }
                std::cout << shape[j];
            }
            std::cout << "]\n";
        }
    }
}

std::vector<float> read_floats(const fs::path& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open input: " + path.string());
    }
    if (count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float) ||
        file.tellg() != static_cast<std::streamoff>(count * sizeof(float))) {
        throw std::runtime_error("input must contain exactly " + std::to_string(count) + " float32 values: " + path.string());
    }
    std::vector<float> data(count);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!file) {
        throw std::runtime_error("failed to read input: " + path.string());
    }
    return data;
}

void write_floats(const fs::path& path, const Tensor& tensor) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot open output: " + path.string());
    }
    file.write(reinterpret_cast<const char*>(tensor.data.data()),
               static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("failed to write output: " + path.string());
    }
}

Tensor run(Ort::Session& session, std::vector<std::pair<std::string, Tensor>> inputs,
           const std::string& output_name) {
    if (session.GetInputCount() != inputs.size() || session.GetOutputCount() != 1) {
        throw std::runtime_error("unexpected model input/output count");
    }
    if (name_at(session, 0, false) != output_name) {
        throw std::runtime_error("model has no output named " + output_name);
    }
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char*> names;
    std::vector<Ort::Value> values;
    for (auto& [name, tensor] : inputs) {
        check_shape(session, name, tensor.shape, true);
        names.push_back(name.c_str());
        values.push_back(Ort::Value::CreateTensor<float>(memory, tensor.data.data(),
                            tensor.data.size(), tensor.shape.data(), tensor.shape.size()));
    }
    const char* output = output_name.c_str();
    auto results = session.Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(), &output, 1);
    auto info = results[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("model returned a non-float32 tensor");
    }
    auto shape = info.GetShape();
    check_shape(session, output_name, shape, false);
    const auto* data = results[0].GetTensorData<float>();
    return {std::move(shape), std::vector<float>(data, data + info.GetElementCount())};
}

int usage() {
    std::cerr << "Usage:\n"
              << "  trustmark-runner inspect MODEL.onnx\n"
              << "  trustmark-runner decode MODEL.onnx IMAGE.f32\n"
              << "  trustmark-runner encode MODEL.onnx IMAGE.f32 BITS100 OUTPUT.f32\n"
              << "  trustmark-runner decode-image MODELS_DIR VARIANT INPUT.png\n"
              << "  trustmark-runner encode-image MODELS_DIR VARIANT INPUT.png OUTPUT.png PAYLOAD [BCH_5] [STRENGTH]\n"
              << "IMAGE.f32 is normalized [-1,1] float32 RGB in NCHW order.\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        return usage();
    }
    try {
        const std::string command = argv[1];
        if ((command == "inspect" && argc != 3) ||
            (command == "decode" && argc != 4) ||
            (command == "encode" && argc != 6) ||
            (command == "decode-image" && argc != 5) ||
            (command == "encode-image" && (argc < 7 || argc > 9)) ||
            (command != "inspect" && command != "decode" && command != "encode" &&
             command != "decode-image" && command != "encode-image")) {
            return usage();
        }

        const bool image_mode = command == "decode-image" || command == "encode-image";
        const bool decoding = command == "decode" || command == "decode-image";
        float strength = 0.95f;
        if (command == "encode-image" && argc == 9) {
            strength = std::stof(argv[8]);
            if (!std::isfinite(strength) || strength < 0 || strength > 1) {
                throw std::runtime_error("strength must be between 0 and 1");
            }
        }
        char variant = 0;
        if (image_mode) {
            if (std::string(argv[3]).size() != 1 || std::string("BCPQ").find(argv[3][0]) == std::string::npos) {
                throw std::runtime_error("variant must be B, C, P, or Q");
            }
            variant = argv[3][0];
        }
        const auto model_path = image_mode
            ? fs::path(argv[2]) / (std::string(decoding ? "decoder_" : "encoder_") + variant + ".onnx")
            : fs::path(argv[2]);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "trustmark");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        const auto path = model_path.wstring();
#else
        const auto path = model_path.string();
#endif
        Ort::Session session(env, path.c_str(), options);
        if (command == "inspect") {
            inspect(session);
            return 0;
        }

        int64_t size = 256;
        if (decoding) {
            if (session.GetInputCount() != 1 || name_at(session, 0, true) != "image") {
                throw std::runtime_error("decoder must have one input named image");
            }
            const auto shape = shape_at(session, 0, true);
            if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 ||
                shape[2] != shape[3] || (shape[2] != 224 && shape[2] != 256)) {
                throw std::runtime_error("decoder input must have shape [1,3,224,224] or [1,3,256,256]");
            }
            size = shape[2];
        }
        cv::Mat original;
        if (image_mode) original = load_image(argv[4]);
        const bool square = image_mode && (variant == 'P' || original.cols > 2 * original.rows ||
                                            original.rows > 2 * original.cols);
        Tensor image = image_mode ? image_tensor(original, static_cast<int>(size), square)
            : Tensor{{1, 3, size, size}, read_floats(argv[3], 3 * size * size)};
        if (decoding) {
            auto output = run(session, {{"image", std::move(image)}}, "output");
            if (output.shape != std::vector<int64_t>{1, 100}) {
                throw std::runtime_error("decoder output must have shape [1,100]");
            }
            std::string packet;
            for (float value : output.data) {
                packet += value < 0 ? '0' : '1';
            }
            if (image_mode) {
                trustmark::Decoded decoded;
                if (trustmark::decode_bits(packet, decoded)) {
                    std::cout << "Found watermark (" << decoded.version << "): " << decoded.bits << '\n';
                } else {
                    std::cout << "Corrupt or missing watermark\n";
                }
            } else {
                std::cout << packet << '\n';
            }
        } else {
            if (image_mode && size != 256) {
                throw std::runtime_error("encoder input must be 256x256");
            }
            const std::string bits = image_mode
                ? trustmark::encode_bits(argv[6], argc >= 8 ? argv[7] : "BCH_5") : argv[4];
            if (bits.size() != 100 || bits.find_first_not_of("01") != std::string::npos) {
                throw std::runtime_error("BITS100 must be exactly 100 binary digits, including ECC and schema bits");
            }
            Tensor payload{{1, 100}, {}};
            for (char bit : bits) {
                payload.data.push_back(bit == '1' ? 1.0f : 0.0f);
            }
            auto output = run(session, {{"onnx::Concat_0", image},
                                        {"onnx::Gemm_1", std::move(payload)}}, "image");
            if (output.shape != std::vector<int64_t>{1, 3, 256, 256}) {
                throw std::runtime_error("encoder output must have shape [1,3,256,256]");
            }
            if (image_mode) {
                if (variant == 'P') strength *= 1.25f;
                embed_image(original, image, output, argv[5], strength, square, square);
            } else {
                write_floats(argv[5], output);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "trustmark-runner: " << error.what() << '\n';
        return 1;
    }
}
