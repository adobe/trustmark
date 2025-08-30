#include "onnx_session.h"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace TrustMark {

// Static member initialization
static Ort::Env globalEnv_;
static bool globalEnvInitialized_ = false;

// Constructor
ONNXRuntimeSession::ONNXRuntimeSession(const std::string& modelPath, const std::string& sessionName)
    : modelPath_(modelPath)
    , sessionName_(sessionName)
    , session_(nullptr)
{
    // Initialize global environment if not already done
    if (!globalEnvInitialized_) {
        try {
            globalEnv_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, sessionName_.c_str());
            globalEnvInitialized_ = true;
        } catch (const Ort::Exception& e) {
            setLastError("Failed to initialize ONNX Runtime environment: " + std::string(e.what()));
            return;
        }
    }

    // Set session options
    sessionOptions_.SetIntraOpNumThreads(8);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Initialize session
    if (!initializeSession()) {
        setLastError("Failed to initialize ONNX Runtime session: " + getLastError());
        return;
    }
}

// Destructor
ONNXRuntimeSession::~ONNXRuntimeSession() = default;

// Move constructor
ONNXRuntimeSession::ONNXRuntimeSession(ONNXRuntimeSession&& other) noexcept
    : modelPath_(std::move(other.modelPath_))
    , sessionName_(std::move(other.sessionName_))
    , env_(std::move(other.env_))
    , sessionOptions_(std::move(other.sessionOptions_))
    , session_(std::move(other.session_))
    , inputNames_(std::move(other.inputNames_))
    , outputNames_(std::move(other.outputNames_))
    , inputShapes_(std::move(other.inputShapes_))
    , outputShapes_(std::move(other.outputShapes_))
    , lastError_(std::move(other.lastError_))
{
}

// Move assignment
ONNXRuntimeSession& ONNXRuntimeSession::operator=(ONNXRuntimeSession&& other) noexcept {
    if (this != &other) {
        modelPath_ = std::move(other.modelPath_);
        sessionName_ = std::move(other.sessionName_);
        env_ = std::move(other.env_);
        sessionOptions_ = std::move(other.sessionOptions_);
        session_ = std::move(other.session_);
        inputNames_ = std::move(other.inputNames_);
        outputNames_ = std::move(other.outputNames_);
        inputShapes_ = std::move(other.inputShapes_);
        outputShapes_ = std::move(other.outputShapes_);
        lastError_ = std::move(other.lastError_);
    }
    return *this;
}

// Initialize session
bool ONNXRuntimeSession::initializeSession() {
    try {
        // Check if model file exists
        std::ifstream file(modelPath_);
        if (!file.good()) {
            setLastError("Model file not found: " + modelPath_);
            return false;
        }
        file.close();

        // Create session
        session_ = std::make_unique<Ort::Session>(globalEnv_, modelPath_.c_str(), sessionOptions_);

        // Get input/output information
        Ort::AllocatorWithDefaultOptions allocator;

        // Input names
        size_t numInputs = session_->GetInputCount();
        inputNames_.reserve(numInputs);
        for (size_t i = 0; i < numInputs; ++i) {
            auto inputName = session_->GetInputNameAllocated(i, allocator);
            inputNames_.push_back(inputName.get());
        }

        // Output names
        size_t numOutputs = session_->GetOutputCount();
        outputNames_.reserve(numOutputs);
        for (size_t i = 0; i < numOutputs; ++i) {
            auto outputName = session_->GetOutputNameAllocated(i, allocator);
            outputNames_.push_back(outputName.get());
        }

        // Input shapes
        inputShapes_.reserve(numInputs);
        for (size_t i = 0; i < numInputs; ++i) {
            auto typeInfo = session_->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            inputShapes_.push_back(tensorInfo.GetShape());
        }

        // Output shapes
        outputShapes_.reserve(numOutputs);
        for (size_t i = 0; i < numOutputs; ++i) {
            auto typeInfo = session_->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            outputShapes_.push_back(tensorInfo.GetShape());
        }

        return true;

    } catch (const Ort::Exception& e) {
        setLastError("ONNX Runtime error: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        setLastError("Unexpected error: " + std::string(e.what()));
        return false;
    }
}

// Run inference
std::vector<Ort::Value> ONNXRuntimeSession::run(const std::vector<Ort::Value>& inputs) {
    try {
        if (!session_) {
            setLastError("Session not initialized");
            return {};
        }

        if (inputs.size() != inputNames_.size()) {
            setLastError("Number of inputs (" + std::to_string(inputs.size()) +
                        ") doesn't match expected (" + std::to_string(inputNames_.size()) + ")");
            return {};
        }

        // Convert string names to const char* arrays
        std::vector<const char*> inputNamePtrs;
        std::vector<const char*> outputNamePtrs;

        for (const auto& name : inputNames_) {
            inputNamePtrs.push_back(name.c_str());
        }
        for (const auto& name : outputNames_) {
            outputNamePtrs.push_back(name.c_str());
        }

        // Debug: Print input/output names
        std::cout << "Model expects " << inputNames_.size() << " inputs: ";
        for (const auto& name : inputNames_) {
            std::cout << "'" << name << "' ";
        }
        std::cout << std::endl;

        std::cout << "Model expects " << outputNames_.size() << " outputs: ";
        for (const auto& name : outputNames_) {
            std::cout << "'" << name << "' ";
        }
        std::cout << std::endl;

        std::cout << "Providing " << inputs.size() << " inputs" << std::endl;

        // Run inference
        std::cout << "Running ONNX inference..." << std::endl;

        // Validate input tensors before running
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i].IsTensor()) {
                throw std::runtime_error("Input " + std::to_string(i) + " is not a tensor");
            }
            auto tensorInfo = inputs[i].GetTensorTypeAndShapeInfo();
            std::cout << "Input " << i << " shape: ";
            for (auto dim : tensorInfo.GetShape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                   inputNamePtrs.data(),
                                   inputs.data(),
                                   inputs.size(),
                                   outputNamePtrs.data(),
                                   outputNamePtrs.size());

        std::cout << "ONNX inference completed. Got " << outputs.size() << " outputs" << std::endl;
        return outputs;

    } catch (const Ort::Exception& e) {
        setLastError("ONNX Runtime inference error: " + std::string(e.what()));
        return {};
    } catch (const std::exception& e) {
        setLastError("Unexpected inference error: " + std::string(e.what()));
        return {};
    }
}

// Get input names
std::vector<std::string> ONNXRuntimeSession::getInputNames() const {
    return inputNames_;
}

// Get output names
std::vector<std::string> ONNXRuntimeSession::getOutputNames() const {
    return outputNames_;
}

// Get input shapes
std::vector<std::vector<int64_t>> ONNXRuntimeSession::getInputShapes() const {
    return inputShapes_;
}

// Get output shapes
std::vector<std::vector<int64_t>> ONNXRuntimeSession::getOutputShapes() const {
    return outputShapes_;
}

// Set last error
void ONNXRuntimeSession::setLastError(const std::string& error) const {
    lastError_ = error;
}

// ONNX utility functions
namespace onnx_utils {

// Create input tensor from OpenCV Mat
Ort::Value createInputTensor(const cv::Mat& image,
                            const std::string& inputName,
                            const std::vector<int64_t>& expectedShape) {
    try {
        // Image should already be in RGB format from preprocessing
        cv::Mat processedImage = image;

        // Reshape image to match expected shape
        if (expectedShape.size() == 4) {
            // Expected: (batch, channels, height, width)
            int batch = static_cast<int>(expectedShape[0]);
            int channels = static_cast<int>(expectedShape[1]);
            int height = static_cast<int>(expectedShape[2]);
            int width = static_cast<int>(expectedShape[3]);

            // Resize if needed
            if (processedImage.rows != height || processedImage.cols != width) {
                cv::resize(processedImage, processedImage, cv::Size(width, height));
            }

            // Convert to float
            cv::Mat floatImage;
            processedImage.convertTo(floatImage, CV_32F);

            // Calculate total elements needed
            size_t totalElements = batch * channels * height * width;

            // Verify we have the right number of channels
            if (floatImage.channels() != channels) {
                throw std::runtime_error("Image channels (" + std::to_string(floatImage.channels()) +
                                        ") don't match expected (" + std::to_string(channels) + ")");
            }

            // Create a heap-allocated buffer that will be managed by ONNX Runtime
            float* tensorData = new float[totalElements];

            // Convert HWC to CHW format
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int hwc_idx = h * width * channels + w * channels + c;
                        int chw_idx = c * height * width + h * width + w;
                        tensorData[chw_idx] = floatImage.ptr<float>()[hwc_idx];
                    }
                }
            }

            // Create tensor with the allocated memory
            auto tensor = Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
                                                        tensorData,
                                                        totalElements,
                                                        expectedShape.data(),
                                                        expectedShape.size());

            return tensor;
        }

        throw std::runtime_error("Unsupported tensor shape");

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create input tensor: " + std::string(e.what()));
    }
}

// Create input tensor from vector
Ort::Value createInputTensor(const std::vector<float>& data,
                            const std::string& inputName,
                            const std::vector<int64_t>& expectedShape) {
    // Calculate total size for any number of dimensions
    size_t totalSize = 1;
    for (int64_t dim : expectedShape) {
        totalSize *= dim;
    }

    // Allocate memory using ONNX Runtime's allocator
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create tensor using the input data directly
    return Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(data.data()), totalSize, expectedShape.data(), expectedShape.size());
}

// Extract output tensor to OpenCV Mat
cv::Mat extractOutputTensor(const Ort::Value& output,
                           const std::vector<int64_t>& expectedShape) {
    try {
        // Get tensor data
        float* data = const_cast<Ort::Value&>(output).GetTensorMutableData<float>();
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();

        // Calculate dimensions
        int batch = static_cast<int>(shape[0]);
        int channels = static_cast<int>(shape[1]);
        int height = static_cast<int>(shape[2]);
        int width = static_cast<int>(shape[3]);

        // Create OpenCV Mat
        cv::Mat outputMat(height, width, CV_32FC(channels));

        // Copy data
        size_t dataSize = height * width * channels;
        std::memcpy(outputMat.ptr<float>(), data, dataSize * sizeof(float));

        return outputMat;

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to extract output tensor: " + std::string(e.what()));
    }
}

// Extract output tensor to vector
std::vector<float> extractOutputTensor(const Ort::Value& output) {
    try {
        // Get tensor data
        float* data = const_cast<Ort::Value&>(output).GetTensorMutableData<float>();
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();

        // Calculate total elements
        size_t totalElements = 1;
        for (int64_t dim : shape) {
            totalElements *= dim;
        }

        // Copy data to vector
        std::vector<float> result(data, data + totalElements);
        return result;

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to extract output tensor: " + std::string(e.what()));
    }
}

// Create memory info
Ort::MemoryInfo createMemoryInfo() {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

} // namespace onnx_utils

} // namespace TrustMark
