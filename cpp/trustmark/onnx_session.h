#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace TrustMark {

class ONNXRuntimeSession {
public:
    // Constructor
    ONNXRuntimeSession(const std::string& modelPath, 
                      const std::string& sessionName = "default");
    
    // Destructor
    ~ONNXRuntimeSession();
    
    // Disable copy
    ONNXRuntimeSession(const ONNXRuntimeSession&) = delete;
    ONNXRuntimeSession& operator=(const ONNXRuntimeSession&) = delete;
    
    // Move constructor and assignment
    ONNXRuntimeSession(ONNXRuntimeSession&&) noexcept;
    ONNXRuntimeSession& operator=(ONNXRuntimeSession&&) noexcept;
    
    // Main inference method
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& inputs);
    
    // Utility methods
    bool isInitialized() const { return session_ != nullptr; }
    std::string getModelPath() const { return modelPath_; }
    std::string getSessionName() const { return sessionName_; }
    
    // Get input/output info
    std::vector<std::string> getInputNames() const;
    std::vector<std::string> getOutputNames() const;
    std::vector<std::vector<int64_t>> getInputShapes() const;
    std::vector<std::vector<int64_t>> getOutputShapes() const;
    
    // Error handling
    std::string getLastError() const { return lastError_; }
    void clearLastError() { lastError_.clear(); }

private:
    // Private helper methods
    bool initializeSession();
    void setLastError(const std::string& error) const;
    
    // Member variables
    std::string modelPath_;
    std::string sessionName_;
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    
    // Input/output metadata
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::vector<std::vector<int64_t>> outputShapes_;
    
    // Error state
    mutable std::string lastError_;

};

// Utility functions for ONNX Runtime
namespace onnx_utils {
    // Create input tensor from OpenCV Mat
    Ort::Value createInputTensor(const cv::Mat& image, 
                                const std::string& inputName,
                                const std::vector<int64_t>& expectedShape);
    
    // Create input tensor from vector
    Ort::Value createInputTensor(const std::vector<float>& data,
                                const std::string& inputName,
                                const std::vector<int64_t>& expectedShape);
    
    // Extract output tensor to OpenCV Mat
    cv::Mat extractOutputTensor(const Ort::Value& output, 
                               const std::vector<int64_t>& expectedShape);
    
    // Extract output tensor to vector
    std::vector<float> extractOutputTensor(const Ort::Value& output);
    
    // Memory management
    Ort::MemoryInfo createMemoryInfo();
}

} // namespace TrustMark
