#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>

void test_model(const char* model_path, const std::vector<std::vector<float>>& inputs,
                const std::vector<std::vector<int64_t>>& input_shapes) {
    std::cout << "\nTesting: " << model_path << std::endl;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Test");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        Ort::Session session(env, model_path, opts);

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;

        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < inputs.size(); i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(name.get());

            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(inputs[i].data()), inputs[i].size(),
                input_shapes[i].data(), input_shapes[i].size()));
        }

        auto out_name = session.GetOutputNameAllocated(0, allocator);
        const char* output_names[] = {out_name.get()};

        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                   input_tensors.data(), inputs.size(), output_names, 1);

        const float* output = outputs[0].GetTensorData<float>();
        auto shape_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();
        size_t total = 1;
        for (auto dim : shape) total *= dim;

        std::cout << "  Output (first 10): ";
        for (size_t i = 0; i < std::min(total, size_t(10)); i++) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;

    } catch (const Ort::Exception& e) {
        std::cout << "  ERROR: " << e.what() << std::endl;
    }
}

int main() {
    // Test Sigmoid
    {
        std::vector<float> input(10);
        for (int i = 0; i < 10; i++) input[i] = -1.0f + i * 0.2f;
        test_model("models/test_sigmoid.onnx", {input}, {{1, 10}});
    }

    // Test Mul
    {
        std::vector<float> input1(10, 2.0f);
        std::vector<float> input2(10, 0.5f);
        test_model("models/test_mul.onnx", {input1, input2}, {{1, 10}, {1, 10}});
    }

    // Test Conv
    {
        std::vector<float> input(1 * 3 * 4 * 4, 1.0f);
        test_model("models/test_conv.onnx", {input}, {{1, 3, 4, 4}});
    }

    // Test Resize
    {
        std::vector<float> input(1 * 3 * 4 * 4);
        for (size_t i = 0; i < input.size(); i++) input[i] = float(i % 16);
        test_model("models/test_resize.onnx", {input}, {{1, 3, 4, 4}});
    }

    return 0;
}
