#include <iostream>
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.ort>" << std::endl;
        return 1;
    }

    std::cout << "Testing model loading only (no inference)" << std::endl;
    std::cout << "Model: " << argv[1] << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_INFO, "LoadTest");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);

    std::cout << "Loading model..." << std::endl;
    Ort::Session session(env, argv[1], opts);
    std::cout << "✓ Model loaded successfully!" << std::endl;

    // Get model info
    Ort::AllocatorWithDefaultOptions alloc;
    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    std::cout << "  Inputs: " << num_inputs << std::endl;
    std::cout << "  Outputs: " << num_outputs << std::endl;

    for (size_t i = 0; i < num_inputs; i++) {
        auto name = session.GetInputNameAllocated(i, alloc);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "  Input " << i << ": " << name.get() << " [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "✓ Model structure looks good!" << std::endl;
    return 0;
}
