import sys
sys.path.insert(0, 'onnxruntime-wasi/tools/python')

from onnx import load
model_simple = load('onnxruntime-wasi/simple_model.ort')
model_trustmark = load('models/encoder_P.with_runtime_opt.ort')

print("Simple model (WORKS):")
print(f"  Nodes: {len(model_simple.graph.node)}")
print(f"  Initializers: {len(model_simple.graph.initializer)}")

print("\nTrustMark model (FAILS):")
print(f"  Nodes: {len(model_trustmark.graph.node)}")
print(f"  Initializers: {len(model_trustmark.graph.initializer)}")

print("\nMaybe the issue is model complexity, not specific operators...")
