import onnx
from onnxconverter_common import float16

# Load the FP32 model
model_fp32 = onnx.load("mobilenetv2.onnx")

# Convert weights to FP16
model_fp16 = float16.convert_float_to_float16(model_fp32)

# Save the FP16 model
onnx.save(model_fp16, "mobilenetv2_fp16.onnx")

print("✅ Quantized model saved as mobilenetv2_fp16.onnx (FP16 weights)")
