import onnx
from onnxconverter_common import float16

# Load the model
model_fp32 = onnx.load("mobilenetv2.onnx")

# Convert to float16
model_fp16 = float16.convert_float_to_float16(model_fp32)

# Save
onnx.save(model_fp16, "mobilenetv2_fp16.onnx")

print("✅ Quantized model saved as mobilenetv2_fp16.onnx (FP16 weights)")
