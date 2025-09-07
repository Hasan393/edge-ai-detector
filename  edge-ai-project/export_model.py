import torch
import torchvision.models as models

# Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Dummy input for shape (batch=1, channels=3, 224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13
)

print("✅ Exported model to mobilenetv2.onnx")
