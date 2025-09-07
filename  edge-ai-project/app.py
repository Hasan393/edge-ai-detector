import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model (FP16 quantized)
sess = ort.InferenceSession("mobilenetv2_fp16.onnx")

# Download ImageNet labels
# You can also keep a local copy if needed
import urllib.request
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(labels_url).read().decode("utf-8").splitlines()

def classify(image):
    # Convert numpy array (from Gradio) to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preprocess
    img = image.convert("RGB").resize((224, 224))
    x = np.array(img).astype("float16") / 255.0   # ✅ float16 for FP16 model
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)   # Add batch dim

    # Run inference
    ort_inputs = {sess.get_inputs()[0].name: x}
    ort_outs = sess.run(None, ort_inputs)

    # Get prediction
    pred_class = int(np.argmax(ort_outs[0]))
    label = imagenet_classes[pred_class] if pred_class < len(imagenet_classes) else f"Class {pred_class}"

    return f"Predicted class: {label}"

# Gradio UI
iface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs="text",
    title="Edge AI - MobileNetV2 Classifier (FP16 Optimized)"
)

iface.launch()
