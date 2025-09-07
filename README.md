# 🧠 Edge AI + Model Optimization (MobileNetV2 ONNX)

Deploying a **lightweight AI model** on the edge (mobile/PC) for **real-time inference**, with **quantization & optimization**.
This project demonstrates **model compression, ONNX export, FP16 quantization, and edge deployment** using **Gradio**.

---

## 🚀 Features

* ✅ Train & export a PyTorch model to **ONNX**
* ✅ Apply **FP16 quantization** for faster inference
* ✅ Run on **ONNX Runtime** optimized for edge devices
* ✅ Web demo using **Gradio** (upload an image → get predictions)
* ✅ Model size reduction and latency benchmark

---

## 📂 Project Structure

```
edge-ai-project/
│── export_model.py       # Export MobileNetV2 → ONNX
│── quantize_model.py     # Apply FP16 quantization
│── benchmark.py          # Benchmark latency & size
│── app.py                # Gradio demo app
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
│── models/               # Saved ONNX models (fp32 / fp16)
│── venv/                 # Virtual environment(optinal)
```

---

## ⚙️ Installation

1. Clone the repo:

```bash
git clone https://github.com/your-username/edge-ai-project.git
cd edge-ai-project
```

2. **(Recommended)** Create a Python virtual environment **outside** the project folder:

```bash
# From the parent directory (e.g., /workspaces/edge-ai-detector)
python -m venv venv
source venv/bin/activate   # Linux/Mac
```

> The `venv` folder is **optional but recommended**. Keeping it outside the main project directory helps keep your codebase clean and avoids accidental commits of environment files.

3. Install dependencies:

```bash
pip install -r edge-ai-project/requirements.txt
```

---

## 🏗️ Steps to Run

### 1. Export Model → ONNX

```bash
python export_model.py
```

This creates `mobilenetv2.onnx`.

### 2. Quantize Model (FP16)

```bash
python quantize_model.py
```

This creates `mobilenetv2_fp16.onnx`.

### 3. Benchmark Model

```bash
python benchmark.py
```

Check size & inference speed.

### 4. Run Web App

```bash
python app.py
```

Open in browser → `http://127.0.0.1:7860`
Upload an image → Get predictions 🎉

---

## 📊 Example Results

| Model            | Size    | Avg Inference Time |
| ---------------- | ------- | ------------------ |
| FP32 (original)  | \~13 MB | \~8.5 ms           |
| FP16 (quantized) | \~6 MB  | \~4.2 ms           |

---

## ⚡ Tech Used

* [PyTorch](https://pytorch.org/) → Base model (MobileNetV2)
* [ONNX](https://onnx.ai/) → Model export
* [ONNX Runtime](https://onnxruntime.ai/) → Edge inference
* [Gradio](https://gradio.app/) → Interactive web UI

---

## 🌍 Why It Matters

This project shows you **understand deployment constraints**:

* Small model sizes → runs on mobile/IoT devices
* Optimized inference → real-time AI at the edge
* Interactive demo → production-ready showcase

---

## 🔮 Next Steps

* Add **better datasets** (e.g., DeepFashion for clothing)
* Deploy on **mobile app (Android/iOS)** with ONNX Runtime
* Support **TensorRT** for NVIDIA Jetson devices

---

✨ Built for learning Edge AI + Model Optimization 🚀

