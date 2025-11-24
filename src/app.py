import os
import time
import csv
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
import gradio as gr
from prometheus_client import Counter, Histogram, start_http_server
from torchvision import transforms
from model import ResNet50_CBAM

# --------------------------
# PROMETHEUS METRICS
# --------------------------
start_http_server(8000)  # Exposes /metrics

PREDICTION_COUNTER = Counter(
    "pancreas_prediction_total",
    "Total predictions made by class",
    ["class"]
)

FEEDBACK_COUNTER = Counter(
    "pancreas_feedback_total",
    "User feedback on predictions",
    ["result"]
)

INFERENCE_LATENCY = Histogram(
    "pancreas_inference_latency_seconds",
    "Model inference latency"
)

# --------------------------
# LOAD MODEL
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50_CBAM(num_classes=2).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# --------------------------
# TRANSFORMS
# --------------------------
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------------
# GRADCAM IMPLEMENTATION
# --------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # capture activations
        target_layer.register_forward_hook(self.forward_hook)
        # capture gradients
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        # forward pass
        out = self.model(x)
        class_idx = out.argmax().item()

        # backward pass
        self.model.zero_grad()
        out[0, class_idx].backward(retain_graph=True)

        # Grad-CAM
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (grads * self.activations).sum(dim=1).squeeze()

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam



target_layer = model.cbam4
cam = GradCAM(model, target_layer)


def generate_gradcam(img):
    # Convert uploaded image
    pil_img = Image.fromarray(img).convert("L")
    orig_np = np.array(pil_img)

    # Preprocess
    x = test_transforms(pil_img).unsqueeze(0).to(device)

    # Generate CAM
    cam_map = cam.generate(x)

    # Colorize CAM
    heat = cv2.applyColorMap(np.uint8(cam_map * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # Normalize
    heat = heat.astype(np.float32) / 255

    # Resize original
    resized = cv2.resize(orig_np, (224, 224))
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    resized = resized.astype(np.float32) / 255

    # Blend CAM + Image
    overlay = 0.4 * heat + 0.6 * resized
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)


# --------------------------
# FEEDBACK CSV SETUP
# --------------------------
CSV_FILE = "feedback_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "prediction", "prob_tumor", "prob_normal", "feedback"])

def save_feedback(pred, p_tumor, p_normal, feedback):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), pred, p_tumor, p_normal, feedback])

# --------------------------
# PREDICT FUNCTION (Gradio)
# --------------------------
def predict_ct(img):
    start = time.time()

    pil_img = Image.fromarray(img).convert("L")
    t = test_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1)[0]
    
    prob_normal = float(probs[0])
    prob_tumor = float(probs[1])
    
    pred = "tumor" if prob_tumor > prob_normal else "normal"

    # update metrics
    PREDICTION_COUNTER.labels(pred).inc()
    INFERENCE_LATENCY.observe(time.time() - start)

    grad_img = generate_gradcam(img)

    return pred, {"normal": prob_normal, "tumor": prob_tumor}, grad_img

def feedback_fn(feedback, pred, probs):
    FEEDBACK_COUNTER.labels(feedback).inc()
    save_feedback(pred, probs["tumor"], probs["normal"], feedback)
    return "Feedback saved!"

# --------------------------
# GRADIO UI 
# --------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Pancreas Tumor Classifier with Grad-CAM + Feedback")

    with gr.Row():
        input_img = gr.Image(type="numpy", label="Upload CT Image")

    with gr.Row():
        pred_label = gr.Textbox(label="Prediction")
        pred_probs = gr.Label(label="Probabilities")
        cam_output = gr.Image(label="Grad-CAM Visualization")

    predict_btn = gr.Button("Run Prediction")
    predict_btn.click(
        predict_ct,
        inputs=[input_img],
        outputs=[pred_label, pred_probs, cam_output]
    )

    feedback = gr.Radio(["correct", "incorrect"], label="Was the prediction correct?")
    feedback_btn = gr.Button("Submit Feedback")
    fb_status = gr.Textbox(label="Status")

    feedback_btn.click(
        feedback_fn,
        inputs=[feedback, pred_label, pred_probs],
        outputs=[fb_status]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
