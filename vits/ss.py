import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import time
import requests
import json
import matplotlib.pyplot as plt
from PIL import Image
from timm import create_model
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize  # Fix for EigenCAM heatmap resizing

# ------------------------------
# 1. Define Available Models & Performance Metrics
# ------------------------------
MODEL_OPTIONS = {
    "DeiT": {"name": "deit_base_patch16_224", "params": 86, "flops": 17.5, "top1_acc": 81.8},
    "Swin Transformer": {"name": "swin_base_patch4_window7_224", "params": 88, "flops": 15.4, "top1_acc": 83.5}
}

# ------------------------------
# 2. Streamlit UI Components
# ------------------------------
st.title("Efficiency-360: Vision Transformer Model Evaluation")
st.write("Select a model, upload an image, and analyze classification performance.")

# Model Selection Dropdown
selected_model = st.selectbox("Choose a Vision Transformer:", list(MODEL_OPTIONS.keys()))

# Display Model Performance Metrics
st.write(f"📊 **Model Metrics for {selected_model}:**")
st.write(f"🔹 **Parameters:** {MODEL_OPTIONS[selected_model]['params']}M")
st.write(f"🔹 **FLOPs:** {MODEL_OPTIONS[selected_model]['flops']}G")
st.write(f"🔹 **Top-1 Accuracy:** {MODEL_OPTIONS[selected_model]['top1_acc']}%")

# ------------------------------
# 3. Load Selected Model
# ------------------------------
@st.cache_resource
def load_model(model_name):
    model = create_model(MODEL_OPTIONS[model_name]["name"], pretrained=True)
    model.eval()
    return model

model = load_model(selected_model)

# ------------------------------
# 4. Define Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------
# 5. Image Upload Section
# ------------------------------
uploaded_file = st.file_uploader("Upload an Image for Classification", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0)

    # ------------------------------
    # 6. Perform Inference & Measure Speed
    # ------------------------------
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
    inference_time = time.time() - start_time

    # Get Predicted Class
    predicted_class = torch.argmax(output, dim=1).item()

    # Load ImageNet Labels
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    labels = requests.get(labels_url).json()
    class_name = labels[str(predicted_class)][1]

    # Display Prediction
    st.write(f"🎯 **Predicted Class:** {class_name} (ID: {predicted_class})")
    st.write(f"⚡ **Inference Time:** {inference_time:.4f} seconds")

    # ------------------------------
    # 7. EigenCAM Visualization for ViTs (Fixed Target Layer)
    # ------------------------------
    st.write("🔍 **EigenCAM Visualization:**")

    # Select the correct target layer for Grad-CAM
    def get_target_layer(model, model_name):
        if "deit" in model_name:
            return model.blocks[-1].norm1  # DeiT Target Layer
        elif "swin" in model_name:
            return model.layers[-1].blocks[-1].norm1  # Swin Transformer Target Layer
        else:
            raise ValueError("Unsupported model for EigenCAM")

    target_layer = get_target_layer(model, MODEL_OPTIONS[selected_model]["name"])

    # Use EigenCAM for ViTs
    cam = EigenCAM(model=model, target_layers=[target_layer])

    # Compute heatmap
    grayscale_cam = cam(input_tensor=image_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # Convert image to NumPy array and normalize
    rgb_img = np.array(image) / 255.0  

    # 🔹 **Fix: Resize Grad-CAM heatmap to match original image size**
    grayscale_cam_resized = resize(grayscale_cam, (rgb_img.shape[0], rgb_img.shape[1]), mode="constant")

    # Apply EigenCAM heatmap on image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)

    st.image(cam_image, caption="EigenCAM Heatmap (Fixed for ViTs)", use_column_width=True)
