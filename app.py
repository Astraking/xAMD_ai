import streamlit as st
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import gdown

# Set page config
st.set_page_config(page_title="AMD Screening App", layout="wide")

# Define model classes
class BinaryClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(BinaryClassifier, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        x = self.sigmoid(x)
        return x

class AMDModel(nn.Module):
    def __init__(self, num_classes=2):
        super(AMDModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        return x

@st.cache_resource
def load_model(model_class, path):
    try:
        model = model_class()
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def download_model(model_id, model_path):
    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_path}..."):
            gdown.download(f'https://drive.google.com/uc?id={model_id}', model_path, quiet=False)

# Define model paths
retinal_model_path = 'binary_classifier.pth'
amd_model_path = 'amd_model.pth'

# Download models
download_model('1nlcoXT4u06jSGVFDKZU5G4gbY0IJlWxr', retinal_model_path)
download_model('1D1WZXSRvFJbarBhn1WGq01Xqd11jEUvw', amd_model_path)

# Load models
retinal_model = load_model(BinaryClassifier, retinal_model_path)
amd_model = load_model(AMDModel, amd_model_path)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.feature_map = None

        target_layer.register_forward_hook(self.save_feature_map)
        target_layer.register_backward_hook(self.save_gradient)

    def save_feature_map(self, module, input, output):
        self.feature_map = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        pred = output.argmax(dim=1)

        pred_class = output[0, pred]
        pred_class.backward()

        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
        feature_map = self.feature_map[0]

        for i in range(len(pooled_gradients)):
            feature_map[i, :, :] *= pooled_gradients[i]

        heatmap = feature_map.detach().numpy()
        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        return heatmap

# Function to generate Grad-CAM visualization
def generate_gradcam_image(image, model, target_layer):
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(image)

    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_image = image.squeeze().permute(1, 2, 0).numpy()
    original_image = np.clip(original_image, 0, 1)
    original_image = np.uint8(255 * original_image)

    superimposed_image = heatmap * 0.4 + original_image
    superimposed_image = np.clip(superimposed_image, 0, 255)
    
    return superimposed_image

@st.cache_data
def preprocess_image(_image):
    # Convert to RGB if it's not already
    image = _image.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast by 50%
    
    # Auto-equalize histogram
    image = ImageOps.equalize(image)
    
    return image

def process_image(image, retinal_model, amd_model):
    preprocessed_image = preprocess_image(image)
    image_tensor = transform(preprocessed_image).unsqueeze(0)
    
    with torch.no_grad():
        retinal_output = retinal_model(image_tensor)
        retinal_pred = torch.round(retinal_output).item()
        retinal_conf = retinal_output.item()

    if retinal_pred == 0:
        with torch.no_grad():
            amd_output = amd_model(image_tensor)
            amd_pred = torch.argmax(amd_output, dim=1).item()
            amd_conf = torch.softmax(amd_output, dim=1)[0, amd_pred].item()

        return retinal_pred, retinal_conf, amd_pred, amd_conf, image_tensor
    
    return retinal_pred, retinal_conf, None, None, image_tensor

# Streamlit interface
st.title("AMD Screening App")

# Navigation
menu = ["Home", "Upload Image", "About AMD"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.header("Welcome to the AMD Screening App")
    st.write("""
    This application allows you to upload retinal images to screen for Age-related Macular Degeneration (AMD).
    Please use the sidebar to navigate through the app.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Macula.svg/1200px-Macula.svg.png", width=300, caption="Illustration of the macula")

elif choice == "Upload Image":
    st.header("Upload Retinal Image")
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                with st.spinner("Processing image..."):
                    retinal_pred, retinal_conf, amd_pred, amd_conf, image_tensor = process_image(image, retinal_model, amd_model)

                if retinal_pred == 0:
                    st.write(f"This is a retinal image (Confidence: {retinal_conf:.2f})")
                    if amd_pred == 1:
                        st.write(f"AMD detected (Confidence: {amd_conf:.2f})")
                        # Generate Grad-CAM visualization
                        target_layer = amd_model.features[8]
                        gradcam_image = generate_gradcam_image(image_tensor, amd_model, target_layer)
                        st.image(gradcam_image, caption='Grad-CAM Visualization', use_column_width=True)
                    else:
                        st.write(f"No AMD detected (Confidence: {amd_conf:.2f})")
                else:
                    st.write(f"This is not a retinal image (Confidence: {retinal_conf:.2f})")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

elif choice == "About AMD":
    st.header("About Age-related Macular Degeneration (AMD)")
    st.write("""
    Age-related macular degeneration (AMD) is an eye disease that can blur your central vision. It happens when aging causes damage to the macula — the part of the eye that controls sharp, straight-ahead vision. 
    The macula is part of the retina (the light-sensitive tissue at the back of the eye). AMD is a common condition — it's a leading cause of vision loss for older adults.
    
    ### Types of AMD
    1. **Dry AMD**: This is the most common type. It happens when parts of the macula get thinner with age and tiny clumps of protein called drusen grow. You slowly lose central vision. There's no way to treat dry AMD yet.
    2. **Wet AMD**: This type is less common but more serious. It happens when new, abnormal blood vessels grow under the retina. These vessels may leak blood or other fluids, causing scarring of the macula. You lose vision faster with wet AMD than with dry AMD.

    ### Risk Factors
    - Age (50 and older)
    - Family history and genetics
    - Race (more common in Caucasians)
    - Smoking
    - Cardiovascular disease

    ### Symptoms
    - Blurry or fuzzy vision
    - Straight lines appear wavy
    - Difficulty seeing in low light
    - Difficulty recognizing faces

    ### Prevention and Management
    While there's no cure for AMD, some lifestyle choices can help reduce the risk:
    - Avoid smoking
    - Exercise regularly
    - Maintain normal blood pressure and cholesterol levels
    - Eat a healthy diet rich in green leafy vegetables and fish
    - Protect your eyes from UV light

    For more information, visit [NEI AMD Information](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration).
    """)
    st.image("https://www.nei.nih.gov/sites/default/files/styles/featured_image/public/2019-06/macula_cross_section_v3_500px.jpg", width=300, caption="Cross-section of the macula")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by [Your Name/Organization]")
