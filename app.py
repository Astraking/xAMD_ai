import os
import gdown
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define Google Drive file IDs
retinal_model_id = '1nlcoXT4u06jSGVFDKZU5G4gbY0IJlWxr'
amd_model_id = '1D1WZXSRvFJbarBhn1WGq01Xqd11jEUvw'

# Define model paths
retinal_model_path = 'retinal_model.pth'
amd_model_path = 'amd_model.pth'

# Download models if they don't exist
if not os.path.exists(retinal_model_path):
    gdown.download(f'https://drive.google.com/uc?id={retinal_model_id}', retinal_model_path, quiet=False)

if not os.path.exists(amd_model_path):
    gdown.download(f'https://drive.google.com/uc?id={amd_model_id}', amd_model_path, quiet=False)

# Load models
retinal_model = torch.load(retinal_model_path, map_location=torch.device('cpu'))
retinal_model.eval()

amd_model = torch.load(amd_model_path, map_location=torch.device('cpu'))
amd_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit interface
st.title("AMD Screening App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Check if the image is retinal
    with torch.no_grad():
        retinal_output = retinal_model(image)
        retinal_pred = torch.argmax(retinal_output, dim=1).item()

    if retinal_pred == 1:
        st.write("This is a retinal image. Checking for AMD...")
        with torch.no_grad():
            amd_output = amd_model(image)
            amd_pred = torch.argmax(amd_output, dim=1).item()

        if amd_pred == 1:
            st.write("AMD detected.")
        else:
            st.write("No AMD detected.")
    else:
        st.write("This is not a retinal image.")

