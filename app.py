import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time

# =======================================================
# 1. MODEL ARCHITECTURES
# =======================================================

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# --- Hybrid Model Components ---
class ClassifierModel_Hybrid(nn.Module):
    def __init__(self, input_dim=128, num_classes=1):
        super(ClassifierModel_Hybrid, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(x)

class EncoderModule_Hybrid(nn.Module):
    def __init__(self, latent_dim=128):
        super(EncoderModule_Hybrid, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.flat_size = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
    def forward(self, x):
        conv_out = self.convs(x)
        h = conv_out.view(-1, self.flat_size)
        return self.fc_mu(h), self.fc_logvar(h), conv_out

class GeneratorModule_Hybrid(nn.Module):
    def __init__(self, latent_dim=128):
        super(GeneratorModule_Hybrid, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh()
        )
        
    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.convs(h)

class DiscriminatorModule_Hybrid(nn.Module):
    def __init__(self):
        super(DiscriminatorModule_Hybrid, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(nn.Linear(256 * 4 * 4, 1), nn.Sigmoid())
        
    def forward(self, x):
        h = self.convs(x).view(-1, 256 * 4 * 4)
        return self.out(h)

class HybridGANVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(HybridGANVAE, self).__init__()
        self.encoder = EncoderModule_Hybrid(latent_dim)
        self.generator = GeneratorModule_Hybrid(latent_dim)
        self.discriminator = DiscriminatorModule_Hybrid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassifierModel_Hybrid(256)

    def forward(self, x):
        mu, logvar, conv_out = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.generator(z)
        validity = self.discriminator(recon)
        pooled = self.pool(conv_out).view(conv_out.size(0), -1)
        pred = self.classifier(pooled)
        return recon, mu, logvar, validity, pred

# --- VAE Components ---
class ClassifierModel_VAE(nn.Module):
    def __init__(self, num_classes=1):
        super(ClassifierModel_VAE, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6), 
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Dropout(0.6), 
            nn.Linear(64, num_classes), nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

class EncoderModule_VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(EncoderModule_VAE, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.flat_size = 128 * 8 * 8
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
    def forward(self, x):
        h = self.convs(x).view(-1, self.flat_size)
        return self.fc_mu(h), self.fc_logvar(h)

class DecoderModule_VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(DecoderModule_VAE, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh()
        )
        
    def forward(self, z):
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.convs(h)

class VAEOnly(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAEOnly, self).__init__()
        self.encoder = EncoderModule_VAE(latent_dim)
        self.decoder = DecoderModule_VAE(latent_dim)
        self.classifier = ClassifierModel_VAE()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        pred = self.classifier(recon)
        return recon, mu, logvar, pred

# --- GAN Components ---
class ClassifierModel_GAN(nn.Module):
    def __init__(self, num_classes=1):
        super(ClassifierModel_GAN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), 
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Dropout(0.4), 
            nn.Linear(64, num_classes), nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

class GeneratorModule_GAN(nn.Module):
    def __init__(self, latent_dim=128):
        super(GeneratorModule_GAN, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh()
        )
        
    def forward(self, z):
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.convs(h)

class DiscriminatorModule_GAN(nn.Module):
    def __init__(self):
        super(DiscriminatorModule_GAN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(nn.Linear(128 * 8 * 8, 1), nn.Sigmoid())
        
    def forward(self, x):
        h = self.convs(x).view(-1, 128 * 8 * 8)
        return self.out(h)

class GANOnly(nn.Module):
    def __init__(self, latent_dim=128):
        super(GANOnly, self).__init__()
        self.generator = GeneratorModule_GAN(latent_dim)
        self.discriminator = DiscriminatorModule_GAN()
        self.classifier = ClassifierModel_GAN()

    def forward(self, z):
        recon = self.generator(z)
        pred = self.classifier(recon)
        validity = self.discriminator(recon)
        return recon, pred, validity


# =======================================================
# 2. APP LOGIC
# =======================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transforms (same as test data transforms in Colab)
img_size = 64
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@st.cache_resource
def load_models():
    # Model configs
    models = {
        "Hybrid": {
            "model_class": HybridGANVAE,
            "path": r"hybrid_model_final.pth",
            # "path": r"C:\Users\antis\Downloads\hybrid_model_final.pth",
            "instance": None
        },
        "VAE": {
            "model_class": VAEOnly,
            "path": r"vae_model_final.pth",
            # "path": r"C:\Users\antis\Downloads\vae_model_final.pth",
            "instance": None
        },
        "GAN": {
            "model_class": GANOnly,
            "path": r"gan_model_final.pth",
            # "path": r"C:\Users\antis\Downloads\gan_model_final.pth",
            "instance": None
        }
    }
    
    # Load each model if its file exists
    for name, config in models.items():
        if os.path.exists(config["path"]):
            model = config["model_class"]().to(device)
            # Use map_location for device compatibility
            model.load_state_dict(torch.load(config["path"], map_location=device))
            model.eval()
            models[name]["instance"] = model
    return models

st.title("🧠 Brain Tumor Detection using Generative Deep Learning")
st.markdown("Test the **Hybrid GAN-VAE**, **Standalone VAE**, and **Standalone GAN** models directly from this interface using the downloaded `.pth` files.")

# Load models
models = load_models()

# Sidebar: Check model loaded status
st.sidebar.title("Model Check")
for name, config in models.items():
    if config["instance"] is not None:
        st.sidebar.success(f"✅ {name} loaded")
    else:
        st.sidebar.error(f"❌ {name} not found")

# Main interface
model_choice = st.selectbox("Select Model to Evaluate", ["Hybrid", "VAE", "GAN"])
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    
    if st.button("Predict"):
        selected_model = models[model_choice]["instance"]
        
        if selected_model is None:
            st.error(f"{model_choice} model weights not found. Please ensure it is present in C:/Users/antis/Downloads/.")
        else:
            with st.spinner(f"Running inference with {model_choice} model..."):
                # Preprocess
                try:
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    start_time = time.time()
                    
                    # Inference
                    with torch.no_grad():
                        if model_choice == "Hybrid":
                            _, _, _, _, pred = selected_model(input_tensor)
                        elif model_choice == "VAE":
                            pred = selected_model.classifier(input_tensor)
                        elif model_choice == "GAN":
                            pred = selected_model.classifier(input_tensor)
                            
                    end_time = time.time()
                    
                    # Postprocess
                    probability = pred.item()
                    prediction = "Tumor Detected" if probability > 0.5 else "No Tumor"
                    confidence = probability if probability > 0.5 else 1 - probability
                    
                    # Display results
                    with col2:
                        st.subheader("Results")
                        if probability > 0.5:
                            st.markdown(f"**Prediction:** <span style='color:red; font-size:24px;'>{prediction}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**Prediction:** <span style='color:green; font-size:24px;'>{prediction}</span>", unsafe_allow_html=True)
                        
                        st.write(f"**Confidence:** {confidence * 100:.2f}%")
                        st.write(f"**Probability of Tumor:** {probability * 100:.2f}%")
                        st.write(f"**Inference Time:** {(end_time - start_time) * 1000:.2f} ms")
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
