# %% [markdown]
# # Hybrid GAN-VAE Model for Real-Time Brain Tumor Detection
# This notebook implements the final complete Hybrid GAN-VAE model 
# evaluating metrics (Accuracy, Dice, Inference Time, Model Size).
# We use `MedMNIST` (PneumoniaMNIST) as a binary classification proxy.

# ---------------------------------------------------------
# 1. SETUP & INSTALLATION
# !pip install kaggle scikit-learn matplotlib seaborn
# ---------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. DATA LOADING (KAGGLE DATASET)
# ---------------------------------------------------------
# INSTRUCTIONS for Colab:
# 1. Upload your `kaggle.json` API token to Colab.
# 2. Run the cell below to download and extract the dataset.

import subprocess

def download_kaggle_dataset():
    if not os.path.exists('brain-tumor-mri-dataset'):
        print("Wait! Let's configure kaggle credentials...")
        os.system('mkdir -p ~/.kaggle')
        os.system('cp kaggle.json ~/.kaggle/')
        os.system('chmod 600 ~/.kaggle/kaggle.json')
        print("Downloading primary dataset from Kaggle...")
        os.system('kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset')
        print("Unzipping primary dataset...")
        os.system('unzip -q brain-tumor-mri-dataset.zip -d brain-tumor-mri-dataset')
        os.system('rm brain-tumor-mri-dataset.zip')
        
        print("Downloading extra dataset 1 (Ahmed Hamada)...")
        os.system('kaggle datasets download -d ahmedhamada0/brain-tumor-detection')
        os.system('unzip -q brain-tumor-detection.zip -d brain-tumor-dataset-1')
        os.system('rm brain-tumor-detection.zip')
        
        print("Downloading extra dataset 2 (Navoneel)...")
        os.system('kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection')
        os.system('unzip -q brain-mri-images-for-brain-tumor-detection.zip -d brain-tumor-dataset-2')
        os.system('rm brain-mri-images-for-brain-tumor-detection.zip')
        
        print("Download complete.")
    else:
        print("Datasets already exist.")

from torchvision.datasets import ImageFolder

def get_dataset_root(base_dir):
    for root, dirs, files in os.walk(base_dir):
        lower_dirs = [d.lower() for d in dirs]
        if 'yes' in lower_dirs and 'no' in lower_dirs:
            return root
    return base_dir

def load_brain_tumor_data(batch_size=64, img_size=64):
    download_kaggle_dataset()
    
    train_dir = 'brain-tumor-mri-dataset/Training'
    test_dir = 'brain-tumor-mri-dataset/Testing'
    extra1_dir = get_dataset_root('brain-tumor-dataset-1')
    extra2_dir = get_dataset_root('brain-tumor-dataset-2')
    
    # Augmentation: same base as GAN + extra transforms for generalization
    data_transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    data_transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    def create_target_transform(dataset):
        no_classes = ['no', 'notumor', 'n', 'normal']
        no_indices = [idx for name, idx in dataset.class_to_idx.items() if name.lower() in no_classes]
        if not no_indices:
            no_indices = [0]
        return lambda label: 0.0 if label in no_indices else 1.0

    train_dataset = ImageFolder(root=train_dir, transform=data_transform_train)
    train_dataset.target_transform = create_target_transform(train_dataset)
    
    test_dataset = ImageFolder(root=test_dir, transform=data_transform_test)
    test_dataset.target_transform = create_target_transform(test_dataset)
    
    extra1_dataset = ImageFolder(root=extra1_dir, transform=data_transform_test) if os.path.exists(extra1_dir) else None
    if extra1_dataset: extra1_dataset.target_transform = create_target_transform(extra1_dataset)
        
    extra2_dataset = ImageFolder(root=extra2_dir, transform=data_transform_test) if os.path.exists(extra2_dir) else None
    if extra2_dataset: extra2_dataset.target_transform = create_target_transform(extra2_dataset)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    test_loaders = {'Primary Test': test_loader}
    if extra1_dataset:
        test_loaders['Extra DS 1'] = DataLoader(dataset=extra1_dataset, batch_size=batch_size, shuffle=False)
    if extra2_dataset:
        test_loaders['Extra DS 2'] = DataLoader(dataset=extra2_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loaders

train_loader, test_loaders = load_brain_tumor_data()

# ---------------------------------------------------------
# 3. ARCHITECTURE (Hybrid)
# ---------------------------------------------------------

# CLASSIFIER: MLP that takes the VAE Encoder's latent space (mu) as input
class ClassifierModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=1):
        super(ClassifierModel, self).__init__()
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

class EncoderModule(nn.Module):
    def __init__(self, latent_dim=128):
        super(EncoderModule, self).__init__()
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

class GeneratorModule(nn.Module):
    def __init__(self, latent_dim=128):
        super(GeneratorModule, self).__init__()
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

class DiscriminatorModule(nn.Module):
    def __init__(self):
        super(DiscriminatorModule, self).__init__()
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

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class HybridGANVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(HybridGANVAE, self).__init__()
        self.encoder = EncoderModule(latent_dim)
        self.generator = GeneratorModule(latent_dim)
        self.discriminator = DiscriminatorModule()
        # Classifier uses pooled spatial features from the encoder for richer representations
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassifierModel(256)

    def forward(self, x):
        mu, logvar, conv_out = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.generator(z)
        validity = self.discriminator(recon)
        
        pooled = self.pool(conv_out).view(conv_out.size(0), -1)
        pred = self.classifier(pooled)
        return recon, mu, logvar, validity, pred

# ---------------------------------------------------------
# 4. UTILITIES
# ---------------------------------------------------------
def approx_dice(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    return (2. * intersection) / (np.sum(y_pred) + np.sum(y_true) + 1e-7)

def evaluate_inference_time(model):
    model.eval()
    sample_input = torch.randn(1, 1, 64, 64).to(device)
    times = []
    with torch.no_grad():
        for _ in range(20):
            start = time.time()
            _ = model(sample_input)
            times.append(time.time() - start)
    return np.mean(times)

def get_model_size_mb(model):
    torch.save(model.state_dict(), "temp_hybrid.pt")
    size_mb = os.path.getsize("temp_hybrid.pt") / (1024 * 1024)
    os.remove("temp_hybrid.pt")
    return size_mb

# ---------------------------------------------------------
# 5. TRAINING — mirrors GAN's training setup exactly
# ---------------------------------------------------------
EPOCHS = 100
hybrid_model = HybridGANVAE().to(device)

# Optimizer setup mirrors GAN exactly:
# GAN uses: opt_gan_g = Adam(gen+cls, lr=2e-4, wd=1e-5)
# We use:   opt_g_cls  = Adam(enc+gen+cls, lr=2e-4, wd=1e-5)
# The key: classifier only gets gradients from cls_loss (not from gen/rec/kl)
opt_g_cls = optim.Adam(
    list(hybrid_model.encoder.parameters()) +
    list(hybrid_model.generator.parameters()) +
    list(hybrid_model.classifier.parameters()),
    lr=2e-4, weight_decay=1e-5
)
opt_disc = optim.Adam(hybrid_model.discriminator.parameters(), lr=1e-4)

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

print("Training Hybrid GAN-VAE Model...")
hybrid_train_losses = []
hybrid_test_losses = []

for epoch in range(EPOCHS):
    hybrid_model.train()
    epoch_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # --- Train Discriminator (same as GAN) ---
        opt_disc.zero_grad()
        with torch.no_grad():
            mu, logvar, _ = hybrid_model.encoder(imgs)
            z = reparameterize(mu, logvar)
            fakes = hybrid_model.generator(z)
        
        d_loss = (bce_loss(hybrid_model.discriminator(imgs), real_labels) + 
                  bce_loss(hybrid_model.discriminator(fakes.detach()), fake_labels)) / 2
        d_loss.backward()
        opt_disc.step()

        # --- Train Encoder + Generator + Classifier ---
        opt_g_cls.zero_grad()

        recon, mu, logvar, validity, real_preds = hybrid_model(imgs)

        # VAE losses (affect encoder+generator)
        rec_loss = mse_loss(recon, imgs)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
        
        # Generator wants to fool discriminator
        g_adv_loss = bce_loss(hybrid_model.discriminator(recon), real_labels)
        
        # Classifier loss
        cls_loss = bce_loss(real_preds, labels)

        # Total loss — gradients flow properly through their components.
        # We heavily weight `cls_loss` (5.0) so the encoder prioritizes tumor detection features,
        # while keeping the generative scaling balanced.
        total_loss = rec_loss + 0.05 * kl_loss + 0.5 * g_adv_loss + 5.0 * cls_loss
        total_loss.backward()
        opt_g_cls.step()

        epoch_loss += total_loss.item()
    
    hybrid_train_losses.append(epoch_loss / len(train_loader))
    
    # Test Loss
    hybrid_model.eval()
    t_loss = 0
    with torch.no_grad():
        for imgs, labels in test_loaders['Primary Test']:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            recon, mu, logvar, _, preds = hybrid_model(imgs)
            t_loss += (mse_loss(recon, imgs) + bce_loss(preds, labels)).item()
    hybrid_test_losses.append(t_loss / len(test_loaders['Primary Test']))

    print(f"Epoch {epoch+1}/{EPOCHS} - Train L: {hybrid_train_losses[-1]:.4f} | Test L: {hybrid_test_losses[-1]:.4f}")

# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------
hybrid_model.eval()

eval_results = {}

with torch.no_grad():
    for ds_name, loader in test_loaders.items():
        y_true, y_pred_prob = [], []
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, _, _, _, pred = hybrid_model(imgs)
            
            y_true.extend(labels.numpy())
            y_pred_prob.extend(pred.cpu().numpy())
            
        y_true = np.array(y_true).flatten()
        y_pred_prob = np.array(y_pred_prob).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        eval_results[ds_name] = {'y_true': y_true, 'y_pred_prob': y_pred_prob, 'y_pred': y_pred}

# --- Plot 1: Loss Curve ---
plt.figure(figsize=(10, 5))
plt.plot(hybrid_train_losses, label='Training Loss', color='#4A90E2', lw=2)
plt.plot(hybrid_test_losses, label='Testing Loss', color='#F5A623', lw=2)
plt.title('Training and Testing Loss Curves (Hybrid Model)', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- Plot 2: Confusion Matrices ---
fig, axes = plt.subplots(1, len(test_loaders), figsize=(6 * len(test_loaders), 5))
if len(test_loaders) == 1: axes = [axes]
for ax, (ds_name, results) in zip(axes, eval_results.items()):
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'], ax=ax)
    ax.set_title(f'Confusion Matrix: {ds_name}')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
plt.tight_layout()
plt.show()

# --- Plot 4: Combined ROC Curve ---
plt.figure(figsize=(8, 6))
colors = ['darkorange', '#2ECC71', '#9B59B6']
for (ds_name, results), color in zip(eval_results.items(), colors):
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_prob'])
    plt.plot(fpr, tpr, lw=2, color=color, label=f'{ds_name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curves: Hybrid Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# --- Plot 5: Accuracy Comparison ---
accuracies = [accuracy_score(res['y_true'], res['y_pred']) * 100 for res in eval_results.values()]
plt.figure(figsize=(8, 5))
bar_colors = ['#4A90E2', '#2ECC71', '#9B59B6'][:len(accuracies)]
plt.bar(list(eval_results.keys()), accuracies, color=bar_colors)
plt.title('Accuracy Comparison Across Datasets: Hybrid Model', fontsize=14)
plt.ylabel('Accuracy (%)')
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
plt.ylim(0, 100)
plt.show()

print("\n--- HYBRID MODEL FINAL METRICS ---")
for ds_name, results in eval_results.items():
    print(f"\nDataset: {ds_name}")
    print(f"Accuracy:         {accuracy_score(results['y_true'], results['y_pred'])*100:.2f}%")
    print(f"Dice Coefficient: {approx_dice(results['y_true'], results['y_pred']):.2f}")

print(f"\nModel Size:       {get_model_size_mb(hybrid_model):.2f} MB")
print(f"Inference Time:   {evaluate_inference_time(hybrid_model):.4f} s/img")

# ---------------------------------------------------------
# 7. SAVE AND DOWNLOAD MODEL (FOR COLAB)
# ---------------------------------------------------------
print("\nSaving final model...")
MODEL_PATH = "hybrid_model_final.pth"
torch.save(hybrid_model.state_dict(), MODEL_PATH)

try:
    from google.colab import files
    print(f"Downloading {MODEL_PATH} to your local machine...")
    files.download(MODEL_PATH)
except ImportError:
    print(f"Model saved locally as {MODEL_PATH}. (Not running in Colab, so automated download skipped.)")
