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

def load_brain_tumor_data(batch_size=32, img_size=128):
    download_kaggle_dataset()
    
    train_dir = 'brain-tumor-mri-dataset/Training'
    test_dir = 'brain-tumor-mri-dataset/Testing'
    extra1_dir = get_dataset_root('brain-tumor-dataset-1')
    extra2_dir = get_dataset_root('brain-tumor-dataset-2')
    
    # IMPROVEMENT 7: Stronger data augmentation for better generalisation
    data_transform_train = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.CenterCrop(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    data_transform_test = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.CenterCrop(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    def create_target_transform(dataset):
        no_classes = ['no', 'notumor', 'n', 'normal']
        # Find integer index of "no tumor" class dynamically
        no_indices = [idx for name, idx in dataset.class_to_idx.items() if name.lower() in no_classes]
        if not no_indices:
            no_indices = [0] # fallback to alphabetical first if nothing matches
        return lambda label: 0.0 if label in no_indices else 1.0

    train_dataset = ImageFolder(root=train_dir, transform=data_transform_train)
    train_dataset.target_transform = create_target_transform(train_dataset)
    
    test_dataset = ImageFolder(root=test_dir, transform=data_transform_test)
    test_dataset.target_transform = create_target_transform(test_dataset)
    
    # Check if extra datasets were found correctly
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

# IMPROVEMENT 3: Use 128x128 for finer texture detail, batch_size=32 to fit in GPU memory
train_loader, test_loaders = load_brain_tumor_data(batch_size=32, img_size=128)

# ---------------------------------------------------------
# 3. ARCHITECTURE (Hybrid) — IMPROVED
# ---------------------------------------------------------

# IMPROVEMENT 2: Upgraded classifier with stride-2 convs, 4 conv blocks,
# deeper FC head, higher dropout, and LeakyReLU throughout
class ClassifierModel(nn.Module):
    def __init__(self, latent_dim=128, num_classes=1):
        super(ClassifierModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1) # global avg pool -> (B, 256, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 + latent_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes), nn.Sigmoid()
        )
        
    def forward(self, x, latent_mu):
        x = self.features(x).view(x.size(0), -1)   # (B, 256)
        # Let classifier gradients flow to encoder to improve latent space discriminability
        fused = torch.cat((x, latent_mu), dim=1)     # (B, 256+128)
        return self.classifier(fused)

# IMPROVEMENT 1: Encoder with BatchNorm + LeakyReLU for stable, rich latent features
class EncoderModule(nn.Module):
    def __init__(self, latent_dim=128):
        super(EncoderModule, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),       # 64x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),       # 32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),      # 16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),      # 8x8
        )
        # IMPROVEMENT 8: Flat size updated for 128x128 input → 256*8*8
        self.flat_size = 256 * 8 * 8
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
    def forward(self, x):
        h = self.convs(x).view(-1, self.flat_size)
        return self.fc_mu(h), self.fc_logvar(h)

class GeneratorModule(nn.Module):
    def __init__(self, latent_dim=128):
        super(GeneratorModule, self).__init__()
        # IMPROVEMENT 8: Updated for 128x128 output (start from 8x8 and upsample 4×)
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),    # 32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),     # 64x64
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh()      # 128x128
        )
        
    def forward(self, z):
        h = self.fc(z).view(-1, 256, 8, 8)
        return self.convs(h)

class DiscriminatorModule(nn.Module):
    def __init__(self):
        super(DiscriminatorModule, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),     # 64x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),    # 32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),   # 16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),  # 8x8
        )
        # IMPROVEMENT 8: Flat size updated for 128x128 input → 256*8*8
        self.out = nn.Sequential(nn.Linear(256 * 8 * 8, 1), nn.Sigmoid())
        
    def forward(self, x):
        h = self.convs(x).view(-1, 256 * 8 * 8)
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
        self.classifier = ClassifierModel(latent_dim=latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.generator(z)
        validity = self.discriminator(recon)
        pred = self.classifier(x, mu)
        return recon, mu, logvar, validity, pred

# ---------------------------------------------------------
# 4. UTILITIES
# ---------------------------------------------------------
def approx_dice(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    return (2. * intersection) / (np.sum(y_pred) + np.sum(y_true) + 1e-7)

def evaluate_inference_time(model):
    model.eval()
    sample_input = torch.randn(1, 1, 128, 128).to(device)
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
# 5. TRAINING — IMPROVED
# ---------------------------------------------------------
# IMPROVEMENT 4: 50 epochs (was 25) for better convergence
EPOCHS = 100
hybrid_model = HybridGANVAE().to(device)
opt_enc = optim.Adam(hybrid_model.encoder.parameters(), lr=1e-4)
opt_gen = optim.Adam(hybrid_model.generator.parameters(), lr=1e-4)
opt_disc = optim.Adam(hybrid_model.discriminator.parameters(), lr=1e-4)
opt_cls = optim.Adam(hybrid_model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)

# IMPROVEMENT 4: CosineAnnealing LR scheduler for smooth convergence
sched_enc = optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=EPOCHS)
sched_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=EPOCHS)
sched_disc = optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=EPOCHS)
sched_cls = optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=EPOCHS)

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

print("Training Hybrid GAN-VAE Model (Improved)...")
hybrid_train_losses = []
hybrid_test_losses = []

for epoch in range(EPOCHS):
    hybrid_model.train()
    epoch_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        opt_disc.zero_grad()
        with torch.no_grad():
            mu, logvar = hybrid_model.encoder(imgs)
            z = reparameterize(mu, logvar)
            fakes = hybrid_model.generator(z)
        
        d_loss = (bce_loss(hybrid_model.discriminator(imgs), real_labels) + 
                  bce_loss(hybrid_model.discriminator(fakes.detach()), fake_labels)) / 2
        d_loss.backward()
        # IMPROVEMENT 6: Gradient clipping for discriminator stability
        torch.nn.utils.clip_grad_norm_(hybrid_model.discriminator.parameters(), max_norm=1.0)
        opt_disc.step()

        opt_enc.zero_grad()
        opt_gen.zero_grad()
        opt_cls.zero_grad()

        mu, logvar = hybrid_model.encoder(imgs)
        z = reparameterize(mu, logvar)
        fakes = hybrid_model.generator(z)

        rec_loss = mse_loss(fakes, imgs)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.size(0)
        g_adv_loss = bce_loss(hybrid_model.discriminator(fakes), real_labels)
        
        # Train classifier on REAL images combining with latent mu
        real_preds = hybrid_model.classifier(imgs, mu)
        cls_loss = bce_loss(real_preds, labels)

        # IMPROVEMENT 5: KL warm-up (0→0.5 over 10 epochs) + reduced cls weight (5→2)
        kl_weight = min(0.5, 0.5 * (epoch + 1) / 10)
        total_g_loss = rec_loss + kl_weight * kl_loss + g_adv_loss + 2.0 * cls_loss
        total_g_loss.backward()

        # IMPROVEMENT 6: Gradient clipping for encoder, generator, classifier
        torch.nn.utils.clip_grad_norm_(hybrid_model.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(hybrid_model.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(hybrid_model.classifier.parameters(), max_norm=1.0)

        opt_enc.step(); opt_gen.step(); opt_cls.step()
        epoch_loss += total_g_loss.item()
    
    hybrid_train_losses.append(epoch_loss / len(train_loader))
    
    # Step LR schedulers at end of each epoch
    sched_enc.step(); sched_gen.step(); sched_disc.step(); sched_cls.step()
    
    # Calculate simple Test Loss Proxy
    hybrid_model.eval()
    t_loss = 0
    with torch.no_grad():
        for imgs, labels in test_loaders['Primary Test']:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            fakes, _, _, _, real_preds = hybrid_model(imgs)
            t_loss += (mse_loss(fakes, imgs) + bce_loss(real_preds, labels)).item()
    hybrid_test_losses.append(t_loss / len(test_loaders['Primary Test']))

    print(f"Epoch {epoch+1}/{EPOCHS} - Train L: {hybrid_train_losses[-1]:.4f} | Test L: {hybrid_test_losses[-1]:.4f}")

# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------
hybrid_model.eval()

# To hold results for multiple datasets
eval_results = {}

with torch.no_grad():
    for ds_name, loader in test_loaders.items():
        y_true, y_pred_prob = [], []
        for imgs, labels in loader:
            imgs = imgs.to(device)
            
            # In a real pipeline, the classifier must judge the real test image along with its latent encoding
            mu_val, _ = hybrid_model.encoder(imgs)
            pred = hybrid_model.classifier(imgs, mu_val)
            
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
