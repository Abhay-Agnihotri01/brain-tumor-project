# %% [markdown]
# # Machine Learning and Deep Learning Baselines for Brain Tumor Detection
# This notebook implements non-generative ML (Random Forest, SVM) and DL (CNN, ResNet18) 
# baseline models to provide a true performance floor to compare against GAN, VAE, and Hybrid models.

# ---------------------------------------------------------
# 1. SETUP & INSTALLATION
# !pip install kaggle scikit-learn matplotlib seaborn torchvision
# ---------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. DATA LOADING (Identical to Generative Models)
# ---------------------------------------------------------
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
    
    data_transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1), # grayscale to match generators
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
    
    return train_dataset, train_loader, test_loaders

train_dataset, train_loader, test_loaders = load_brain_tumor_data()

# Utilities
def approx_dice(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    return (2. * intersection) / (np.sum(y_pred) + np.sum(y_true) + 1e-7)

# ---------------------------------------------------------
# 3. MACHINE LEARNING BASELINES (Random Forest, SVM)
# ---------------------------------------------------------
print("\n--- Training Machine Learning Models ---")
# To train ML models, we need flat numpy arrays instead of tensors.
# We'll load the full training set into memory (it's small enough for 64x64 images).
X_train_ml, y_train_ml = [], []
for img, label in DataLoader(train_dataset, batch_size=1024, shuffle=False):
    X_train_ml.append(img.view(img.size(0), -1).numpy()) # Flatten
    y_train_ml.append(label.numpy())
X_train_ml = np.vstack(X_train_ml)
y_train_ml = np.concatenate(y_train_ml)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Training Random Forest...")
start = time.time()
rf_model.fit(X_train_ml, y_train_ml)
print(f"Random Forest Training Time: {time.time() - start:.2f}s")

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
print("Training SVM...")
start = time.time()
svm_model.fit(X_train_ml, y_train_ml)
print(f"SVM Training Time: {time.time() - start:.2f}s")

def evaluate_ml_model(model, name):
    results = {}
    print(f"\nEvaluating {name}...")
    for ds_name, loader in test_loaders.items():
        X_test, y_test = [], []
        for img, label in loader:
            X_test.append(img.view(img.size(0), -1).numpy())
            y_test.append(label.numpy())
        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)
        
        start = time.time()
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        inference_time = (time.time() - start) / len(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        dice = approx_dice(y_test, y_pred)
        results[ds_name] = {'y_true': y_test, 'y_pred': y_pred, 'y_pred_prob': y_pred_prob, 'acc': acc, 'dice': dice, 'inf_time': inference_time}
    return results

rf_results = evaluate_ml_model(rf_model, "Random Forest")
svm_results = evaluate_ml_model(svm_model, "SVM")

# ---------------------------------------------------------
# 4. DEEP LEARNING BASELINES (CNN, ResNet18)
# ---------------------------------------------------------
print("\n--- Training Deep Learning Models ---")

# Simple CNN (Roughly matches our discarded standalone GAN classifier backbone for ablation)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
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

def train_dl_model(model, epochs=25, lr=2e-4):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # print(f"Epoch {epoch+1}/{epochs} Loss: {train_loss/len(train_loader):.4f}")
    return model

print("Training Simple CNN...")
cnn_model = train_dl_model(SimpleCNN(), epochs=25)

print("Training ResNet18 (modified for 1 channel grayscale)...")
resnet_model = models.resnet18(pretrained=False) # No pretraining on ImageNet as its Grayscale
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
resnet_model = train_dl_model(resnet_model, epochs=25)

def evaluate_dl_model(model, name):
    results = {}
    print(f"\nEvaluating {name}...")
    model.eval()
    with torch.no_grad():
        for ds_name, loader in test_loaders.items():
            y_true, y_pred_prob = [], []
            start = time.time()
            for imgs, labels in loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                y_true.extend(labels.numpy())
                y_pred_prob.extend(preds.cpu().numpy())
            inference_time = (time.time() - start) / len(y_true)
            
            y_true = np.array(y_true).flatten()
            y_pred_prob = np.array(y_pred_prob).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            dice = approx_dice(y_true, y_pred)
            results[ds_name] = {'y_true': y_true, 'y_pred': y_pred, 'y_pred_prob': y_pred_prob, 'acc': acc, 'dice': dice, 'inf_time': inference_time}
    return results

cnn_results = evaluate_dl_model(cnn_model, "Simple CNN")
resnet_results = evaluate_dl_model(resnet_model, "ResNet18")

def get_dl_model_size_mb(model):
    torch.save(model.state_dict(), "temp.pt")
    size_mb = os.path.getsize("temp.pt") / (1024 * 1024)
    os.remove("temp.pt")
    return size_mb

# ---------------------------------------------------------
# 5. METRICS PRINT OUT AND COMPARISON PLOTS
# ---------------------------------------------------------
all_results = {
    'Random Forest': rf_results,
    'SVM': svm_results,
    'Simple CNN': cnn_results,
    'ResNet18': resnet_results
}

print("\n\n" + "="*50)
print("FINAL BASELINE METRICS COMPARISON")
print("="*50)

for model_name, res_dict in all_results.items():
    print(f"\n--- {model_name} ---")
    for ds_name, d in res_dict.items():
        print(f"Dataset: {ds_name} | Acc: {d['acc']*100:.2f}% | Dice: {d['dice']:.2f}")
    if 'CNN' in model_name: 
        print(f"Model Size: {get_dl_model_size_mb(cnn_model):.2f} MB")
    elif 'ResNet' in model_name:
        print(f"Model Size: {get_dl_model_size_mb(resnet_model):.2f} MB")
    print(f"Inference Time: {list(res_dict.values())[0]['inf_time']:.4f} s/img")

# Accuracy Bar Plot for all 4 Baselines
ds_names = list(test_loaders.keys())
n_datasets = len(ds_names)
n_models = len(all_results)
bar_width = 0.2
x = np.arange(n_datasets)

plt.figure(figsize=(10, 6))
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
for i, (model_name, res_dict) in enumerate(all_results.items()):
    accs = [res_dict[ds]['acc'] * 100 for ds in ds_names]
    plt.bar(x + i*bar_width, accs, bar_width, label=model_name, color=colors[i])

plt.xlabel('Datasets', fontweight='bold', fontsize=12)
plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
plt.title('Baseline Models Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xticks(x + bar_width*(n_models-1)/2, ds_names)
plt.legend()
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Combined ROC against Primary Test Set
plt.figure(figsize=(8, 6))
for i, (model_name, res_dict) in enumerate(all_results.items()):
    d = res_dict['Primary Test']
    fpr, tpr, _ = roc_curve(d['y_true'], d['y_pred_prob'])
    plt.plot(fpr, tpr, lw=2, color=colors[i], label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curves (Primary Test Dataset): Baseline Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
