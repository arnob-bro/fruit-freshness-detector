# Image-Based Fruit Freshness Detection System

## Full Implementation Plan + Code Structure

---

## 🔹 Project Architecture

```
fruit_freshness_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   │   ├── fresh/
│   │   └── rotten/
│   ├── val/
│   │   ├── fresh/
│   │   └── rotten/
│   └── test/
│       ├── fresh/
│       └── rotten/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── models/
│   ├── cnn_scratch.py
│   ├── resnet_transfer.py
│   └── mobilenet_transfer.py
│
├── utils/
│   ├── dataloader.py
│   ├── augmentations.py
│   ├── metrics.py
│   └── visualization.py
│
├── training/
│   ├── train.py
│   └── config.yaml
│
├── evaluation/
│   ├── evaluate.py
│   └── confusion_matrix.py
│
├── explainability/
│   └── gradcam.py
│
├── deployment/
│   ├── app.py
│   └── api.py
│
├── requirements.txt
└── README.md
```

---

# 🔹 Step-by-Step Implementation

---

## STEP 1 — Environment Setup

```bash
pip install torch torchvision tensorflow opencv-python matplotlib seaborn scikit-learn streamlit grad-cam
```

---

## STEP 2 — Data Loader (utils/dataloader.py)

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_data = datasets.ImageFolder('data/train', transform=transform)
val_data = datasets.ImageFolder('data/val', transform=transform)

def get_loaders(batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
```

---

## STEP 3 — CNN From Scratch (models/cnn_scratch.py)

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*26*26,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,1), nn.Sigmoid()
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

---

## STEP 4 — Transfer Learning (models/resnet_transfer.py)

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,1),
    nn.Sigmoid()
)
```

---

## STEP 5 — Training Loop (training/train.py)

```python
import torch
from utils.dataloader import get_loaders
from models.resnet_transfer import model

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loader, val_loader = get_loaders()

for epoch in range(30):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} complete")
```

---

## STEP 6 — Evaluation (evaluation/evaluate.py)

```python
from sklearn.metrics import classification_report, confusion_matrix

# predictions → y_pred
# labels → y_true

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

---

## STEP 7 — GradCAM (explainability/gradcam.py)

```python
from pytorch_grad_cam import GradCAM

cam = GradCAM(model=model, target_layers=[model.layer4])
```

---

## STEP 8 — Deployment (deployment/app.py)

```python
import streamlit as st
from PIL import Image
import torch

st.title("Fruit Freshness Detector")
file = st.file_uploader("Upload Fruit Image")

if file:
    img = Image.open(file)
    st.image(img)
    # preprocess → model → prediction
    st.write("Prediction: Fresh")
```

---

# 🔹 Training Strategy

```
Train CNN Scratch
Train ResNet50 Transfer
Train MobileNetV2 Transfer
Compare Results
Select Best Model
```

---

# 🔹 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

---

# 🔹 Research-Level Enhancements

- Class imbalance handling
- Data noise robustness
- Early spoilage classification
- Multi-class extension (fresh, semi-rotten, rotten)
- Mobile deployment (TensorFlow Lite)

---

# 🔹 Presentation Flow

1. Problem Statement
2. Dataset
3. Architecture
4. Models
5. Training
6. Evaluation
7. Explainability
8. Deployment Demo
9. Conclusion

---

# 🔹 Viva Questions Prep

- Why CNN?
- Why Transfer Learning?
- Why not classical ML?
- Dataset bias?
- Overfitting control?
- Real-world challenges?
- False classification risk?

---

# ✅ Outcome

This system becomes:
- Research-grade
- Industry-aligned
- Production scalable
- CV-based ML system
- Academic project
- Real deployment capable

---

🚀 This is a complete production + research ML system blueprint.

