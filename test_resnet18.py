import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =====================================
# PATHS
# =====================================

TEST_DIR = r"C:\SignLanguage_Project\dataset_split\test"
MODEL_PATH = r"C:\SignLanguage_Project\models\isl_resnet18.pth"

# =====================================
# TRANSFORMS
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# DATASET
# =====================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Classes:", len(test_dataset.classes))
print("Test images:", len(test_dataset))

# =====================================
# MODEL
# =====================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(test_dataset.classes)
)

# =====================================
# LOAD CHECKPOINT
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

# =====================================
# TEST
# =====================================

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100.0 * correct / total

print("\n======================")
print("TEST ACCURACY:", round(accuracy, 2), "%")
print("======================")