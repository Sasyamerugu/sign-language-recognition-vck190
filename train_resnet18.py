
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =====================================
# PATHS
# =====================================

TRAIN_DIR = r"C:\SignLanguage_Project\dataset_split\train"
VAL_DIR   = r"C:\SignLanguage_Project\dataset_split\val"
TEST_DIR  = r"C:\SignLanguage_Project\dataset_split\test"

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
# DATASETS
# =====================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

print("Classes:", train_dataset.classes)

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

# =====================================
# DATALOADERS
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =====================================
# MODEL
# =====================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(train_dataset.classes)
)

# =====================================
# DEVICE
# =====================================

device = torch.device("cpu")

model = model.to(device)

# =====================================
# LOSS + OPTIMIZER
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

# =====================================
# TRAINING
# =====================================

epochs = 30

best_val_acc = 0.0

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(train_loader)}"
            )

    train_acc = 100 * train_correct / train_total

    # =================================
    # VALIDATION
    # =================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100 * val_correct / val_total

    scheduler.step(val_acc)

    print(
        f"\nEpoch [{epoch+1}/{epochs}] "
        f"Loss={running_loss:.4f} "
        f"Train={train_acc:.2f}% "
        f"Val={val_acc:.2f}%"
    )

    print(
        "Current LR:",
        optimizer.param_groups[0]["lr"]
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best model saved "
            f"(Val Acc = {val_acc:.2f}%)"
        )

# =====================================
# TEST
# =====================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()

test_acc = 100 * test_correct / test_total

print("\n==========================")
print("BEST VAL ACC :", round(best_val_acc, 2))
print("TEST ACC     :", round(test_acc, 2))
print("==========================")

