import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Classes
classes = [chr(ord('a') + i) for i in range(26)]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Device
device = torch.device("cpu")

# Model
model = models.mobilenet_v2()

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    26
)

model.load_state_dict(
    torch.load(
        r"C:\SignLanguage_Project\models\isl_mobilenetv2.pth",
        map_location=device
    )
)

model.eval()

# Load image
image = Image.open(
    r"C:\SignLanguage_Project\test_dataset\a\A480.jpg."
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

# Prediction
with torch.no_grad():

    output = model(image)

    _, predicted = torch.max(output, 1)

print(
    "Prediction:",
    classes[predicted.item()]
)