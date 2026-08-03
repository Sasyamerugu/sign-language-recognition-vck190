import os
import cv2
import torch
import torch.nn as nn
from torchvision import models
from pytorch_nndct.apis import torch_quantizer

# =====================================
# PATHS
# =====================================

MODEL_PATH = "models/isl_mobilenetv2_v2_best.pth"
CALIB_DIR = "calib_images"

# =====================================
# MODEL
# =====================================

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    26
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

model.eval()

# =====================================
# PREPROCESS
# =====================================

def preprocess(path):

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    img = img.astype("float32") / 255.0

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    for c in range(3):
        img[:,:,c] = (img[:,:,c] - mean[c]) / std[c]

    img = img.transpose(2,0,1)

    return torch.tensor(img).unsqueeze(0)

# =====================================
# QUANTIZER
# =====================================

dummy_input = torch.randn(1,3,224,224)

quantizer = torch_quantizer(
    quant_mode="calib",
    module=model,
    input_args=(dummy_input,)
)

quant_model = quantizer.quant_model

# =====================================
# CALIBRATION
# =====================================

files = []

for f in os.listdir(CALIB_DIR):

    if f.lower().endswith(
        (".jpg",".jpeg",".png")
    ):
        files.append(
            os.path.join(CALIB_DIR,f)
        )

print("Calibration Images:", len(files))

with torch.no_grad():

    for idx, path in enumerate(files):

        x = preprocess(path)

        quant_model(x)

        if idx % 20 == 0:
            print(
                f"Processed {idx+1}/{len(files)}"
            )

quantizer.export_quant_config()

print("Calibration Complete")