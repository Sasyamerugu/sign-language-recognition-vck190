# Sign Language Recognition on AMD Versal VCK190

An end-to-end Indian Sign Language (ISL) alphabet recognition system built using ResNet18 and deployed on the AMD Versal™ VCK190 platform through the Vitis AI workflow.

The project demonstrates the complete deployment pipeline—from transfer learning and model optimization to INT8 quantization, DPU compilation, and hardware inference.

## Documentation

A detailed project presentation is available in

docs/SignLanguageModel.pdf# sign-language-recognition-vck190


---

## Features

- Fine-tuned a pretrained ResNet18 model for Indian Sign Language alphabet recognition.
- Applied transfer learning by freezing early layers and training higher-level feature representations.
- Performed INT8 quantization using the AMD Vitis AI Quantizer.
- Compiled the quantized model for execution on the AMD Versal VCK190 DPU.
- Achieved high inference accuracy with minimal degradation after quantization.
- Supports image-based prediction and webcam-based real-time inference.

---

## Hardware

- AMD Versal™ VCK190 Evaluation Kit

---

## Software & Frameworks

- Python
- PyTorch
- OpenCV
- NumPy
- AMD Vitis AI
- Vitis AI Quantizer
- Vitis AI Compiler

---

## Model Architecture

- Backbone: ResNet18
- Pretrained Weights: ImageNet
- Number of Classes: 26 (Indian Sign Language Alphabets)

### Training Strategy

- Layers 1–3 frozen
- Layer 4 and final fully connected layer fine-tuned
- Cross Entropy Loss
- Adam Optimizer
- ReduceLROnPlateau Learning Rate Scheduler
- Batch Size: 32
- Training Epochs: 30

---

## Workflow

```
Indian Sign Language Dataset
            │
            ▼
Transfer Learning (ResNet18)
            │
            ▼
FP32 Model (.pth)
            │
            ▼
INT8 Quantization
            │
            ▼
Vitis AI Compilation
            │
            ▼
resnet18_isl.xmodel
            │
            ▼
Deployment on AMD Versal VCK190
            │
            ▼
Real-Time Sign Language Recognition
```

---

## Dataset

Indian Sign Language Alphabet Dataset

https://github.com/ayeshatasnim-h/Indian-Sign-Language-dataset

---

## Repository Structure

```
sign-language-recognition-vck190
│
├── models/
│   ├── isl_resnet18.pth
│   ├── resnet18_isl_vck190.xmodel
│   └── resnet18_isl_zcu104.xmodel
│
├── train_resnet18.py
├── evaluate.py
├── predict.py
├── test_resnet18.py
├── calibrate.py
├── calibimggen.py
├── classes.txt
│
├── README.md
├── requirements.txt


```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/sign-language-recognition-vck190.git

cd sign-language-recognition-vck190
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the ResNet18 model

```bash
python train_resnet18.py
```

---

## Evaluation

Evaluate the trained model

```bash
python evaluate.py
```

---

## Prediction

Predict an image

```bash
python predict.py
```

Run webcam inference

```bash
python webcam.py
```

---

## Quantization

Generate calibration images

```bash
python calibimggen.py
```

Run calibration

```bash
python calibrate.py
```

Compile the quantized model using the AMD Vitis AI compiler for deployment on the VCK190 DPU.

---

## Results

| Model | Accuracy |
|--------|---------:|
| FP32 ResNet18 | **99.69%** |
| INT8 Quantized Model | **99.27%** |

Only **0.42%** accuracy degradation was observed after INT8 quantization while enabling efficient execution on the AMD Versal VCK190 DPU.

---

## Future Work

- Dynamic gesture recognition
- Continuous sentence recognition
- Support for larger sign language vocabularies
- Real-time FPGA camera pipeline integration
- Edge deployment optimization

---

## Documentation

A detailed presentation describing the complete workflow, training methodology, quantization process, and deployment pipeline is available in the `docs/` directory.

---

## License

This project is released under the MIT License.
