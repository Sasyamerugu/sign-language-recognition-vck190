import os
import cv2
import xir
import vart
import numpy as np
import time

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/home/root/resnet18_isl.xmodel"
DATASET_DIR = "/home/root/test"

CLASSES = [chr(ord('a') + i) for i in range(26)]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# -----------------------------
# LOAD DPU RUNNER
# -----------------------------
graph = xir.Graph.deserialize(MODEL_PATH)
root = graph.get_root_subgraph()

dpu_subgraph = None

for sg in root.toposort_child_subgraph():
    if sg.has_attr("device"):
        if sg.get_attr("device").upper() == "DPU":
            dpu_subgraph = sg
            break

if dpu_subgraph is None:
    raise RuntimeError("No DPU subgraph found!")

runner = vart.Runner.create_runner(dpu_subgraph, "run")

input_tensor = runner.get_input_tensors()[0]
output_tensor = runner.get_output_tensors()[0]

input_shape = tuple(input_tensor.dims)
output_shape = tuple(output_tensor.dims)

print("Input shape :", input_shape)
print("Output shape:", output_shape)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img_path):

    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32) / 255.0

    img = (img - MEAN) / STD

    return img

# -----------------------------
# INFERENCE
# -----------------------------
def predict(img_path):

    img = preprocess(img_path)

    # Quantize using input fixpos = 5
    single = np.round(img * (2 ** 5)).astype(np.int8)

    # Create full batch expected by DPU
    input_data = np.zeros(input_shape, dtype=np.int8)

    for i in range(input_shape[0]):
        input_data[i] = single

    # Output buffer
    output_data = np.empty(output_shape, dtype=np.int8)

    start = time.perf_counter()

    job_id = runner.execute_async(
        [input_data],
        [output_data]
    )

    runner.wait(job_id)

    end = time.perf_counter()

    latency = (end - start) * 1000   # milliseconds

    pred_idx = np.argmax(output_data[0])

    return CLASSES[pred_idx], latency



# -----------------------------
# EVALUATION
# -----------------------------
total = 0
correct = 0
latencies = []

overall_start = time.perf_counter()

for true_class in CLASSES:

    class_dir = os.path.join(DATASET_DIR, true_class)

    if not os.path.isdir(class_dir):
        continue

    for fname in os.listdir(class_dir):

        if not fname.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        img_path = os.path.join(class_dir, fname)

        pred, latency = predict(img_path)
        latencies.append(latency)

        if pred == true_class:
            correct += 1

        total += 1

        if total <= 20:
            print(
                f"{fname:30s} "
                f"GT={true_class} "
                f"PRED={pred}"
            )

overall_end = time.perf_counter()

total_time = overall_end - overall_start
print("\n-------------------")
print("Total   :", total)
print("Correct :", correct)

if total > 0:
    print(
        "Accuracy:",
        round(100.0 * correct / total, 2),
        "%"
    )

if len(latencies) > 0:

    total_dpu_time = sum(latencies)          # milliseconds

    avg_latency = total_dpu_time / len(latencies)

    fps = 1000.0 / avg_latency

    throughput = total / total_time

    print(f"Average DPU Latency     : {avg_latency:.3f} ms")
    print(f"Total DPU Inference Time: {total_dpu_time:.2f} ms")
    print(f"FPS                     : {fps:.2f}")
    print(f"Throughput              : {throughput:.2f} images/sec")
    print(f"Total Evaluation Time   : {total_time:.2f} sec")

