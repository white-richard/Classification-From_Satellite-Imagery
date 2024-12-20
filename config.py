import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/HRPlanes_coco/"
TRAIN_DIR = os.path.join(DATA_DIR, "train/")
VAL_DIR = os.path.join(DATA_DIR, "valid/")
TEST_DIR = os.path.join(DATA_DIR, "test/")
INFERENCE_DIR = "data/inference_input/"

TRAIN_ANNOTATIONS = os.path.join(TRAIN_DIR, "_annotations.coco.json")
VAL_ANNOTATIONS = os.path.join(VAL_DIR, "_annotations.coco.json")
TEST_ANNOTATIONS = os.path.join(TEST_DIR, "_annotations.coco.json")

MODEL_DIR = "model_weights/"
MODEL_PATH = os.path.join(MODEL_DIR, "faster_rcnn.pth")

OUTPUT_DIR = "output/"
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test/")
INFERENCE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "inference/")
EVAL_METRICS_PATH = os.path.join(OUTPUT_DIR, "test/eval_metrics.json")

BATCH_SIZE = 10
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0005
PATIENCE = 5
CONFIDENCE_THRESHOLD = 0.9
INFERENCE_THRESHOLD = 0.5

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
