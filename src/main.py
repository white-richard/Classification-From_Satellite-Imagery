import os
from config import (
    TRAIN_DIR, TRAIN_ANNOTATIONS, VAL_DIR, VAL_ANNOTATIONS, TEST_DIR, TEST_ANNOTATIONS,
    MODEL_PATH, TEST_OUTPUT_DIR, INFERENCE_OUTPUT_DIR, INFERENCE_DIR, CONFIDENCE_THRESHOLD, INFERENCE_THRESHOLD,
    BATCH_SIZE, EVAL_METRICS_PATH, LEARNING_RATE, NUM_EPOCHS, PATIENCE, WEIGHT_DECAY
)
from datasets.coco_custom_dataset import COCOCustomDataset
from datasets.unlabeled_dataset import UnlabeledImagesDataset
from transforms.augmentations import get_train_transform, get_test_transform, get_inference_transform
from models.object_detection_model import ObjectDetectionModel
from inference.inference_runner import InferenceRunner
from utils.collate import collate_fn

from torch.utils.data import DataLoader


def train_model():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Prepare datasets and loaders
    train_dataset = COCOCustomDataset(
        root_dir=TRAIN_DIR,
        annotation_file=TRAIN_ANNOTATIONS,
        transform=get_train_transform()
    )
    val_dataset = COCOCustomDataset(
        root_dir=VAL_DIR,
        annotation_file=VAL_ANNOTATIONS,
        transform=get_test_transform()
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    # Initialize model
    model = ObjectDetectionModel(
        num_classes=len(train_dataset.coco.getCatIds()),
        model_path=MODEL_PATH,
        pretrained=True
    )

    # Train model
    model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=PATIENCE)


def evaluate_model():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Prepare datasets and loaders
    test_dataset = COCOCustomDataset(
        root_dir=TEST_DIR,
        annotation_file=TEST_ANNOTATIONS,
        transform=get_test_transform()
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    # Initialize and load model
    model = ObjectDetectionModel(
        num_classes=len(test_dataset.coco.getCatIds()),
        model_path=MODEL_PATH,
        pretrained=True
    )
    model.load_model()

    # Evaluate model
    stats = model.evaluate_model(test_loader, test_dataset.coco, output_json=EVAL_METRICS_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, output_dir=TEST_OUTPUT_DIR)
    print("Evaluation Stats:", stats)


def run_inference():
    os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

    # Prepare dataset and loader
    unlabeled_dataset = UnlabeledImagesDataset(root_dir=INFERENCE_DIR, transform=get_inference_transform())
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    # Initialize and load model
    model = ObjectDetectionModel(
        num_classes=None,
        model_path=MODEL_PATH,
        pretrained=True
    )
    model.load_model()

    # Run inference
    inference_runner = InferenceRunner(model, threshold=INFERENCE_THRESHOLD, output_dir=INFERENCE_OUTPUT_DIR)
    results = inference_runner.predict(unlabeled_loader)
    inference_runner.save_predictions(results)


if __name__ == '__main__':
    print("Use the dedicated commands for training, evaluation, or inference.")


# import os
# from config import (
#     TRAIN_DIR, TRAIN_ANNOTATIONS, VAL_DIR, VAL_ANNOTATIONS, TEST_DIR, TEST_ANNOTATIONS,
#     MODEL_PATH, TEST_OUTPUT_DIR, INFERENCE_OUTPUT_DIR, INFERENCE_DIR, CONFIDENCE_THRESHOLD, INFERENCE_THRESHOLD,
#     BATCH_SIZE, EVAL_METRICS_PATH,LEARNING_RATE,NUM_EPOCHS,PATIENCE, WEIGHT_DECAY
# )
# from datasets.coco_custom_dataset import COCOCustomDataset
# from datasets.unlabeled_dataset import UnlabeledImagesDataset
# from transforms.augmentations import get_train_transform, get_test_transform, get_inference_transform
# from models.object_detection_model import ObjectDetectionModel
# from inference.inference_runner import InferenceRunner
# from utils.collate import collate_fn

# from torch.utils.data import DataLoader
# import argparse

# def main(train=False, test=False, inference=False):
#     os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
#     os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

#     # Prepare datasets and loaders
#     train_dataset = COCOCustomDataset(
#         root_dir=TRAIN_DIR,
#         annotation_file=TRAIN_ANNOTATIONS,
#         transform=get_train_transform()
#     )

#     val_dataset = COCOCustomDataset(
#         root_dir=VAL_DIR,
#         annotation_file=VAL_ANNOTATIONS,
#         transform=get_test_transform()
#     )

#     test_dataset = COCOCustomDataset(
#         root_dir=TEST_DIR,
#         annotation_file=TEST_ANNOTATIONS,
#         transform=get_test_transform()
#     )

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)

#     model = ObjectDetectionModel(
#         num_classes=len(train_dataset.coco.getCatIds()),
#         model_path=MODEL_PATH,
#         # backbone="resnet50",
#         pretrained=True
#     )

#     if train:
#         model.train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=PATIENCE)

#     model.load_model()

#     if test:
#         stats = model.evaluate_model(test_loader, test_dataset.coco, output_json=EVAL_METRICS_PATH, confidence_threshold=CONFIDENCE_THRESHOLD,output_dir=TEST_OUTPUT_DIR)
#         print("Evaluation Stats:", stats)

#     if inference:
#         inference_runner = InferenceRunner(model, threshold=INFERENCE_THRESHOLD, output_dir=INFERENCE_OUTPUT_DIR)
#         unlabeled_dataset = UnlabeledImagesDataset(root_dir=INFERENCE_DIR, transform=get_inference_transform())
#         unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)
#         results = inference_runner.predict(unlabeled_loader)
#         inference_runner.save_predictions(results)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train", action="store_true", help="Train the model")
#     parser.add_argument("--test", action="store_true", help="Evaluate the model on the test set")
#     parser.add_argument("--inference", action="store_true", help="Run inference on unlabeled images")
#     args = parser.parse_args()
#     main(train=args.train, test=args.test, inference=args.inference)
