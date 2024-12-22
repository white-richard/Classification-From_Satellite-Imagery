from torch.utils.data import DataLoader
from datasets.coco_custom_dataset import COCOCustomDataset
from models.object_detection_model import ObjectDetectionModel
from transforms.augmentations import get_test_transform
from utils.collate import collate_fn
from config import TEST_DIR, TEST_ANNOTATIONS, MODEL_PATH, CONFIDENCE_THRESHOLD, EVAL_METRICS_PATH

def evaluate_model():
    test_dataset = COCOCustomDataset(
        root_dir=TEST_DIR,
        annotation_file=TEST_ANNOTATIONS,
        transform=get_test_transform()
    )

    test_loader = DataLoader(
        test_dataset, batch_size=10, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn
    )

    model = ObjectDetectionModel(
        num_classes=len(test_dataset.coco.getCatIds()),
        model_path=MODEL_PATH,
        backbone="resnet50",
        pretrained=True
    )

    model.load_model()
    # stats = model.evaluate_model(test_loader, test_dataset.coco, EVAL_METRICS_PATH, CONFIDENCE_THRESHOLD)
    stats = model.evaluate_model(
        test_loader,
        test_dataset.coco,
        EVAL_METRICS_PATH,
        CONFIDENCE_THRESHOLD,
    )

    print("Evaluation Stats:", stats)
