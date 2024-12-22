import torch
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from torchvision.models.detection.rpn import AnchorGenerator
from pycocotools.cocoeval import COCOeval
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.image_utils import denormalize_image
from config import MEAN, STD
import numpy as np

from config import DEVICE

class ObjectDetectionModel:
    def __init__(self, num_classes, model_path, backbone='resnet50_v2', pretrained=True):
        if num_classes is None:
            num_classes = 2
        self.num_classes = num_classes
        self.model_path = model_path
        self.backbone = backbone
        self.pretrained = pretrained
        self.model = self._initialize_model()
        self.model.to(DEVICE)

    def _initialize_model(self):
        # Load the appropriate backbone and weights
        if self.backbone == 'resnet50_v2':
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if self.pretrained else None
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")


       # Custom Anchor Generator
        anchor_generator = AnchorGenerator(
            # sizes=((16,), (32,), (64,), (128,), (256,)),  # Sizes for 5 feature maps
            sizes=((8,), (16,), (32,), (64,), (128,)),  # One size tuple per feature map
            aspect_ratios=((0.5, 1.0, 2.0),) * 5        
        )

        # Assign the custom anchor generator to the RPN
        model.rpn.anchor_generator = anchor_generator

        # Replace the box predictor (classification head) for the custom number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model

    def train_model(self, train_loader, val_loader, num_epochs, learning_rate, weight_decay, patience):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                total_loss += losses.item()
                print(f"\rEpoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {losses.item():.4f}", end="")

            total_loss /= len(train_loader)
            val_loss = self._validate(val_loader)
            scheduler.step(val_loss)

            print(f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def _validate(self, val_loader):
        total_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

        return total_loss / len(val_loader)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE,weights_only=True))
        self.model.to(DEVICE)
        self.model.eval()
        print("Model loaded for inference.")

    def evaluate_model(self, data_loader, coco_gt, output_json, confidence_threshold, output_dir):
        """
        Evaluate the model and save prediction images with bounding boxes.

        Args:
            data_loader: DataLoader for the test dataset.
            coco_gt: Ground-truth COCO object for evaluation.
            output_json: Path to save the evaluation results.
            confidence_threshold: Threshold for filtering low-confidence predictions.
            output_dir: Directory to save the evaluation images with predictions.
        """
        self.model.eval()
        results = []
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for img_idx, (images, targets) in enumerate(data_loader):
                batch_image_ids = [
                    data_loader.dataset.image_ids[idx]
                    for idx in range(
                        img_idx * data_loader.batch_size,
                        min((img_idx + 1) * data_loader.batch_size, len(data_loader.dataset)),
                    )
                ]

                images = [img.to(DEVICE) for img in images]
                outputs = self.model(images)

                for i, (img, output) in enumerate(zip(images, outputs)):
                    # Get predictions
                    boxes = output['boxes'][output['scores'] >= confidence_threshold].cpu().numpy()
                    scores = output['scores'][output['scores'] >= confidence_threshold].cpu().numpy()
                    labels = output['labels'][output['scores'] >= confidence_threshold].cpu().numpy()

                    # Save results for COCO evaluation
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        results.append({
                            "image_id": batch_image_ids[i],
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score),
                        })

                    # Save image with bounding boxes
                    self._save_prediction_image(img.cpu(), boxes, scores, labels, output_dir, batch_image_ids[i])

        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)

        coco_dt = coco_gt.loadRes(output_json)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = np.array([0.5])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()



        stats = coco_eval.stats
        return stats

    def _save_prediction_image(self, img, boxes, scores, labels, output_dir, image_id):
        """
        Save an image with predictions drawn as bounding boxes.

        Args:
            img: The image tensor (before normalization).
            boxes: Bounding boxes for the image.
            scores: Confidence scores for the bounding boxes.
            labels: Class labels for the bounding boxes.
            output_dir: Directory to save the images.
            image_id: Unique identifier for the image (used in filename).
        """
        # Denormalize the image
        img = denormalize_image(img, MEAN, STD)

        # Plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw each box
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min, y_min - 10,
                f'{label}: {score:.2f}', color='yellow',
                fontsize=10, weight='bold'
            )

        plt.axis('off')
        output_path = os.path.join(output_dir, f"eval_image_{image_id}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Saved evaluation image to {output_path}")

    def extract_region_proposals(self, images):
        self.model.eval()
        with torch.no_grad():
            features = self.model.backbone(images)
            proposals, _ = self.model.rpn(images, features)
        return proposals

    def extract_features(backbone, images, proposals):
        features = []
        for img, proposal in zip(images, proposals):
            cropped_regions = []
            for box in proposal:
                x_min, y_min, x_max, y_max = box.int().tolist()
                cropped_region = img[:, y_min:y_max, x_min:x_max]
                cropped_region = F.interpolate(cropped_region.unsqueeze(0), size=(224, 224))  # Resize to fixed size
                cropped_regions.append(cropped_region)
            features.append(torch.cat(cropped_regions, dim=0))
        return features
