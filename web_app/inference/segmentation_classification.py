from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

CODE_DIR = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Code work")
SEGMENTATION_MODEL_PATH = CODE_DIR / "Yolo_seg" / "runs" / "segment" / "breed_segmentation" / "yolov8n_seg_optimized-2" / "weights" / "best.pt"
GATE_MODEL_PATH = CODE_DIR / "yolov8m.pt"
CLASSIFICATION_MODEL_PATH = CODE_DIR / "best_model_v2_effnetb2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
GATE_CONFIDENCE = 0.30
SEGMENTATION_CONFIDENCE = 0.35
SEGMENTATION_IOU = 0.45
COW_CLASS_ID = 19
CLASSIFICATION_CONFIDENCE = 40.0
CLASSIFICATION_ENTROPY = 0.9


class SegmentationClassificationService:
    def __init__(self):
        self.gate_model = YOLO(str(GATE_MODEL_PATH))
        self.segmentation_model = YOLO(str(SEGMENTATION_MODEL_PATH))

        checkpoint = torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE)
        self.classes = checkpoint["classes"]
        self.classification_model = models.efficientnet_b2(weights=None)
        self.classification_model.classifier[1] = nn.Linear(
            self.classification_model.classifier[1].in_features,
            len(self.classes),
        )
        self.classification_model.load_state_dict(checkpoint["model_state_dict"])
        self.classification_model = self.classification_model.to(DEVICE).eval()
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _classify_crop(self, crop: Image.Image) -> dict:
        tensor = self.transform(crop).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(self.classification_model(tensor), dim=1)[0]

        entropy = float(-torch.sum(probabilities * torch.log(probabilities + 1e-12)).item())
        values, indices = torch.topk(probabilities, min(5, len(self.classes)))
        predictions = [
            {"class_name": self.classes[index.item()], "confidence": round(value.item() * 100, 2)}
            for value, index in zip(values, indices)
        ]
        is_known = (
            predictions[0]["confidence"] >= CLASSIFICATION_CONFIDENCE
            and entropy <= CLASSIFICATION_ENTROPY
        )
        return {
            "is_known": is_known,
            "top": predictions[0],
            "top5": predictions,
            "entropy": round(entropy, 4),
        }

    def process_image(self, input_path: Path, annotated_path: Path, crop_dir: Path) -> dict:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise ValueError("Could not read the uploaded image.")

        gate_result = self.gate_model.predict(
            frame, classes=[COW_CLASS_ID], conf=GATE_CONFIDENCE,
            device=YOLO_DEVICE, verbose=False,
        )[0]
        if gate_result.boxes is None or len(gate_result.boxes) == 0:
            annotated = frame.copy()
            cv2.putText(annotated, "No cattle/buffalo detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if not cv2.imwrite(str(annotated_path), annotated):
                raise ValueError("Could not save the annotated image.")
            return {"image_name": input_path.name, "detected": False, "items": []}

        segmentation = self.segmentation_model.predict(
            frame, conf=SEGMENTATION_CONFIDENCE, iou=SEGMENTATION_IOU,
            device=YOLO_DEVICE, verbose=False,
        )[0]
        annotated = frame.copy()
        overlay = frame.copy()
        height, width = frame.shape[:2]
        if segmentation.masks is not None:
            for mask_tensor in segmentation.masks.data:
                mask = cv2.resize(mask_tensor.cpu().numpy(), (width, height)) > 0.5
                overlay[mask] = (60, 180, 75)
            annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)

        crop_dir.mkdir(parents=True, exist_ok=True)
        items = []
        if segmentation.boxes is not None:
            for index, box in enumerate(segmentation.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(width, x2), min(height, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                crop_bgr = frame[y1c:y2c, x1c:x2c]
                crop_path = crop_dir / f"crop_{index + 1}.jpg"
                if not cv2.imwrite(str(crop_path), crop_bgr):
                    raise ValueError("Could not save a cropped image.")
                classification = self._classify_crop(
                    Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                )
                label = (
                    f"{classification['top']['class_name']} {classification['top']['confidence']:.1f}%"
                    if classification["is_known"]
                    else f"Unknown (conf {classification['top']['confidence']:.1f}%)"
                )
                color = (0, 200, 0) if classification["is_known"] else (0, 0, 200)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, text_height + 8)
                cv2.rectangle(annotated, (x1, label_y - text_height - 8),
                              (x1 + text_width + 4, label_y), color, -1)
                cv2.putText(annotated, label, (x1 + 2, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                items.append({"crop_name": crop_path.name, "box": [x1, y1, x2, y2], **classification})

        if not cv2.imwrite(str(annotated_path), annotated):
            raise ValueError("Could not save the annotated image.")
        return {"image_name": input_path.name, "detected": True, "items": items}
