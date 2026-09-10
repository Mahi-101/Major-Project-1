from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import timm

CODE_DIR = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Code work")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
REJECTION_CONFIDENCE_THRESHOLD = 40.0
REJECTION_ENTROPY_THRESHOLD = 0.9


class ClassificationService:
    def __init__(self):
        self.models = {
            "EfficientNet-B2": self._load_effnet(),
            "ViT-Small": self._load_vit(),
            "ConvNeXt-Tiny": self._load_convnext(),
        }
        self.transforms = {
            "EfficientNet-B2": self._transform(260),
            "ViT-Small": self._transform(224),
            "ConvNeXt-Tiny": self._transform(224),
        }

    @staticmethod
    def _transform(size: int):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def _load_effnet(self):
        checkpoint = torch.load(CODE_DIR / "best_model_v2_effnetb2.pth", map_location=DEVICE)
        classes = checkpoint["classes"]
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(DEVICE).eval(), classes

    def _load_vit(self):
        checkpoint = torch.load(CODE_DIR / "best_model_v2_vit_small.pth", map_location=DEVICE)
        classes = checkpoint["classes"]
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(DEVICE).eval(), classes

    def _load_convnext(self):
        mapping_path = CODE_DIR / "ConvNeXT Model" / "label_mapping.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        classes = [mapping[str(index)] for index in range(len(mapping))]
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(classes))
        state_dict = torch.load(CODE_DIR / "ConvNeXT Model" / "best_convnext_tiny.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        return model.to(DEVICE).eval(), classes

    def predict(self, image_path: Path, top_k: int = 5):
        image = Image.open(image_path).convert("RGB")
        output = {"image_name": image_path.name, "models": []}
        with torch.inference_mode():
            for name, (model, classes) in self.models.items():
                tensor = self.transforms[name](image).unsqueeze(0).to(DEVICE)
                probabilities = torch.softmax(model(tensor), dim=1)[0]
                entropy = float(-torch.sum(probabilities * torch.log(probabilities + 1e-12)).item())
                values, indices = torch.topk(probabilities, min(top_k, len(classes)))
                predictions = [
                    {"class_name": classes[index.item()], "confidence": round(value.item() * 100, 2)}
                    for value, index in zip(values, indices)
                ]
                top_confidence = predictions[0]["confidence"]
                is_known = (
                    top_confidence >= REJECTION_CONFIDENCE_THRESHOLD
                    and entropy <= REJECTION_ENTROPY_THRESHOLD
                )
                output["models"].append({
                    "name": name,
                    "top": predictions[0],
                    "top5": predictions,
                    "is_known": is_known,
                    "entropy": round(entropy, 4),
                })
        return output
