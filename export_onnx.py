"""Export the trained PyTorch model to ONNX for lightweight deployment.

Run once from the project root:
    python export_onnx.py
"""

import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, 128, 130)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = EmotionCNN()
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu", weights_only=True))
model.eval()

dummy = torch.zeros(1, 1, 128, 130)
torch.onnx.export(
    model, dummy, "models/emotion_cnn.onnx",
    input_names=["spectrogram"], output_names=["logits"],
    dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
print("exported models/emotion_cnn.onnx")
