"""Gradio demo for the speech emotion classifier.

    conda activate audioml
    cd app && python app.py
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import gradio as gr

SR = 22050
N_MELS = 128
MAX_LEN = 130
MODEL_PATH = "../models/best_model.pt"
EMOTIONS = ["angry", "calm", "disgust", "fearful",
            "happy", "neutral", "sad", "surprised"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmotionCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, N_MELS, MAX_LEN)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = EmotionCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()


def fix_width(mel_db, max_len=MAX_LEN):
    w = mel_db.shape[1]
    if w > max_len:
        return mel_db[:, :max_len]
    return np.pad(mel_db, ((0, 0), (0, max_len - w)), mode="minimum")


def wav_to_input(path):
    y, _ = librosa.load(path, sr=SR)
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS), ref=np.max)
    mel_db = (fix_width(mel_db) + 80.0) / 80.0  # scale dB to [0, 1], same as training
    return torch.tensor(mel_db, dtype=torch.float32)[None, None].to(device)


def predict(audio_path):
    if audio_path is None:
        return {}
    with torch.no_grad():
        probs = torch.softmax(model(wav_to_input(audio_path)), dim=1)[0].cpu().numpy()
    return {emo: float(p) for emo, p in zip(EMOTIONS, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload or record a speech clip"),
    outputs=gr.Label(num_top_classes=3, label="Predicted emotion"),
    title="🎙️ Speech Emotion Recognition",
    description="A CNN trained on RAVDESS with a speaker-independent split "
                "(~46% test accuracy on unheard voices, 8 emotions).",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
