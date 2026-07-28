"""Gradio demo for the speech emotion classifier (ONNX runtime).

Local:
    python app.py
Deployment (Render etc.) binds to the PORT env var automatically.
"""

import os
import numpy as np
import librosa
import onnxruntime as ort
import gradio as gr

SR = 22050
N_MELS = 128
MAX_LEN = 130
EMOTIONS = ["angry", "calm", "disgust", "fearful",
            "happy", "neutral", "sad", "surprised"]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "emotion_cnn.onnx")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def fix_width(mel_db, max_len=MAX_LEN):
    w = mel_db.shape[1]
    if w > max_len:
        return mel_db[:, :max_len]
    return np.pad(mel_db, ((0, 0), (0, max_len - w)), mode="minimum")


def wav_to_input(path):
    y, _ = librosa.load(path, sr=SR)
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS), ref=np.max)
    mel_db = (fix_width(mel_db) + 80.0) / 80.0  # scale dB to [0, 1], same as training
    return mel_db[None, None].astype(np.float32)  # (1, 1, 128, 130)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(audio_path):
    if audio_path is None:
        return {}
    logits = session.run(None, {input_name: wav_to_input(audio_path)})[0][0]
    probs = softmax(logits)
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
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
