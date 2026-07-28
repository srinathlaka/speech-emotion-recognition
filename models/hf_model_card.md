---
license: mit
tags:
  - audio-classification
  - speech
  - emotion-recognition
  - onnx
  - cnn
  - ravdess
pipeline_tag: audio-classification
---

# Speech Emotion Recognition (CNN, RAVDESS)

A small convolutional network that classifies the emotion in a spoken clip from its
mel-spectrogram. Trained on [RAVDESS](https://zenodo.org/record/1188976) with a
**speaker-independent** split, so the reported accuracy reflects performance on
completely unheard voices.

- 🔗 **Live demo:** https://speech-emotion-recognition-apc1.onrender.com
- 💻 **Code:** https://github.com/srinathlaka/speech-emotion-recognition

## Classes

`angry, calm, disgust, fearful, happy, neutral, sad, surprised` (indices 0–7, alphabetical).

## Results

| Metric | Value |
|--------|-------|
| Test accuracy (unseen speakers) | **46.1%** |
| Best validation accuracy | 60.0% |
| Random baseline | 12.5% |

Errors are concentrated among emotions of similar arousal (angry↔happy↔surprised,
calm↔sad↔neutral). The model is strongest on **surprised** (~88% recall) and weakest on
**neutral** (underrepresented and acoustically ambiguous).

## How the input is built

Each audio clip is converted the same way at train and inference time:

1. Load mono at `sr = 22050`.
2. Mel-spectrogram with `n_mels = 128`, then `power_to_db(ref=np.max)`.
3. Pad/truncate the time axis to `130` frames → shape `(128, 130)`.
4. Scale from dB to `[0, 1]` via `(x + 80) / 80`.
5. Add batch + channel dims → `(1, 1, 128, 130)`.

## Usage (ONNX Runtime)

```python
import numpy as np
import librosa
import onnxruntime as ort

EMOTIONS = ["angry", "calm", "disgust", "fearful",
            "happy", "neutral", "sad", "surprised"]

session = ort.InferenceSession("emotion_cnn.onnx")


def preprocess(path):
    y, _ = librosa.load(path, sr=22050)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128), ref=np.max)
    if mel.shape[1] > 130:
        mel = mel[:, :130]
    else:
        mel = np.pad(mel, ((0, 0), (0, 130 - mel.shape[1])), mode="minimum")
    mel = (mel + 80.0) / 80.0
    return mel[None, None].astype(np.float32)


logits = session.run(None, {session.get_inputs()[0].name: preprocess("clip.wav")})[0][0]
probs = np.exp(logits) / np.exp(logits).sum()
print(EMOTIONS[int(probs.argmax())], float(probs.max()))
```

## Limitations

- Trained only on RAVDESS (acted North-American English speech); expect lower accuracy on
  spontaneous speech, other languages, or noisy audio.
- The `neutral` class is unreliable.
- Intended as a learning/portfolio project, not for real-world decision-making.
