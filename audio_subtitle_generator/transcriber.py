from faster_whisper import WhisperModel
import numpy as np

MODEL_SIZE = "small"  # You can choose "tiny", "base", "small", "medium", or "large"
MODEL_DIR = "models"  # Directory to store model files
try:
    # Try loading the model from local files first to avoid unnecessary downloads
    model = WhisperModel(
        MODEL_SIZE,
        device="auto",
        compute_type="int8",
        download_root=MODEL_DIR,
        local_files_only=True
    )
except Exception:
    # If loading from local files fails, fall back to downloading the model
    model = WhisperModel(
        MODEL_SIZE,
        device="auto",
        compute_type="int8",
        download_root=MODEL_DIR,
        local_files_only=False
    )

def start_transcription(audio_queue, text_queue):
    buffer = np.array([], dtype=np.float32)

    while True:
        audio = audio_queue.get()
        buffer = np.concatenate((buffer, audio))

        # Wait until we have ~5 seconds audio
        if len(buffer) < 16000 * 5:
            continue

        print("🧠 Transcribing 5 sec audio...")

        segments, _ = model.transcribe(
            buffer,
            task="translate",
            beam_size=1,
            vad_filter=True,
            best_of=1
        )

        texts = [seg.text for seg in segments]
        print("🧠 Segments:", texts)

        text = " ".join(texts)

        if text.strip():
            print("✅ Final text:", text)
            text_queue.put(text)

        # Clear buffer after transcription
        buffer = np.array([], dtype=np.float32)
