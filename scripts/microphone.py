import pyaudio
import webrtcvad

# --- WebRTC Target Specifications ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
BUFFER_TARGET_MS = 500
CHUNK_SIZE_SAMPLES = int(RATE * (CHUNK_DURATION_MS / 1000.0))  # 480 samples per 30ms
BYTES_PER_TARGET_BUFFER = int(RATE * (BUFFER_TARGET_MS / 1000.0) * 2)


def live_speech_stream():
    """Listens to the microphone and yields 500ms chunks of isolated speech."""
    vad = webrtcvad.Vad(3)

    # Initialize Microphone Hardware
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE_SAMPLES,
    )

    voice_buffer = bytearray()
    print(f"🎙️ Listening for speech... (Press Ctrl+C to stop)")

    try:
        while True:
            frame = stream.read(CHUNK_SIZE_SAMPLES, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)

            if is_speech:
                voice_buffer.extend(frame)

                if len(voice_buffer) >= BYTES_PER_TARGET_BUFFER:
                    chunk_to_send = bytes(voice_buffer[:BYTES_PER_TARGET_BUFFER])
                    yield chunk_to_send

                    # Keep any remainder for the next bucket
                    voice_buffer = voice_buffer[BYTES_PER_TARGET_BUFFER:]

    except KeyboardInterrupt:
        print("\n🛑 Stopped listening.")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    for index, voiced_chunk in enumerate(live_speech_stream()):

        print(
            f"[{index}] Captured {BUFFER_TARGET_MS}ms of continuous speech! Sending to Deepfake Model..."
        )

        # BENCHMARKING
        # prediction_score = sota_model.predict(voiced_chunk)
