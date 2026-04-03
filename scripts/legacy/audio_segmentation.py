import webrtcvad
from pydub import AudioSegment


def normalize_audio_for_webrtc(file_path, target_sample_rate=16000):
    """
    Converts any audio file to 16-bit Mono PCM at an acceptable WebRTC sample rate.
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # 1. Force mono channel
    # 2. Force 16-bit audio (2 bytes)
    # 3. Force into acceptable sample rate [8khz, 16khz, 32khz, 48khz]
    if audio.channels != 1:
        audio = audio.set_channels(1)
    if audio.sample_width != 2:
        audio = audio.set_sample_width(2)
    valid_rates = [8000, 16000, 32000, 48000]
    if audio.frame_rate not in valid_rates:
        audio = audio.set_frame_rate(target_sample_rate)

    # Sanity checks to ensure WebRTC won't crash
    if audio.channels != 1:
        raise ValueError("WebRTC VAD requires Mono audio.")
    if audio.sample_width != 2:
        raise ValueError("WebRTC VAD requires 16-bit audio.")
    if audio.frame_rate not in valid_rates:
        raise ValueError("WebRTC VAD requires 8k, 16k, 32k, or 48k Hz sample rate.")

    return audio.raw_data, audio.frame_rate


class VADStreamer:
    def __init__(self, mode=3, chunk_duration_ms=30):
        """
        Initializes the WebRTC VAD.
        :param mode: Aggressiveness mode (0 to 3). 3 is the most aggressive.
        :param chunk_duration_ms: Duration of each frame. Must be 10, 20, or 30ms.
        """
        self.vad = webrtcvad.Vad(mode)
        self.chunk_duration_ms = chunk_duration_ms

    def stream_voiced_audio(self, pcm_data, sample_rate, buffer_ms=300, hop_ms=100):
        """
        Simulates a live stream. Yields continuous blocks of voiced audio.
        """
        frames = self.frame_generator(pcm_data, sample_rate)

        # How many bytes equal the buffer size we want to send to the model?
        bytes_per_buffer = int(sample_rate * (buffer_ms / 1000.0) * 2)
        # How many bytes to slide the window forward (e.g., step forward 100ms)
        hop_bytes = int(sample_rate * (hop_ms / 1000.0) * 2)
        voice_buffer = bytearray()

        # Send frames through the VAD and collect voiced segments into the voice_buffer
        # Segment 1: [0ms - 500ms] -> Send to model
        # Segment 2: [100ms - 600ms] -> Send to model
        # Segment 3: [200ms - 700ms] -> Send to model
        for frame in frames:
            is_speech = self.vad.is_speech(frame, sample_rate)

            if is_speech:
                voice_buffer.extend(frame)

                # If we have collected enough continuous speech, yield it!
                while len(voice_buffer) >= bytes_per_buffer:

                    # Convert bytearray back to standard immutable bytes for downstream safety
                    chunk_to_send = bytes(voice_buffer[:bytes_per_buffer])
                    yield chunk_to_send

                    # Slide the window forward by the hop size.
                    voice_buffer = voice_buffer[hop_bytes:]

    def frame_generator(self, pcm_data, sample_rate):
        """Generates raw audio frames of the exact duration required by WebRTC."""
        offset = 0
        # Calculate exactly how many bytes per chunk
        chunk_size = int(sample_rate * (self.chunk_duration_ms / 1000.0) * 2)

        while offset + chunk_size <= len(pcm_data):
            yield pcm_data[offset : offset + chunk_size]
            offset += chunk_size


def segment_audio(path, mode=3, chunk_duration_ms=30, buffer_ms=500):
    streamer = VADStreamer(mode=mode, chunk_duration_ms=chunk_duration_ms)
    pcm_data, sample_rate = normalize_audio_for_webrtc(path)

    return [
        chunk
        for chunk in streamer.stream_voiced_audio(
            pcm_data, sample_rate, buffer_ms=buffer_ms
        )
    ]
