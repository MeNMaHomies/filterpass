import threading
import queue
import time


class RealTimeDeepfakeDetector:
    def __init__(self, detection_model):
        self.model = detection_model

        # 1. Thread-Safe Synchronization [3]
        # We restrict the maxsize so if the AI falls behind, we don't consume infinite RAM.
        self.chunk_queue = queue.Queue(maxsize=10)

        # 2. Static Thread Pool Outsourcing [4]
        # We create a single, static background thread dedicated strictly to inference.
        self.is_running = False
        self.inference_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )

    def start(self):
        self.is_running = True
        self.inference_thread.start()

    def stop(self):
        self.is_running = False
        self.inference_thread.join()

    def _inference_worker(self):
        """
        The separated inference engine. This thread safely polls the queue and
        executes the neural network without interrupting the audio stream [2].
        """
        while self.is_running:
            try:
                # Controlled blocking using semaphores: waits for data but times out safely [3]
                chunk = self.chunk_queue.get(timeout=0.05)

                # --- NEURAL NETWORK INFERENCE HAPPENS HERE ---
                # Because this is on a separate thread, unpredictable execution times
                # (dynamic memory allocation, OS thread locks) will not stutter the audio.
                prediction = self.model.evaluate(chunk)
                print(f"Evaluated 300ms chunk. Deepfake Score: {prediction}")

                self.chunk_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")

    def ingest_audio(self, streamer, pcm_data, sample_rate):
        """
        The main audio callback loop. It remains completely free of blocking operations [2].
        """
        # Call the overlapping sliding window generator we created earlier
        generator = streamer.stream_voiced_audio(
            pcm_data, sample_rate, buffer_ms=300, hop_ms=100
        )

        # Continuously process discrete chunks independent of the host's buffer [2]
        for chunk_to_send in generator:
            if not self.chunk_queue.full():
                # Pass the chunk to the inference engine safely
                self.chunk_queue.put(chunk_to_send)
            else:
                # REAL-TIME SAFETY NET:
                # If the neural network is taking too long to process (a real-time violation),
                # we must drop the frame here rather than pausing the microphone stream.
                print(
                    "Warning: Inference engine overloaded. Dropping 100ms hop to maintain real-time flow."
                )
