# Decoupling Audio Ingestion from Inference: The ANIRA Pattern

## Overview
In real-time audio applications, maintaining a continuous and uninterrupted signal flow is critical. Standard neural network inference engines—such as ONNX Runtime, LibTorch, and TensorFlow Lite—are typically optimized for average processing throughput rather than strict, deterministic real-time constraints [1]. Executing these models directly on the audio processing thread inevitably leads to "real-time violations." These violations include unpredictable dynamic memory allocations, thread synchronization locks, or delays introduced by the operating system's thread scheduler [2]. 

If the audio stream waits for these non-deterministic operations to finish, the stream will block, causing stuttering, dropped frames, and audio dropouts. To resolve this, our project adopts the architectural principles of **ANIRA (Architecture for Neural Network Inference in Real-Time Audio)** [3]. This pattern guarantees real-time safety by strictly decoupling the audio callback from the neural network inference process.

## Core Principles of the ANIRA Pattern
1. **Decoupled Processing:** The audio ingestion loop operates completely independently of the inference engine. This ensures the continuous microphone stream is never blocked by the AI's evaluation time.
2. **Thread-Safe Synchronization:** Data is passed between the audio thread and the inference thread using thread-safe structures (such as lock-free atomics or semaphores). This prevents race conditions and avoids unconditional blocking [4].
3. **Static Thread Pool:** Inference is offloaded to a pre-allocated, static background thread or thread pool. By keeping the pool static, the system avoids "oversubscription"—a scenario where dynamically spawning new high-priority threads for every inference task exceeds available hardware threads and severely degrades system performance [1].

## Application to the Audio Deepfake Detection Project
For our real-time deepfake detection pipeline, we apply the ANIRA architecture to safely feed live audio to our lightweight detection models (such as AASIST-L or RawNet2) [5] without causing audio dropouts.

### 1. The Audio Ingestion Loop (Main Thread)
We capture a live 16kHz microphone stream and utilize a Voice Activity Detection (VAD) generator to isolate voiced frames. These frames are continuously accumulated into a sliding, overlapping buffer (e.g., passing a 300ms window every 100ms). Because this loop runs on the primary audio thread, it must **never wait** for the neural network to finish its evaluation of the previous chunk.

### 2. Thread-Safe Queue (Synchronization)
When the audio buffer reaches the exact input shape required by the downstream model, the resulting chunk of bytes is immediately pushed into a thread-safe queue (`queue.Queue` in Python, which utilizes semaphores internally for safe, controlled blocking). We apply a maximum size limit to this queue. If the queue becomes full—indicating the inference engine is falling behind—the system intentionally drops the oldest audio chunk rather than pausing the microphone stream. This serves as a vital real-time safety net.

### 3. Static Inference Worker (Background Thread)
Upon initialization, the system spawns a single, static daemon thread dedicated entirely to model inference. This thread continuously polls the thread-safe queue. When a chunk arrives, it runs the forward pass of the deepfake detection model and outputs the probability score. Because the execution is entirely isolated on this static thread, any real-time violations committed by the underlying deep learning framework (like OS thread locking or memory allocation) do not impact the live audio stream.