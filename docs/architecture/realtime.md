# Real-Time Audio Chunking Flowchart

This flowchart visualizes the continuous process of normalizing, extracting, and streaming valid voice chunks in real-time fashion, as defined in `audio_segmentation.py`.

```mermaid
flowchart TD
    %% Define color classes
    classDef startEnd fill:#f96,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef process fill:#4a90e2,stroke:#333,stroke-width:2px,color:#fff;
    classDef condition fill:#f8e71c,stroke:#333,stroke-width:2px,color:#333;
    classDef drop fill:#d0021b,stroke:#333,stroke-width:2px,color:#fff;
    classDef success fill:#7ed321,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef buffer fill:#9013fe,stroke:#333,stroke-width:2px,color:#fff;
    
    Start([Incoming Audio Stream]):::startEnd --> Normalize[Standardize Audio Format]:::process
    
    subgraph Preprocessing [Format Standardization]
        style Preprocessing fill:#f4f4f4,stroke:#ccc,stroke-width:2px,color:#333;
        Normalize --> C1[Convert to Mono Audio]:::process
        C1 --> C2[Set to 16-bit Depth]:::process
        C2 --> C3[Adjust Sample Rate\n e.g., 16kHz]:::process
        C3 --> PCM[Raw Audio Data Pipeline]:::process
    end
    
    PCM --> Generator[Slice into Tiny Chunks\n e.g., 30ms]:::process
    
    subgraph VAD [Voice Activity Filtering]
        style VAD fill:#f4f4f4,stroke:#ccc,stroke-width:2px,color:#333;
        Generator -->|Inspect each slice| IsSpeech(Analyze for Voice\nWebRTC VAD):::process
        IsSpeech --> Check{Is someone speaking?}:::condition
        Check -->|No| Drop[Discard Silence]:::drop
        Check -->|Yes| Buffer[Save Spoken Chunk to Buffer]:::buffer
        
        Buffer --> SizeCheck{Do we have enough speech yet?\n e.g., half a second}:::condition
        
        SizeCheck -->|Not Yet| NextFrame[Grab Next Slice]:::process
        NextFrame --> Generator
        
        SizeCheck -->|Yes, Ready!| Yield[Send the Full Speech Segment!]:::success
        Yield --> KeepRemainder[Save any leftover audio\nfor the next segment]:::buffer
        KeepRemainder --> NextFrame
    end

    Yield --> Inference([Run SOTA Model Prediction \n& Measure Latency]):::startEnd
```
