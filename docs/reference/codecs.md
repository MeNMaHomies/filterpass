# Audio Codecs Reference

This document provides a comprehensive list of notable audio codecs, categorized by their compression type and primary use cases.

### Codec Type Overview

- **Uncompressed**: No audio data is removed, offering the highest quality but resulting in very large file sizes.
- **Lossless**: Compressed so that it is an exact, bit-for-bit replica of the original when decoded (~50% size reduction).
- **Lossy**: Audio data is permanently discarded to achieve massive file size reductions (up to 1/10th of the original).
- **Speech**: Specialized lossy codecs ruthlessly optimized for the human voice at extremely low bitrates.

| Codec Name | Type | Key Characteristics / Use Case |
| :--- | :--- | :--- |
| **PCM** | Uncompressed | Raw digital audio data; standard for CDs and DVDs. |
| **WAV** | Uncompressed | Standard Windows container (Microsoft/IBM); used in professional audio editing. |
| **AIFF** | Uncompressed | Apple's equivalent to WAV; used for high-fidelity audio on macOS. |
| **FLAC** | Lossless | Most popular open-source lossless format; wide hardware and software support. |
| **ALAC** | Lossless | Apple's lossless format; used for Apple Music and the Apple ecosystem. |
| **WMA Lossless** | Lossless | Microsoft's proprietary lossless codec for Windows Media Player. |
| **Monkey's Audio (APE)** | Lossless | High compression ratio but high CPU requirement; less common on mobile. |
| **WavPack (WV)** | Lossless | Open-source; features a unique hybrid mode (lossy + correction file). |
| **Dolby TrueHD** | Lossless | Proprietary surround sound codec used primarily on Blu-ray discs. |
| **DTS-HD Master Audio** | Lossless | Proprietary surround sound; used on Blu-ray media as a Dolby competitor. |
| **MP3** | Lossy | Universal compatibility; revolutionary but technically surpassed by AAC/Opus. |
| **AAC** | Lossy | Modern standard; better quality than MP3 at similar bitrates. Used by YouTube and Apple. |
| **Ogg Vorbis** | Lossy | Open-source and patent-free; widely used in video games and by Spotify. |
| **Opus** | Lossy | Current gold standard for low-latency; highly versatile (WebRTC, VoIP, streaming). |
| **WMA** | Lossy | Microsoft's legacy lossy format; legacy Windows support. |
| **AC-3 (Dolby Digital)** | Lossy | Standard for multi-channel audio in DVDs and digital television. |
| **DTS** | Lossy | Home-theater surround sound; often higher bitrate than AC-3. |
| **Musepack (MPC)** | Lossy | Highly optimized for transparency at high bitrates; relatively niche. |
| **ATRAC** | Lossy | Sony proprietary codec; famously used in Minidisc players. |
| **AMR** | Speech | Optimized for human voice; standard for 3G and cellular voice recordings. |
| **G.7xx (G.711, etc.)** | Speech | ITU-T standards for landline phones and business VoIP systems. |
| **Speex** | Speech | Older open-source speech codec; largely replaced by Opus. |
| **SILK** | Speech | Skype's advanced speech codec; merged into the modern Opus format. |
