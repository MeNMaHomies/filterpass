import torch
import torch.nn.functional as F
import librosa
import soundfile as sf


def load_single_file(file_path, max_length):
    """
    Preprocess a single audio file to match dataset pipeline.
    
    Args:
        file_path (str): Path to audio file
        max_length (int): Expected sequence length (e.g., 16000 * seconds)
    
    Returns:
        torch.Tensor: (T,)
    """
    waveform_np, sample_rate = sf.read(file_path)

    # 1. Convert to mono
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=1)

    # 2. Resample to 16kHz
    if sample_rate != 16000:
        waveform_np = librosa.resample(
            waveform_np, orig_sr=sample_rate, target_sr=16000
        )

    waveform = torch.tensor(waveform_np, dtype=torch.float32)

    # 3. RMS normalize (matches dataset.py — more robust than peak norm)
    eps = 1e-9
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > eps:
        waveform = torch.clamp(waveform / (rms + eps), -1.0, 1.0)
    else:
        peak = torch.max(torch.abs(waveform))
        if peak > eps:
            waveform = waveform / peak

    # 4. Centre crop or pad (matches dataset.py eval behaviour)
    seq_len = waveform.shape[0]
    if seq_len > max_length:
        start = (seq_len - max_length) // 2
        waveform = waveform[start : start + max_length]
    elif seq_len < max_length:
        waveform = F.pad(waveform, (0, max_length - seq_len))

    return waveform


def predict_file(file_path, model, device, max_length):
    """
    Run inference on a single audio file.
    
    Args:
        file_path (str)
        model (torch.nn.Module)
        device (torch.device)
        max_length (int)
    
    Returns:
        dict: prediction + probability
    """
    model.eval()

    waveform = load_single_file(file_path, max_length)
    input_values = waveform.unsqueeze(0).to(device)           # (1, T)
    attention_mask = torch.ones(1, max_length, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)

        spoof_prob = probs[:, 1].item()
        pred = torch.argmax(logits, dim=1).item()

    return {
        "prediction": pred,
        "spoof_probability": spoof_prob
    }