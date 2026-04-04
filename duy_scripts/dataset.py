import os
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from torch.utils.data import Dataset

class ASVspoof2019LADataset(Dataset):
    def __init__(self, base_dir, split="train", max_seconds=4.0):
        """
        Args:
            base_dir (str): Path to the root 'ASVspoof2019_LA' directory.
            split (str): One of 'train', 'dev', or 'eval'.
            max_seconds (float): Maximum length of audio in seconds.
        """
        self.base_dir = base_dir
        self.split = split
        self.max_length = int(16000 * max_seconds) # Wav2Vec2 strictly expects 16kHz
        
        # Map splits to their specific directories and protocol files
        split_map = {
            "train": ("train", "ASVspoof2019.LA.cm.train.trn.txt"),
            "dev":   ("dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
            "eval":  ("eval", "ASVspoof2019.LA.cm.eval.trl.txt")
        }
        
            
        audio_dir_name, protocol_name = split_map[split]
        # self.audio_dir = os.path.join(base_dir, audio_dir_name, "flac")
        self.audio_dir = os.path.join("../output/", audio_dir_name)
        protocol_path = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols", protocol_name)
        
        print(f"Reading folder: {self.audio_dir}")

        self.data = []
        
        # Define labels: 0 for bonafide (minority), 1 for spoof (majority)
        self.label_map = {"bonafide": 0, "spoof": 1}
        
        # Parse standard ASVspoof 2019 LA protocol file
        # Format: SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID ATTACK_ID KEY
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    filename = parts[1]
                    label_str = parts[4]
                    self.data.append({
                        "path": os.path.join(self.audio_dir, f"{filename}.flac"),
                        "label": self.label_map[label_str]
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 0. Load audio using soundfile
        waveform_np, sample_rate = sf.read(item["path"]) # waveform_np returns (T,) -> Mono, or (T, C). T = Time step, C = Channel
        labels = torch.tensor(item["label"], dtype=torch.long)

        # 1. Convert to mono if stereo BEFORE resampling (saves computation time)
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1) # Average across channels
            
        # 2. Resample to 16kHz
        if sample_rate != 16000:
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=16000)
            
        # Convert the processed numpy array to a PyTorch tensor
        waveform = torch.tensor(waveform_np, dtype=torch.float32)
            
        # 3. Amplitude normalization [-1, 1]
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
            
        # 4. Padding or Truncating to exact length for uniform batching
        seq_len = waveform.shape[0]
        if seq_len > self.max_length:
            waveform = waveform[:self.max_length]
        elif seq_len < self.max_length:
            pad_amount = self.max_length - seq_len
            # F.pad on a 1D tensor adds to the right side of the sequence
            waveform = F.pad(waveform, (0, pad_amount))
        
        return {
            "input_values": waveform,
            "labels": labels
        }