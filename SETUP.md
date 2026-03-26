# Contributing to FatigueSense Dataset

This guide covers how to set up access to the Hugging Face repository and push/pull data.

---

## Authentication

There are several ways to authenticate with Hugging Face. Pick whichever suits your workflow — you only need one.

---

### Option 1: Access Token (Easiest)

Access tokens work over HTTPS and are the quickest way to get started.

**Step 1 — Generate a token:**

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Give it a name (e.g., `fatigue-sense`)
4. Set role to **Write** (so you can push)
5. Click **Generate a token** and copy it somewhere safe

**Step 2 — Configure Git to use your token:**

```bash
git config --global credential.helper store
```

**Step 3 — Clone the repo (you'll be prompted for credentials):**

```bash
git clone https://huggingface.co/datasets/Jlords32/FatigueSense
```

When prompted:
- **Username:** your Hugging Face username
- **Password:** paste your access token (not your account password)

Your credentials will be saved and you won't be asked again.

---

### Option 2: SSH Key

SSH is more secure and doesn't require entering credentials on every operation.

**Step 1 — Generate an SSH key (skip if you already have one):**

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept the default file location. Optionally set a passphrase.

**Step 2 — Copy your public key:**

```bash
# Mac/Linux
cat ~/.ssh/id_ed25519.pub

# Windows (PowerShell)
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

Copy the entire output.

**Step 3 — Add the key to Hugging Face:**

1. Go to [https://huggingface.co/settings/keys](https://huggingface.co/settings/keys)
2. Click **Add SSH key**
3. Paste your public key and save

**Step 4 — Clone the repo using SSH:**

```bash
git clone git@hf.co:datasets/Jlords32/FatigueSense
```

**Step 5 — Verify the connection:**

```bash
ssh -T git@hf.co
```

You should see a greeting message confirming it worked.

---

### Option 3: CLI Login (For Python Scripts)

If you're using the Hugging Face Python SDK (e.g., `HfApi` in `upload_dataset.py`), you need to log in via the CLI. This caches your token locally so the SDK can authenticate automatically.

**Step 1 — Install the `huggingface_hub` package (if not already installed):**

```bash
pip install huggingface_hub
```

**Step 2 — Log in:**

```bash
hf auth login
```

When prompted, paste your access token (see [Option 1, Step 1](#option-1-access-token-easiest) to generate one).

Your token is saved to `~/.cache/huggingface/token` and will be used automatically by `HfApi`, `hf_hub_download`, and other SDK functions.

**Step 3 — Verify you're logged in:**

```bash
hf auth whoami
```

This prints your username and organizations. If you're not logged in, you'll see an error.

---

### Option 4: Notebook Login (For Jupyter Notebooks)

If you're working in a Jupyter notebook, you can log in interactively with a widget:

```python
from huggingface_hub import notebook_login
notebook_login()
```

This displays a text field where you paste your access token. It's saved to the same location as the CLI login, so the SDK will pick it up automatically.

Alternatively, if you prefer not to use the widget, you can log in programmatically:

```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

> **Tip:** Avoid hardcoding tokens in notebooks that might be committed. Use environment variables or a `.env` file instead:
> ```python
> import os
> from huggingface_hub import login
> login(token=os.environ["HF_TOKEN"])
> ```

---

## Pushing and Pulling Data

Once you've cloned the repo using either method above:

### Pull latest changes

Always pull before you start working to avoid conflicts:

```bash
cd FatigueSense
git pull origin main
```

### Download via Python (using `huggingface_hub`)

You can also pull data using the Python SDK instead of Git.

**Download the entire dataset:**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Jlords32/fatigue-sense",
    repo_type="dataset",
    local_dir="./dataset",
)
```

**Download a single file:**

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Jlords32/fatigue-sense",
    filename="data/train.csv",
    repo_type="dataset",
    local_dir="./dataset",
)
```

**Download only specific file types:**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Jlords32/fatigue-sense",
    repo_type="dataset",
    local_dir="./dataset",
    allow_patterns=["*.csv", "*.parquet"],
)
```

> **Note:** You must be logged in first (see [Option 3: CLI Login](#option-3-cli-login-for-python-scripts) or [Option 4: Notebook Login](#option-4-notebook-login-for-jupyter-notebooks)). If omitting `local_dir`, files are cached to `~/.cache/huggingface/hub/`.

> **Why not `load_dataset`?** The Hugging Face `datasets` library provides a `load_dataset()` function, but it expects structured formats like CSV, JSON, Parquet, or HF-native layouts (e.g., `ImageFolder`). Our dataset uses **YOLO object detection format** (images + `.txt` label files with bounding box coordinates), which `load_dataset` doesn't understand out of the box. Use `snapshot_download` to pull the raw files instead, and load them with Ultralytics/YOLO for training.

### Add and push your changes

```bash
# 1. Stage your files
git add .

# 2. Commit with a descriptive message
git commit -m "Add session 4 EEG recordings"

# 3. Push to Hugging Face
git push origin main
```

### Upload via Python (using `HfApi`)

You can also push data directly from a Python script using the Hugging Face SDK — no Git commands needed. This is especially useful for large uploads. See `upload_dataset.py` for an example:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_large_folder(
    folder_path="./dataset",
    repo_id="Jlords32/fatigue-sense",
    repo_type="dataset",
)
```

> **Note:** You must be logged in first (see [Option 3: CLI Login](#option-3-cli-login-for-python-scripts) or [Option 4: Notebook Login](#option-4-notebook-login-for-jupyter-notebooks)).

To run the existing script:

```bash
python upload_dataset.py
```

---

### Push a large file (using Git LFS)

Hugging Face uses Git LFS for large files (datasets, models). It's handled automatically when you clone from Hugging Face, but make sure it's installed:

```bash
# Check if installed
git lfs version

# If not installed: https://git-lfs.com
# Then track large file types
git lfs track "*.csv"
git lfs track "*.parquet"

# Commit the .gitattributes file that was updated
git add .gitattributes
git commit -m "Track large files with LFS"
git push origin main
```
   
---

## Quick Reference

| Task | Command |
|---|---|
| Clone (HTTPS) | `git clone https://huggingface.co/datasets/Jlords32/FatigueSense` |
| Clone (SSH) | `git clone git@hf.co:datasets/Jlords32/FatigueSense` |
| CLI login | `hf auth login` |
| Verify login | `hf auth whoami` |
| Notebook login | `from huggingface_hub import notebook_login; notebook_login()` |
| Pull latest | `git pull origin main` |
| Download via Python | `snapshot_download(repo_id="Jlords32/fatigue-sense", repo_type="dataset", local_dir="./dataset")` |
| Stage all changes | `git add .` |
| Commit | `git commit -m "your message"` |
| Push | `git push origin main` |
| Upload via Python | `python upload_dataset.py` |
| Check status | `git status` |
