# 🚀 QUICK START GUIDE

Get up and running in 5 minutes!

## For Complete Beginners

### Step 1: Download This Folder

Click the green "Code" button → "Download ZIP" → Extract the folder

### Step 2: Open Terminal

- **macOS**: Press `Cmd + Space`, type "Terminal", press Enter
- **Windows**: Press `Win + R`, type "cmd", press Enter
- **Linux**: Press `Ctrl + Alt + T`

### Step 3: Navigate to the Folder

```bash
cd path/to/srk-voice-clone-demo
```

(Replace `path/to` with where you extracted the folder)

### Step 4: Run Setup (First Time Only)

```bash
./setup.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 5: Run the Demo

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python voice_clone.py
```

### Step 6: Listen to the Result

```bash
open output/cloned_srk_voice.wav
```

---

## For Experienced Users

```bash
# One-liner setup and run
git clone <your-repo> && cd srk-voice-clone-demo && \
python3 -m venv venv && source venv/bin/activate && \
pip install -r requirements.txt && python voice_clone.py
```

---

## What Happens When You Run It?

1. **First run** (10-15 minutes):
   - Downloads Qwen3-TTS model (~2.5GB)
   - Loads model into memory
   - Clones SRK's voice
   - Generates new speech
   - Saves to `output/cloned_srk_voice.wav`

2. **Subsequent runs** (30 seconds):
   - Uses cached model (no download)
   - Generates speech immediately

---

## Customize It

Edit `voice_clone.py` and change these lines:

```python
# Line 17: Your reference audio
REFERENCE_AUDIO = "input/your_audio.wav"

# Line 20: What's said in your reference
REFERENCE_TEXT = """Your reference transcript here"""

# Line 26: New text to generate
NEW_TEXT = """Your new text here"""
```

Then run: `python voice_clone.py`

---

## Need Help?

### Common Issues

**"ModuleNotFoundError"**
→ Did you activate the environment? Run: `source venv/bin/activate`

**"CUDA out of memory"**
→ Edit `voice_clone.py` line 47: Change `cuda:0` to `cpu`

**"File not found"**
→ Make sure you're in the correct folder: `cd srk-voice-clone-demo`

**"Permission denied"**
→ macOS/Linux: Run `chmod +x setup.sh` first

### Still Stuck?

1. Read the full [README.md](README.md)
2. Check your Python version: `python3 --version` (need 3.9+)
3. Check your internet connection (first run downloads model)
4. Open an issue on GitHub

---

## What's in This Folder?

```
📁 srk-voice-clone-demo/
├── 📄 README.md              ← Full documentation
├── 📄 QUICKSTART.md          ← This file
├── 📄 requirements.txt       ← Dependencies list
├── 📄 LICENSE                ← MIT License
├── 🐍 voice_clone.py         ← Main script (RUN THIS!)
├── 🔧 setup.sh               ← Auto setup script
├── 📁 input/                 ← Reference audio files
│   └── 🎵 srk_reference_19s.wav
└── 📁 output/                ← Generated audio files
    └── 🎵 cloned_srk_voice.wav
```

---

## Next Steps

After your first successful run:

1. ✅ Try with your own voice
2. ✅ Experiment with different texts
3. ✅ Share your results!
4. ✅ Star the repo if you found it helpful ⭐

---

**Happy Voice Cloning! 🎤**

Questions? Open an issue or read the full README.md
