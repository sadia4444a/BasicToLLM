# 🎤 Shah Rukh Khan Voice Cloning Demo with Qwen3-TTS

This project demonstrates AI-powered voice cloning using Alibaba's Qwen3-TTS model. Clone Shah Rukh Khan's voice (or any voice) and generate new speech that sounds like him!

## 🌟 Features

- ✅ Clone any voice from just 19 seconds of reference audio
- ✅ Generate natural-sounding speech in the cloned voice
- ✅ Support for multiple languages (auto-detection)
- ✅ High-quality 24kHz audio output
- ✅ GPU acceleration support (CPU fallback available)
- ✅ Easy to use and well-documented

## 📋 Prerequisites

- **Python**: 3.9 or higher (3.12 recommended)
- **Hardware**:
  - Recommended: GPU with 8GB+ VRAM (NVIDIA with CUDA)
  - Minimum: 8GB RAM for CPU inference (slower)
- **Disk Space**: ~3GB for model download
- **Internet**: Required for first-time model download

## 🚀 Quick Start

### Step 1: Clone or Download This Repository

```bash
git clone <your-repo-url>
cd srk-voice-clone-demo
```

Or download and extract the ZIP file.

### Step 2: Set Up Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda create -n voice-clone python=3.12 -y

# Activate environment
conda activate voice-clone
```

#### Option B: Using venv

```bash
# Create environment
python3 -m venv venv

# Activate environment (macOS/Linux)
source venv/bin/activate

# Activate environment (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional: Install FlashAttention for better performance** (requires compatible GPU):

```bash
pip install flash-attn --no-build-isolation
```

### Step 4: Run the Demo

```bash
python voice_clone.py
```

**First run**: The model (~2.5GB) will download automatically. This takes 5-10 minutes.

**Subsequent runs**: Model loads instantly from cache.

### Step 5: Listen to the Result

```bash
# macOS
open output/cloned_srk_voice.wav

# Linux
xdg-open output/cloned_srk_voice.wav

# Windows
start output/cloned_srk_voice.wav
```

## 📁 Project Structure

```
srk-voice-clone-demo/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── voice_clone.py                     # Main script
├── input/                             # Input audio files
│   └── srk_reference_19s.wav         # Reference audio (19 seconds)
└── output/                            # Generated audio files
    └── cloned_srk_voice.wav          # Output (created after running)
```

## 🎯 How It Works

### The Process

1. **Reference Audio**: Provide a short clip (3-20 seconds) of the target voice
2. **Reference Transcript**: Provide the exact text spoken in the reference
3. **New Text**: Write the text you want to generate in the cloned voice
4. **Generation**: The model analyzes the reference and generates new speech

### Technical Details

- **Model**: Qwen3-TTS-12Hz-0.6B-Base
- **Technology**: Discrete multi-codebook language model
- **Audio Quality**: 24kHz, 16-bit PCM WAV
- **Languages Supported**: 10+ (Chinese, English, Japanese, Korean, etc.)

## 🔧 Customization

### Use Your Own Voice

1. **Extract reference audio** (3-20 seconds):

   ```bash
   ffmpeg -i your_video.mp4 -t 10 -vn -acodec pcm_s16le -ar 24000 reference.wav
   ```

2. **Edit `voice_clone.py`**:

   ```python
   REFERENCE_AUDIO = "input/your_reference.wav"
   REFERENCE_TEXT = "Exact transcript of your reference audio"
   NEW_TEXT = "Your new text to generate"
   ```

3. **Run the script**:
   ```bash
   python voice_clone.py
   ```

### Adjust Generation Parameters

Modify the `generate_voice_clone()` call in `voice_clone.py`:

```python
wavs, sr = model.generate_voice_clone(
    text=NEW_TEXT,
    language="English",          # Specify language explicitly
    ref_audio=REFERENCE_AUDIO,
    ref_text=REFERENCE_TEXT,
    max_new_tokens=2048,         # Control output length
    temperature=0.9,             # Control randomness (0.1-2.0)
    top_k=50,                    # Sampling diversity
    top_p=1.0,                   # Nucleus sampling
)
```

## 📊 Sample Results

### Input

- **Reference**: 19-second clip from Shah Rukh Khan interview
- **Reference Text**: "Failure is an amazing teacher..."
- **New Text**: "You are someone who has failed so many times..."

### Output

- **File**: `output/cloned_srk_voice.wav`
- **Duration**: ~30 seconds (depends on text length)
- **Quality**: High-fidelity voice clone

## ⚙️ System Requirements

### Minimum (CPU)

- 8GB RAM
- 3GB disk space
- ~2 minutes per 10 seconds of audio

### Recommended (GPU)

- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- 16GB RAM
- ~10 seconds per 10 seconds of audio

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Solution**: Activate your Python environment

```bash
conda activate voice-clone  # or: source venv/bin/activate
```

### "CUDA out of memory"

**Solution**: Use CPU instead

```python
# In voice_clone.py, change:
device_map="cpu"
```

### "Model download is slow"

**Solution**: Wait for the first download. The model is cached and won't download again.

### "Audio quality is poor"

**Solution**:

- Use higher quality reference audio (16kHz+)
- Ensure accurate reference transcript
- Use longer reference clip (10-20 seconds optimal)

### "Import error: flash_attn"

**Solution**: FlashAttention is optional. Either:

1. Remove `attn_implementation="flash_attention_2"` from code
2. Or install: `pip install flash-attn --no-build-isolation`

## 📚 Additional Resources

- **Qwen3-TTS GitHub**: https://github.com/QwenLM/Qwen3-TTS
- **Paper**: https://arxiv.org/abs/2601.15621
- **Hugging Face Models**: https://huggingface.co/collections/Qwen/qwen3-tts
- **Official Blog**: https://qwen.ai/blog?id=qwen3tts-0115

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:

- Open an issue
- Submit a pull request
- Share your results!

## ⚖️ License & Ethics

### Model License

- Qwen3-TTS is licensed under Apache 2.0
- Free for commercial and non-commercial use

### Ethical Use

⚠️ **Important**: This technology should be used responsibly:

- ✅ DO: Use for creative projects, accessibility, education
- ✅ DO: Get permission before cloning someone's voice
- ✅ DO: Disclose when audio is AI-generated
- ❌ DON'T: Create deepfakes or misleading content
- ❌ DON'T: Clone voices without consent
- ❌ DON'T: Use for fraud, impersonation, or harmful purposes

**Remember**: With great power comes great responsibility! 🕷️

## 📧 Contact

Created by: Sadia Khatun
Date: March 12, 2026

For questions or collaborations, open an issue or reach out!

## 🎉 Acknowledgments

- **Alibaba Qwen Team** for the amazing Qwen3-TTS model
- **Shah Rukh Khan** for the inspiration
- **Open Source Community** for making AI accessible

---

**Happy Voice Cloning! 🎤✨**

If you found this helpful, please ⭐ star the repository!
