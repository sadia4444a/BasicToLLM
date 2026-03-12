"""
Voice Cloning Demo using Qwen3-TTS
===================================
Demonstrates voice cloning by cloning Shah Rukh Khan's voice from a reference
audio sample and generating new speech in the cloned voice.

Model: Qwen3-TTS-12Hz-0.6B-Base
License: Apache 2.0
"""

import os
import sys
import logging
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
REFERENCE_AUDIO = "input/srk_reference_19s.wav"
OUTPUT_AUDIO = "output/cloned_srk_voice.wav"

REFERENCE_TEXT = """Failure is an amazing teacher. There's a well-known story of this very successful man, 
and a reporter asked him Sir how is it that you always succeed? And he said, Right decisions. 
The reporter asked, So how do you make right decisions? And he said, You know… experience. 
How do you get so experienced? It's the wrong decisions and failure."""

NEW_TEXT = """You are someone who has failed so many times, but you never give up. 
You keep trying and trying until you succeed. Failure is not the opposite of success, 
it's part of success. So embrace failure, learn from it, and keep moving forward."""


def validate_inputs():
    """Validate that required input files exist."""
    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}")
        logger.error("Ensure the reference audio file is placed in the 'input/' directory")
        sys.exit(1)
    logger.info(f"Reference audio validated: {REFERENCE_AUDIO}")


def initialize_model():
    """Initialize and load the Qwen3-TTS model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    logger.info("Initializing Qwen3-TTS model...")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map=device,
        dtype=dtype,
    )
    
    logger.info("Model loaded successfully")
    return model


def generate_cloned_voice(model):
    """Generate voice-cloned audio from the model."""
    logger.info("Starting voice cloning process...")
    logger.info(f"Input text length: {len(NEW_TEXT)} characters")
    
    wavs, sr = model.generate_voice_clone(
        text=NEW_TEXT,
        language="Auto",
        ref_audio=REFERENCE_AUDIO,
        ref_text=REFERENCE_TEXT,
    )
    
    logger.info("Voice generation completed")
    return wavs, sr


def save_output(wavs, sr):
    """Save generated audio to file."""
    os.makedirs(os.path.dirname(OUTPUT_AUDIO), exist_ok=True)
    sf.write(OUTPUT_AUDIO, wavs[0], sr)
    
    duration = len(wavs[0]) / sr
    file_size = os.path.getsize(OUTPUT_AUDIO) / (1024 * 1024)
    
    logger.info(f"Audio saved: {OUTPUT_AUDIO}")
    logger.info(f"Duration: {duration:.2f}s | Sample rate: {sr}Hz | Size: {file_size:.2f}MB")


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Shah Rukh Khan Voice Cloning Demo")
    logger.info("=" * 60)
    
    validate_inputs()
    model = initialize_model()
    wavs, sr = generate_cloned_voice(model)
    save_output(wavs, sr)
    
    logger.info("=" * 60)
    logger.info("Process completed successfully")
    logger.info(f"Play audio: open {OUTPUT_AUDIO}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        logger.error("Troubleshooting:")
        logger.error("  1. Install dependencies: pip install -r requirements.txt")
        logger.error("  2. Verify reference audio exists in input/ directory")
        logger.error("  3. Check internet connection for model download")
        sys.exit(1)
