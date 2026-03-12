# Shah Rukh Khan Voice Cloning Demo - CHANGELOG

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-03-12

### Added

- Initial release of SRK voice cloning demo
- Voice cloning script with Qwen3-TTS-12Hz-0.6B-Base model
- Sample reference audio (19 seconds from SRK interview)
- Sample output audio demonstrating voice cloning
- Comprehensive README with setup instructions
- Requirements.txt for easy dependency management
- Automated setup script (setup.sh)
- Error handling and user-friendly console output
- Support for both GPU and CPU inference
- .gitignore for clean repository
- MIT License

### Features

- Clone Shah Rukh Khan's voice from reference audio
- Generate new speech in cloned voice
- Auto-detect language from input
- High-quality 24kHz audio output
- Progress indicators and status messages
- Detailed error messages with troubleshooting tips

### Technical Details

- Model: Qwen3-TTS-12Hz-0.6B-Base
- Audio format: WAV, 24kHz, 16-bit PCM
- Supported languages: 10+ (auto-detection enabled)
- Platform: Cross-platform (Windows, macOS, Linux)
- Python: 3.9+ required, 3.12 recommended

### Documentation

- README.md with complete setup guide
- Inline code comments for clarity
- Troubleshooting section for common issues
- Customization guide for using your own voice

### Future Plans

- [ ] Add more example reference audios
- [ ] Create Jupyter notebook version
- [ ] Add batch processing support
- [ ] Web interface option
- [ ] Docker container for easy deployment
- [ ] Support for multiple voices in one script
- [ ] Real-time voice cloning demo

---

## How to Contribute

Found a bug or want to add a feature? Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Alibaba Qwen Team for the Qwen3-TTS model
- Open source community for inspiration and tools
