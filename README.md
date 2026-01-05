# CSM Lester Voice - Voice Cloning with Text-to-Speech

A Python script that generates high-quality speech audio using voice cloning with the fine-tuned [lesterthomas/csm-lester-voice](https://huggingface.co/lesterthomas/csm-lester-voice) model based on Sesame CSM-1B.

## Features

- 🎙️ **Voice Cloning:** Generate speech that matches a reference voice
- 🎯 **High-Quality Audio:** 24kHz speech generation
- 📝 Text-to-speech with voice style consistency
- 🚀 GPU acceleration support (CUDA) with CPU fallback
- 📦 Managed with uv package manager
- 🔧 Automatic bug fixes for transformers 4.52.3

## Installation

### Prerequisites

- Python 3.9 or higher
- [uv package manager](https://github.com/astral-sh/uv)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/LesterThomas/csm-lester-voice.git
cd csm-lester-voice
```

2. Install dependencies using uv:
```bash
uv sync
```

This will install all required packages:
- `transformers==4.52.3` - HuggingFace transformers library
- `datasets==4.3.0` - Dataset utilities
- `torch>=2.0.0` - Deep learning framework
- `soundfile>=0.12.0` - Audio file I/O
- `accelerate>=0.20.0` - Model device management
- `librosa>=0.10.0` - Audio resampling

## Usage

### Voice Cloning Inference

The script uses reference audio to clone voice characteristics and speaking style.

**Basic usage:**
```bash
uv run inference_voice_clone.py -t "Your text to synthesize"
```

**Full example with custom files:**
```bash
uv run inference_voice_clone.py \
  -t "Hello world, this is my cloned voice!" \
  -r my-voice-sample.wav \
  -rt my-voice-sample.txt \
  -o output.wav
```

### Command-Line Options

```
usage: inference_voice_clone.py [-h] -t TEXT [-m MODEL] [-r REFERENCE_AUDIO] 
                                [-rt REFERENCE_TEXT] [-o OUTPUT] 
                                [--max-tokens MAX_TOKENS] [--speaker-id SPEAKER_ID]

required arguments:
  -t, --text TEXT              Text to synthesize (required)

optional arguments:
  -m, --model MODEL            Path to model (default: lesterthomas/csm-lester-voice)
  -r, --reference-audio FILE   Reference audio file (default: reference-audio.wav)
  -rt, --reference-text FILE   Reference text file (default: reference-utterance.txt)
  -o, --output FILE            Output audio file (default: cloned_output.wav)
  --max-tokens INT             Max tokens to generate, 125 ≈ 10 seconds (default: 125)
  --speaker-id INT             Speaker ID (default: 0)
```

### Reference Files

The script requires reference files to clone voice characteristics:

1. **reference-audio.wav** - A 24kHz audio sample of the target voice (included)
2. **reference-utterance.txt** - The exact transcript of the reference audio (included)

You can use your own reference files by:
- Recording or obtaining a clear audio sample (WAV format preferred)
- Creating a text file with the exact transcript
- Passing them via `-r` and `-rt` arguments

### Output Format

- **Format:** WAV (uncompressed)
- **Sampling Rate:** 24,000 Hz (24 kHz)
- **Channels:** Mono
- **Default filename:** `cloned_output.wav`

## How It Works

1. **Voice Reference:** The model analyzes the reference audio and text to understand voice characteristics and speaking style

2. **Model Loading:** Downloads and loads the fine-tuned CSM model from HuggingFace (cached after first run)

3. **Voice Cloning:** Generates new speech using the reference voice characteristics while speaking the input text

4. **Audio Generation:** Produces high-quality 24kHz audio with voice consistency

5. **Bug Fixes:** Automatically applies patches for known issues in transformers 4.52.3

## Examples

### Quick Start

Generate speech with the included reference voice:
```bash
uv run inference_voice_clone.py -t "The quick brown fox jumps over the lazy dog."
```

### Custom Output Location

```bash
uv run inference_voice_clone.py \
  -t "Welcome to voice cloning with CSM!" \
  -o greetings/welcome.wav
```

### Longer Speech

For longer text (adjust max-tokens accordingly, 125 tokens ≈ 10 seconds):
```bash
uv run inference_voice_clone.py \
  -t "This is a longer sentence that requires more time to synthesize properly." \
  --max-tokens 200 \
  -o long_speech.wav
```

### Using Your Own Voice

```bash
uv run inference_voice_clone.py \
  -t "This will sound like my own voice!" \
  -r path/to/your-voice.wav \
  -rt path/to/your-transcript.txt \
  -o my_cloned_voice.wav
```

## Requirements

### Hardware

- **Minimum:** CPU (slower generation, ~30-60 seconds per 10s audio)
- **Recommended:** NVIDIA GPU with CUDA support (faster, ~5-10 seconds per 10s audio)

### Model Access

1. **HuggingFace Account:** Create a free account at [huggingface.co](https://huggingface.co)

2. **Model Access:** The model will be automatically downloaded from HuggingFace on first run

3. **Authentication (if needed):** For private models or gated access:
   ```bash
   huggingface-cli login
   ```

## Troubleshooting

### Missing Dependencies

If you encounter import errors:
```bash
uv sync
```

### Model Loading Error

**Issue:** Model fails to load
- Check internet connection (first-time download required)
- Verify HuggingFace model exists at the specified path
- Ensure you're logged in if using a private model

### Out of Memory

**Issue:** CUDA out of memory or system memory error
- The model requires ~2-4GB VRAM (GPU) or ~4-8GB RAM (CPU)
- Close other applications
- Try shorter text inputs
- Reduce `--max-tokens` value

### Audio Quality Issues

**Issue:** Generated speech sounds unclear or distorted
- Ensure reference audio is clean and clear
- Use 24kHz reference audio when possible
- Verify reference text exactly matches the audio
- Try a longer or different reference sample

### Generation Warnings

The script may show warnings like:
```
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']
```
**This is normal** - these warnings don't affect output quality and can be safely ignored.

### Slow Generation on CPU

**Issue:** Generation takes a long time
- This is expected on CPU (30-60 seconds for 10 seconds of audio)
- Consider using a GPU/cloud instance for faster generation
- Generate shorter segments and concatenate them

## Technical Details

### Model Information

- **Base Model:** [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)
- **Fine-tuned Model:** [lesterthomas/csm-lester-voice](https://huggingface.co/lesterthomas/csm-lester-voice)
- **Architecture:** Character-level Speech Model (CSM) with Mimi codec
- **Training:** Fine-tuned using Unsloth with LoRA adapters
- **Voice Cloning:** Uses reference audio context for speaker consistency

### Known Issues & Fixes

The script automatically patches a bug in transformers 4.52.3 where `pad_to_multiple_of` is incorrectly passed to `EncodecFeatureExtractor`. This patch is applied at startup.

### Audio Processing

- Input reference audio is automatically resampled to 24kHz if needed (requires librosa)
- Output is always mono channel at 24kHz
- Generation uses temperature-based sampling for natural speech

## Development

### Project Structure

```
csm-lester-voice/
├── inference_voice_clone.py    # Main inference script
├── reference-audio.wav          # Sample reference audio
├── reference-utterance.txt      # Sample reference transcript
├── pyproject.toml              # Python dependencies (uv)
├── uv.lock                     # Locked dependencies
└── README.md                   # This file
```

### Running Without uv

If you prefer pip:
```bash
pip install transformers==4.52.3 datasets==4.3.0 torch soundfile librosa accelerate
python inference_voice_clone.py -t "Your text here"
```

## License

This project is open source. The models used are licensed under Apache 2.0. See the HuggingFace model pages for full license details.

## Credits

- **Base Model:** Sesame CSM-1B by Cartesia
- **Fine-tuning Framework:** Unsloth AI
- **Notebook Template:** Adapted from Unsloth's CSM TTS notebook
- **Workshop:** Trelis Research AI World's Fair 2025

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/LesterThomas/csm-lester-voice)
- Check the [Unsloth Discord](https://discord.gg/unsloth) for CSM model help
- Review [transformers documentation](https://huggingface.co/docs/transformers)
