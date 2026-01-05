# CSM Lester Voice - Text-to-Speech Generator

A Python script that generates high-quality speech audio using the fine-tuned [lesterthomas/csm-lester-voice](https://huggingface.co/lesterthomas/csm-lester-voice) model based on sesame/csm-1b.

## Features

- 🎙️ High-quality 24kHz speech generation
- 📝 Accepts text via command-line or file input
- 🔄 Automatic dependency checking with helpful error messages
- 🚀 GPU acceleration support (CUDA)
- 📦 Managed with uv package manager

## Installation

### Prerequisites

- Python 3.9 or higher
- [uv package manager](https://github.com/astral-sh/uv)

### Setup

1. Clone or navigate to this repository:
```bash
cd d:\Dev\lesterthomas\csm-lester-voice
```

2. Install dependencies using uv:
```bash
uv sync
```

This will install all required packages:
- `transformers>=4.52.1` - For the CSM model
- `torch>=2.0.0` - Deep learning framework
- `soundfile>=0.12.0` - Audio file I/O
- `accelerate>=0.20.0` - Model device management
- `datasets>=2.14.0` - Dataset utilities

## Usage

### Basic Examples

**Generate from command-line text:**
```bash
python generate_speech.py -t "Hello, this is a test of text to speech."
```

**Generate from text file:**
```bash
python generate_speech.py -f input.txt
```

**Specify custom output location:**
```bash
python generate_speech.py -t "Welcome to CSM TTS" -o audio/welcome.wav
```

**Generate from file with custom output:**
```bash
python generate_speech.py -f script.txt -o narration.wav
```

### Command-Line Options

```
usage: generate_speech.py [-h] (-t TEXT | -f FILE) [-o OUTPUT] [-m MODEL]
                          [--no-speaker-tag] [--device {cuda,cpu}]

required arguments:
  -t, --text TEXT       Text to convert to speech
  -f, --file FILE       Path to text file containing input text

optional arguments:
  -o, --output OUTPUT   Output audio file path (default: speech_TIMESTAMP.wav)
  -m, --model MODEL     HuggingFace model ID 
                        (default: lesterthomas/csm-lester-voice)
  --no-speaker-tag      Don't automatically add [0] speaker tag to text
  --device {cuda,cpu}   Device to run model on (default: auto-detect)
```

### Output Format

- **Format:** WAV (uncompressed)
- **Sampling Rate:** 24,000 Hz (24 kHz)
- **Channels:** Mono
- **Default filename:** `speech_YYYYMMDD_HHMMSS.wav`

## How It Works

1. **Text Preparation:** The script automatically adds a `[0]` speaker tag to your text (unless `--no-speaker-tag` is specified). This tells the CSM model which voice to use.

2. **Model Loading:** Downloads and caches the lesterthomas/csm-lester-voice model from HuggingFace (first run only).

3. **Audio Generation:** Uses the CSM (Character-level Speech Model) to generate high-quality speech at 24kHz sampling rate.

4. **Saving:** Outputs a WAV file compatible with all audio players and editing software.

## Requirements

### Hardware

- **Minimum:** CPU-only (slower generation)
- **Recommended:** NVIDIA GPU with CUDA support (faster generation)

### First-Time Setup

**Important:** Before using this script, ensure your model is properly uploaded to HuggingFace:

1. **Upload model files:** Your fine-tuned model at [lesterthomas/csm-lester-voice](https://huggingface.co/lesterthomas/csm-lester-voice) needs the model weights (`.safetensors` or `.bin` files) uploaded. 

2. **Base model access:** The CSM models require accepting terms:
   - Accept terms for [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) (base model)
   - Your fine-tuned model inherits these requirements

3. **HuggingFace authentication:** Log in to access gated models:
   ```bash
   huggingface-cli login
   ```

#### Testing Before Model Upload

If your model files aren't uploaded yet, you can test the script with any publicly available TTS model:
```bash
# Test with a different model (example)
python generate_speech.py -t "Hello world" -m "microsoft/speecht5_tts"
```

## Troubleshooting

### Missing Dependencies

If you see a "Missing required dependencies" error:
```bash
uv sync
```

### Model Loading Error

If the model fails to load:
- Ensure you've accepted the model terms on HuggingFace
- Check your internet connection (first-time download required)
- Verify HuggingFace authentication if model requires it

### Out of Memory

If you encounter memory errors:
- Use CPU mode: `--device cpu`
- Close other applications
- Try shorter text inputs

## Examples

Create a sample text file:
```bash
echo "The quick brown fox jumps over the lazy dog. This is a test of the character-level speech model." > sample.txt
```

Generate speech:
```bash
python generate_speech.py -f sample.txt -o output.wav
```

## Model Information

- **Base Model:** [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)
- **Fine-tuned Model:** [lesterthomas/csm-lester-voice](https://huggingface.co/lesterthomas/csm-lester-voice)
- **Architecture:** Character-level Speech Model with Mimi codec
- **License:** Apache 2.0

## License

This project uses models licensed under Apache 2.0. See the HuggingFace model pages for full license details.
