"""
Voice cloning inference script using CSM model with reference audio.
Based on the 'Voice and style consistency' section from the notebook.
"""

import torch
import soundfile as sf
import warnings
import numpy as np
from transformers import CsmForConditionalGeneration, AutoProcessor
from transformers.utils import logging as transformers_logging
import argparse

# Suppress transformers warnings about generation flags
# The CSM model internally uses some parameters that aren't applicable to audio generation
# but this doesn't affect functionality - the audio is generated correctly
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


# Monkey-patch to fix the pad_to_multiple_of issue in transformers 4.52.3
def patch_csm_processor():
    """Apply a monkey-patch to fix the pad_to_multiple_of bug in CSM processor."""
    try:
        from transformers.models.encodec import feature_extraction_encodec
        original_call = feature_extraction_encodec.EncodecFeatureExtractor.__call__
        
        def patched_call(self, *args, **kwargs):
            # Remove pad_to_multiple_of if present (not supported by EncodecFeatureExtractor)
            kwargs.pop('pad_to_multiple_of', None)
            return original_call(self, *args, **kwargs)
        
        feature_extraction_encodec.EncodecFeatureExtractor.__call__ = patched_call
        print("Applied EncodecFeatureExtractor patch for pad_to_multiple_of issue")
    except Exception as e:
        print(f"Warning: Could not apply processor patch: {e}")


patch_csm_processor()


def check_cuda_setup():
    """Display CUDA setup information."""
    print("\n=== CUDA Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n⚠ CUDA not detected. To enable GPU acceleration:")
        print("  1. Verify NVIDIA drivers are installed")
        print("  2. Install PyTorch with CUDA:")
        print("     uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("========================\n")


def load_model_and_processor(model_path, force_cuda=False):
    """Load the fine-tuned model and processor."""
    print(f"Loading model from {model_path}...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA is available: {device_name}")
    else:
        print("✗ CUDA is not available")
        if force_cuda:
            print("\nERROR: CUDA forced but not available!")
            print("Install PyTorch with CUDA support:")
            print("  uv pip install torch --index-url https://download.pytorch.org/whl/cu121")
            raise RuntimeError("CUDA not available but was forced")
    
    # Determine device and dtype
    use_cuda = cuda_available or force_cuda
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    
    print(f"Loading model with {dtype} on {device}...")
    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    if use_cuda:
        print(f"✓ Model loaded on CUDA ({device_name})")
    else:
        print("Model loaded on CPU (slower)")
    
    return model, processor


def trim_silence(audio, sample_rate=24000, threshold_db=-40, chunk_size=0.1):
    """
    Remove trailing silence from audio.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
        threshold_db: Silence threshold in dB (default -40dB)
        chunk_size: Chunk size in seconds to analyze (default 0.1s)
    
    Returns:
        Trimmed audio array
    """
    # Convert threshold from dB to amplitude
    threshold = 10 ** (threshold_db / 20)
    
    # Calculate chunk size in samples
    chunk_samples = int(sample_rate * chunk_size)
    
    # Find the last non-silent chunk
    for i in range(len(audio) - chunk_samples, 0, -chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if np.abs(chunk).max() > threshold:
            # Found non-silent audio, trim here (add small buffer)
            trim_point = min(i + chunk_samples + int(sample_rate * 0.2), len(audio))
            return audio[:trim_point]
    
    # If all silent (unlikely), return original
    return audio


def generate_speech_with_reference(
    model,
    processor,
    text_to_speak,
    reference_audio_path,
    reference_text_path,
    output_path="cloned_output.wav",
    speaker_id=0,
    max_new_tokens=None,
    depth_decoder_temperature=0.6,
    depth_decoder_top_k=0,
    depth_decoder_top_p=0.9,
):
    """Generate speech using reference audio for voice cloning.
    
    Note: The model stops generation at periods. For multi-sentence text,
    each sentence is generated separately and concatenated.
    
    Args:
        max_new_tokens: Max audio tokens to generate per sentence. If None, automatically 
                       calculated based on text length (~8 tokens per word). Default: None.
    """
    
    # Load reference audio
    print(f"Loading reference audio from {reference_audio_path}...")
    utterance, sample_rate = sf.read(reference_audio_path)
    
    # Resample if needed (CSM expects 24kHz)
    if sample_rate != 24000:
        print(f"Resampling audio from {sample_rate}Hz to 24000Hz...")
        import librosa
        utterance = librosa.resample(utterance, orig_sr=sample_rate, target_sr=24000)
    
    # Load reference text
    print(f"Loading reference text from {reference_text_path}...")
    with open(reference_text_path, 'r', encoding='utf-8') as f:
        utterance_text = f.read().strip()
    
    print(f"Reference text: {utterance_text}")
    print(f"Text to synthesize: {text_to_speak}")
    
    # Split text by sentences (the model stops at periods)
    import re
    sentences = [s.strip() + '.' for s in text_to_speak.split('.') if s.strip()]
    
    if len(sentences) > 1:
        print(f"Detected {len(sentences)} sentences - generating each separately...")
    
    # Generate audio for each sentence
    all_audio_segments = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    for i, sentence in enumerate(sentences, 1):
        if len(sentences) > 1:
            print(f"  Generating sentence {i}/{len(sentences)}: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
        
        # Auto-calculate max_new_tokens for this sentence if not specified
        if max_new_tokens is None:
            word_count = len(sentence.split())
            sentence_max_tokens = max(125, word_count * 8)
        elif max_new_tokens == -1:
            sentence_max_tokens = 10000
        else:
            sentence_max_tokens = max_new_tokens
        
        # Create conversation for this sentence
        conversation = [
            {
                "role": str(speaker_id),
                "content": [
                    {"type": "text", "text": utterance_text},
                    {"type": "audio", "path": utterance}
                ]
            },
            {
                "role": str(speaker_id),
                "content": [{"type": "text", "text": sentence}]
            },
        ]
        
        # Prepare inputs
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Convert inputs to correct device and dtype
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # Convert float tensors to the right dtype
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor) and inputs[k].dtype in [torch.float32, torch.float64]:
                inputs[k] = inputs[k].to(dtype)
        
        # Generate audio for this sentence
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=sentence_max_tokens,
                depth_decoder_temperature=depth_decoder_temperature,
                depth_decoder_top_k=depth_decoder_top_k,
                depth_decoder_top_p=depth_decoder_top_p,
                output_audio=True
            )
        
        # Convert to numpy and trim silence
        audio = audio_values[0].to(torch.float32).cpu().numpy()
        audio = trim_silence(audio, sample_rate=24000)
        all_audio_segments.append(audio)
    
    # Concatenate all segments with small gaps
    if len(all_audio_segments) > 1:
        gap_samples = int(24000 * 0.3)  # 300ms gap between sentences
        gap = np.zeros(gap_samples)
        
        combined_audio = all_audio_segments[0]
        for audio_segment in all_audio_segments[1:]:
            combined_audio = np.concatenate([combined_audio, gap, audio_segment])
        
        audio = combined_audio
        print(f"Combined {len(all_audio_segments)} sentences with {len(gap)/24000:.2f}s gaps")
    else:
        audio = all_audio_segments[0]
    
    # Save
    audio_length = len(audio) / 24000
    sf.write(output_path, audio, 24000)
    print(f"✅ Audio saved to {output_path} ({audio_length:.2f}s)")
    
    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech using voice cloning with reference audio"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="lesterthomas/csm-lester-voice",
        help="Path to fine-tuned model (local or HuggingFace)"
    )
    parser.add_argument(
        "-t", "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "-r", "--reference-audio",
        type=str,
        default="reference-audio.wav",
        help="Path to reference audio file"
    )
    parser.add_argument(
        "-rt", "--reference-text",
        type=str,
        default="reference-utterance.txt",
        help="Path to reference text file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="cloned_output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max audio tokens to generate. If not specified, auto-calculated (~4 tokens/word)"
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID"
    )
    parser.add_argument(
        "--force-cuda",
        action="store_true",
        help="Force CUDA usage (error if not available)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use: cuda, cpu, or auto (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Show CUDA setup
    check_cuda_setup()
    
    # Determine CUDA usage
    force_cuda = args.force_cuda or args.device == "cuda"
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model, force_cuda=force_cuda)
    
    # Generate speech
    generate_speech_with_reference(
        model=model,
        processor=processor,
        text_to_speak=args.text,
        reference_audio_path=args.reference_audio,
        reference_text_path=args.reference_text,
        output_path=args.output,
        speaker_id=args.speaker_id,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
