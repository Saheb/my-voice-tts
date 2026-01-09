"""
Voice cloning and speech synthesis

Supports multiple models:
- chatterbox: Full quality model (slower, better quality)
- turbo: Faster Chatterbox model (2-3x faster, slightly lower quality)
- f5: F5-TTS-MLX (fastest, native Apple Silicon, good quality)
"""

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import torch
import soundfile as sf
from typing import Literal
import tempfile
import os

console = Console()

# Supported models
ModelType = Literal["chatterbox", "turbo", "f5"]
AVAILABLE_MODELS = ["chatterbox", "turbo", "f5"]

# Global model cache - keyed by model type
_models: dict = {}

# Default reference text for F5-TTS (must match what was read during recording)
DEFAULT_REF_TEXT = """Hello, this is my voice sample for text-to-speech cloning.
I'm reading this passage to help the AI learn how I speak.
My voice has certain unique qualities - the pitch, the rhythm, the way I pause.
Sometimes I speak quickly, and other times I slow down for emphasis.
This should be enough for the model to capture my voice characteristics."""


def get_device() -> str:
    """Get the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _patch_perth_for_apple_silicon():
    """
    Patch the perth module to use DummyWatermarker on Apple Silicon.
    The PerthImplicitWatermarker native extension doesn't build on macOS ARM64.
    """
    import perth
    if perth.PerthImplicitWatermarker is None:
        console.print("[dim]Using DummyWatermarker (native watermarker unavailable on Apple Silicon)[/dim]")
        perth.PerthImplicitWatermarker = perth.DummyWatermarker


def load_model(model_type: ModelType = "chatterbox"):
    """
    Load a TTS model (cached).
    
    Args:
        model_type: "chatterbox", "turbo", or "f5"
    
    Returns:
        The loaded model instance (or None for f5 which uses a function API)
    """
    global _models
    
    # F5 doesn't need preloading - it loads on demand
    if model_type == "f5":
        return None
    
    if model_type in _models:
        return _models[model_type]
    
    model_name = "Chatterbox" if model_type == "chatterbox" else "Chatterbox Turbo"
    console.print(f"[cyan]Loading {model_name} model...[/cyan]")
    
    # Patch perth for Apple Silicon before importing
    _patch_perth_for_apple_silicon()
    
    device = get_device()
    console.print(f"[dim]Using device: {device}[/dim]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        
        if model_type == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
        
        progress.update(task, description="Model loaded!")
    
    _models[model_type] = model
    return model


def _get_voice_ref_text(voice_path: Path) -> str:
    """Get reference text for a voice file (for F5-TTS)."""
    # Look for a .txt file with the same name as the voice
    txt_path = voice_path.with_suffix('.txt')
    if txt_path.exists():
        return txt_path.read_text().strip()
    return DEFAULT_REF_TEXT


def synthesize_speech(
    text: str,
    voice_path: Path,
    output_path: Path,
    model_type: ModelType = "chatterbox",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> Path:
    """
    Generate speech from text using a cloned voice.
    
    Args:
        text: The text to speak
        voice_path: Path to the reference voice audio file
        output_path: Where to save the output audio
        model_type: "chatterbox", "turbo", or "f5"
        exaggeration: Emotion exaggeration factor (0.0-1.0) - Chatterbox only
        cfg_weight: Classifier-free guidance weight (0.0-1.0) - Chatterbox only
    
    Returns:
        Path to the generated audio file
    """
    voice_path = Path(voice_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    
    model_names = {"chatterbox": "Chatterbox", "turbo": "Turbo", "f5": "F5-TTS"}
    model_name = model_names.get(model_type, model_type)
    
    console.print(f"\n[bold cyan]🔊 Generating Speech ({model_name})[/bold cyan]")
    console.print(f"Voice: [yellow]{voice_path.name}[/yellow]")
    console.print(f"Text: [dim]{text[:100]}{'...' if len(text) > 100 else ''}[/dim]")
    
    if model_type == "f5":
        return _synthesize_f5(text, voice_path, output_path)
    else:
        return _synthesize_chatterbox(text, voice_path, output_path, model_type, exaggeration, cfg_weight)


def _synthesize_f5(text: str, voice_path: Path, output_path: Path) -> Path:
    """Generate speech using F5-TTS-MLX."""
    from f5_tts_mlx.generate import generate
    import librosa
    import numpy as np
    
    ref_text = _get_voice_ref_text(voice_path)
    
    # F5-TTS requires 24kHz audio - resample if needed
    data, sr = sf.read(voice_path)
    temp_voice_path = None
    
    if sr != 24000:
        console.print(f"[dim]Resampling reference audio from {sr}Hz to 24000Hz...[/dim]")
        # Resample to 24kHz
        data_resampled = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=24000)
        temp_voice_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(temp_voice_path), data_resampled, 24000)
        voice_path_to_use = temp_voice_path
    else:
        voice_path_to_use = voice_path
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating with F5-TTS...", total=None)
            
            # F5-TTS generate function - returns None when output_path is provided
            result = generate(
                generation_text=text,
                ref_audio_path=str(voice_path_to_use),
                ref_audio_text=ref_text,
                output_path=str(output_path),
                steps=8,  # Default fast steps
            )
            
            progress.update(task, description="Complete!")
    finally:
        # Cleanup temp file
        if temp_voice_path and temp_voice_path.exists():
            try:
                os.unlink(temp_voice_path)
            except:
                pass
    
    console.print(f"\n[bold green]✅ Audio saved to:[/bold green] {output_path}")
    
    # Get duration from the output file
    if output_path.exists():
        data, sr = sf.read(output_path)
        duration = len(data) / sr
        console.print(f"[dim]Duration: {duration:.1f}s[/dim]")
    
    return output_path


def _synthesize_chatterbox(
    text: str,
    voice_path: Path,
    output_path: Path,
    model_type: str,
    exaggeration: float,
    cfg_weight: float
) -> Path:
    """Generate speech using Chatterbox."""
    model = load_model(model_type)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating speech...", total=None)
        
        wav = model.generate(
            text=text,
            audio_prompt_path=str(voice_path),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        
        progress.update(task, description="Saving audio...")
        
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        
        if wav.ndim > 1:
            wav = wav.flatten()
        
        wav = wav.astype('float32')
        if wav.max() > 1.0 or wav.min() < -1.0:
            wav = wav / max(abs(wav.max()), abs(wav.min()))
        
        sample_rate = model.sr if hasattr(model, 'sr') else 24000
        sf.write(str(output_path), wav, sample_rate)
        
        progress.update(task, description="Complete!")
    
    console.print(f"\n[bold green]✅ Audio saved to:[/bold green] {output_path}")
    console.print(f"[dim]Duration: {len(wav) / sample_rate:.1f}s[/dim]")
    
    return output_path


def synthesize_long_text(
    text: str,
    voice_path: Path,
    output_path: Path,
    model_type: ModelType = "chatterbox",
    chunk_size: int = 200,
    **kwargs
) -> Path:
    """
    Synthesize long text by splitting into chunks and concatenating.
    
    Args:
        text: The full text to speak
        voice_path: Path to the reference voice
        output_path: Where to save the output
        model_type: "chatterbox", "turbo", or "f5"
        chunk_size: Maximum characters per chunk
        **kwargs: Additional arguments for synthesis
    
    Returns:
        Path to the combined audio file
    """
    from pydub import AudioSegment
    
    chunks = split_text_into_chunks(text, chunk_size)
    
    if len(chunks) == 1:
        return synthesize_speech(chunks[0], voice_path, output_path, model_type=model_type, **kwargs)
    
    model_names = {"chatterbox": "Chatterbox", "turbo": "Turbo", "f5": "F5-TTS"}
    model_name = model_names.get(model_type, model_type)
    
    console.print(f"\n[bold cyan]📚 Processing Long Text ({model_name})[/bold cyan]")
    console.print(f"Total chunks: [yellow]{len(chunks)}[/yellow]")
    
    combined = None
    temp_files = []
    
    # For F5, we need ref text
    ref_text = _get_voice_ref_text(voice_path) if model_type == "f5" else None
    
    # Preload model for Chatterbox variants
    if model_type != "f5":
        model = load_model(model_type)
        sample_rate = model.sr if hasattr(model, 'sr') else 24000
    else:
        sample_rate = 24000  # F5 default
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Generating speech...", total=len(chunks))
        
        for i, chunk in enumerate(chunks):
            temp_path = Path(tempfile.mktemp(suffix=".wav"))
            temp_files.append(temp_path)
            
            progress.update(task, description=f"Chunk {i+1}/{len(chunks)}...")
            
            if model_type == "f5":
                from f5_tts_mlx.generate import generate
                generate(
                    generation_text=chunk,
                    ref_audio_path=str(voice_path),
                    ref_audio_text=ref_text,
                    output_path=str(temp_path),
                    steps=8,
                )
            else:
                wav = model.generate(
                    text=chunk,
                    audio_prompt_path=str(voice_path),
                    exaggeration=kwargs.get('exaggeration', 0.5),
                    cfg_weight=kwargs.get('cfg_weight', 0.5),
                )
                
                if isinstance(wav, torch.Tensor):
                    wav = wav.squeeze().cpu().numpy()
                
                if wav.ndim > 1:
                    wav = wav.flatten()
                
                wav = wav.astype('float32')
                if wav.max() > 1.0 or wav.min() < -1.0:
                    wav = wav / max(abs(wav.max()), abs(wav.min()))
                
                sf.write(str(temp_path), wav, sample_rate)
            
            segment = AudioSegment.from_wav(str(temp_path))
            if combined is None:
                combined = segment
            else:
                combined = combined + AudioSegment.silent(duration=300) + segment
            
            progress.advance(task)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format="wav")
    
    for temp_path in temp_files:
        try:
            os.unlink(temp_path)
        except:
            pass
    
    console.print(f"\n[bold green]✅ Audio saved to:[/bold green] {output_path}")
    console.print(f"[dim]Total duration: {len(combined) / 1000:.1f}s[/dim]")
    
    return output_path


def split_text_into_chunks(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks, preferring sentence boundaries."""
    import re
    
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    if len(text) <= max_chars:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            if len(sentence) > max_chars:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk = f"{current_chunk} {word}".strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
