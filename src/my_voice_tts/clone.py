"""
Voice cloning and speech synthesis using Chatterbox TTS

Supports multiple models:
- chatterbox: Full quality model (slower, better quality)
- turbo: Faster model (2-3x faster, slightly lower quality)
"""

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import torch
import soundfile as sf
from typing import Literal

console = Console()

# Supported models
ModelType = Literal["chatterbox", "turbo"]
AVAILABLE_MODELS = ["chatterbox", "turbo"]

# Global model cache - keyed by model type
_models: dict = {}


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
    Load a Chatterbox TTS model (cached).
    
    Args:
        model_type: "chatterbox" for full quality, "turbo" for faster inference
    
    Returns:
        The loaded model instance
    """
    global _models
    
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
        model_type: "chatterbox" or "turbo"
        exaggeration: Emotion exaggeration factor (0.0-1.0)
        cfg_weight: Classifier-free guidance weight (0.0-1.0)
    
    Returns:
        Path to the generated audio file
    """
    voice_path = Path(voice_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    
    model_name = "Turbo" if model_type == "turbo" else "Chatterbox"
    console.print(f"\n[bold cyan]🔊 Generating Speech ({model_name})[/bold cyan]")
    console.print(f"Voice: [yellow]{voice_path.name}[/yellow]")
    console.print(f"Text: [dim]{text[:100]}{'...' if len(text) > 100 else ''}[/dim]")
    
    model = load_model(model_type)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating speech...", total=None)
        
        # Generate audio using Chatterbox
        wav = model.generate(
            text=text,
            audio_prompt_path=str(voice_path),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        
        progress.update(task, description="Saving audio...")
        
        # Save the output
        # Chatterbox returns a tensor, convert to numpy and save
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        
        # Ensure 1D array for mono audio
        if wav.ndim > 1:
            wav = wav.flatten()
        
        # Ensure float32 and normalize if needed
        wav = wav.astype('float32')
        if wav.max() > 1.0 or wav.min() < -1.0:
            wav = wav / max(abs(wav.max()), abs(wav.min()))
        
        # Get sample rate from model
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
        model_type: "chatterbox" or "turbo"
        chunk_size: Maximum characters per chunk
        **kwargs: Additional arguments for synthesize_speech
    
    Returns:
        Path to the combined audio file
    """
    from pydub import AudioSegment
    import tempfile
    import os
    
    # Split text into sentences/chunks
    chunks = split_text_into_chunks(text, chunk_size)
    
    if len(chunks) == 1:
        return synthesize_speech(chunks[0], voice_path, output_path, model_type=model_type, **kwargs)
    
    model_name = "Turbo" if model_type == "turbo" else "Chatterbox"
    console.print(f"\n[bold cyan]📚 Processing Long Text ({model_name})[/bold cyan]")
    console.print(f"Total chunks: [yellow]{len(chunks)}[/yellow]")
    
    combined = None
    temp_files = []
    
    model = load_model(model_type)
    sample_rate = model.sr if hasattr(model, 'sr') else 24000
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Generating speech...", total=len(chunks))
        
        for i, chunk in enumerate(chunks):
            # Create temp file for this chunk
            temp_path = Path(tempfile.mktemp(suffix=".wav"))
            temp_files.append(temp_path)
            
            progress.update(task, description=f"Chunk {i+1}/{len(chunks)}...")
            
            # Generate this chunk
            wav = model.generate(
                text=chunk,
                audio_prompt_path=str(voice_path),
                exaggeration=kwargs.get('exaggeration', 0.5),
                cfg_weight=kwargs.get('cfg_weight', 0.5),
            )
            
            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().numpy()
            
            # Ensure 1D array for mono audio
            if wav.ndim > 1:
                wav = wav.flatten()
            
            wav = wav.astype('float32')
            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / max(abs(wav.max()), abs(wav.min()))
            
            sf.write(str(temp_path), wav, sample_rate)
            
            # Load and concatenate
            segment = AudioSegment.from_wav(str(temp_path))
            if combined is None:
                combined = segment
            else:
                # Add small pause between chunks
                combined = combined + AudioSegment.silent(duration=300) + segment
            
            progress.advance(task)
    
    # Export combined audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format="wav")
    
    # Cleanup temp files
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
    
    # Clean up text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # Handle very long sentences
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
