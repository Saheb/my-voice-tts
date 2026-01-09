"""
Voice recorder module - Record voice samples for cloning
"""

import sounddevice as sd
import soundfile as sf
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()

# Sample text to read for voice cloning (varied speech patterns)
SAMPLE_TEXT = """
Hello, this is my voice sample for text-to-speech cloning.
I'm reading this passage to help the AI learn how I speak.
My voice has certain unique qualities - the pitch, the rhythm, the way I pause.
Sometimes I speak quickly, and other times I slow down for emphasis.
This should be enough for the model to capture my voice characteristics.
"""

DEFAULT_SAMPLE_RATE = 22050  # Standard for TTS models
DEFAULT_CHANNELS = 1  # Mono audio


def get_audio_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate']
            })
    return input_devices


def record_voice(
    output_path: Path,
    duration: float = 15.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: int | None = None
) -> Path:
    """
    Record voice from microphone.
    
    Args:
        output_path: Where to save the WAV file
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        device: Audio device ID (None for default)
    
    Returns:
        Path to the saved audio file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]🎤 Voice Recording[/bold cyan]")
    console.print(f"Duration: [yellow]{duration}[/yellow] seconds")
    console.print(f"Sample rate: [yellow]{sample_rate}[/yellow] Hz")
    console.print()
    
    # Show the text to read
    console.print("[bold]Please read the following text aloud:[/bold]")
    console.print()
    console.print(f"[italic green]{SAMPLE_TEXT.strip()}[/italic green]")
    console.print()
    
    # Countdown
    console.print("[bold red]Recording starts in...[/bold red]")
    for i in range(3, 0, -1):
        console.print(f"[bold yellow]{i}...[/bold yellow]")
        time.sleep(1)
    
    console.print("[bold green]🔴 RECORDING NOW - Speak clearly![/bold green]")
    
    # Record audio
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Recording...", total=None)
        
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=DEFAULT_CHANNELS,
            dtype='float32',
            device=device
        )
        sd.wait()
        
        progress.update(task, description="Recording complete!")
    
    # Save to file
    sf.write(str(output_path), audio_data, sample_rate)
    
    # Save the reference text for F5-TTS (needs transcription of what was spoken)
    txt_path = output_path.with_suffix('.txt')
    txt_path.write_text(SAMPLE_TEXT.strip())
    
    console.print(f"\n[bold green]✅ Saved to:[/bold green] {output_path}")
    console.print(f"[dim]Reference text saved to: {txt_path.name}[/dim]")
    console.print(f"[dim]File size: {output_path.stat().st_size / 1024:.1f} KB[/dim]")
    
    return output_path


def list_recordings(voices_dir: Path) -> list[Path]:
    """List all voice recordings in the voices directory."""
    voices_dir = Path(voices_dir)
    if not voices_dir.exists():
        return []
    
    recordings = list(voices_dir.glob("*.wav"))
    return sorted(recordings, key=lambda p: p.stat().st_mtime, reverse=True)
