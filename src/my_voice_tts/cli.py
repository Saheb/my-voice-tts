"""
CLI interface for my-voice-tts
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VOICES_DIR = PROJECT_ROOT / "voices"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="my-voice-tts",
        description="🎤 Local voice cloning TTS - Read blog posts in your voice"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Record your voice sample")
    record_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename (default: voice_TIMESTAMP.wav)"
    )
    record_parser.add_argument(
        "-d", "--duration",
        type=float,
        default=15.0,
        help="Recording duration in seconds (default: 15)"
    )
    record_parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices"
    )
    record_parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device ID to use"
    )
    
    # Speak command
    speak_parser = subparsers.add_parser("speak", help="Generate speech from text")
    speak_parser.add_argument(
        "-t", "--text",
        type=str,
        required=True,
        help="Text to speak"
    )
    speak_parser.add_argument(
        "-v", "--voice",
        type=str,
        required=True,
        help="Path to voice reference audio file"
    )
    speak_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename (default: speech_TIMESTAMP.wav)"
    )
    speak_parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion exaggeration (0.0-1.0, default: 0.5)"
    )
    speak_parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["chatterbox", "turbo"],
        default="chatterbox",
        help="TTS model: 'chatterbox' (better quality) or 'turbo' (2-3x faster)"
    )
    
    # Read-blog command
    blog_parser = subparsers.add_parser("read-blog", help="Read a blog post in your voice")
    blog_parser.add_argument(
        "-u", "--url",
        type=str,
        required=True,
        help="Blog post URL"
    )
    blog_parser.add_argument(
        "-v", "--voice",
        type=str,
        required=True,
        help="Path to voice reference audio file"
    )
    blog_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename (default: blog_TIMESTAMP.wav)"
    )
    blog_parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview extracted content without generating audio"
    )
    blog_parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["chatterbox", "turbo"],
        default="chatterbox",
        help="TTS model: 'chatterbox' (better quality) or 'turbo' (2-3x faster)"
    )
    
    # Voices command - list available voice samples
    voices_parser = subparsers.add_parser("voices", help="List available voice samples")
    
    args = parser.parse_args()
    
    if args.command is None:
        show_welcome()
        parser.print_help()
        return 0
    
    try:
        if args.command == "record":
            return cmd_record(args)
        elif args.command == "speak":
            return cmd_speak(args)
        elif args.command == "read-blog":
            return cmd_read_blog(args)
        elif args.command == "voices":
            return cmd_voices(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        return 1
    
    return 0


def show_welcome():
    """Show welcome message."""
    console.print(Panel.fit(
        "[bold cyan]🎤 My Voice TTS[/bold cyan]\n"
        "[dim]Local voice cloning for reading blog posts[/dim]",
        border_style="cyan"
    ))
    console.print()


def cmd_record(args):
    """Handle record command."""
    from .recorder import record_voice, get_audio_devices, list_recordings
    
    if args.list_devices:
        devices = get_audio_devices()
        table = Table(title="Audio Input Devices")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Channels")
        
        for device in devices:
            table.add_row(
                str(device['id']),
                device['name'],
                str(device['channels'])
            )
        
        console.print(table)
        return 0
    
    # Determine output path
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = VOICES_DIR / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = VOICES_DIR / f"voice_{timestamp}.wav"
    
    record_voice(output_path, duration=args.duration, device=args.device)
    
    return 0


def cmd_speak(args):
    """Handle speak command."""
    from .clone import synthesize_speech
    
    voice_path = Path(args.voice)
    if not voice_path.is_absolute():
        # Check if it's in voices dir
        if (VOICES_DIR / args.voice).exists():
            voice_path = VOICES_DIR / args.voice
    
    if not voice_path.exists():
        console.print(f"[red]Voice file not found:[/red] {args.voice}")
        console.print(f"[dim]Tip: Run 'my-voice-tts voices' to see available voice samples[/dim]")
        return 1
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = OUTPUT_DIR / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"speech_{timestamp}.wav"
    
    synthesize_speech(
        text=args.text,
        voice_path=voice_path,
        output_path=output_path,
        model_type=args.model,
        exaggeration=args.exaggeration
    )
    
    return 0


def cmd_read_blog(args):
    """Handle read-blog command."""
    from .blog_reader import fetch_blog_content
    from .clone import synthesize_long_text
    
    # Fetch blog content
    content = fetch_blog_content(args.url)
    
    if args.preview:
        console.print(Panel(
            f"[bold]{content['title']}[/bold]\n\n"
            f"[dim]Source: {content['source']}[/dim]\n\n"
            f"{content['content'][:1000]}..."
            if len(content['content']) > 1000 else content['content'],
            title="Blog Preview"
        ))
        console.print(f"\n[cyan]Total length:[/cyan] {len(content['content'])} characters")
        return 0
    
    # Get voice path
    voice_path = Path(args.voice)
    if not voice_path.is_absolute():
        if (VOICES_DIR / args.voice).exists():
            voice_path = VOICES_DIR / args.voice
    
    if not voice_path.exists():
        console.print(f"[red]Voice file not found:[/red] {args.voice}")
        return 1
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = OUTPUT_DIR / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"blog_{timestamp}.wav"
    
    # Prepare TTS text - include title
    tts_text = f"{content['title']}. {content['content']}"
    
    console.print(f"\n[cyan]Generating audio for {len(tts_text)} characters...[/cyan]")
    
    synthesize_long_text(
        text=tts_text,
        voice_path=voice_path,
        output_path=output_path,
        model_type=args.model
    )
    
    return 0


def cmd_voices(args):
    """Handle voices command."""
    from .recorder import list_recordings
    
    recordings = list_recordings(VOICES_DIR)
    
    if not recordings:
        console.print("[yellow]No voice samples found.[/yellow]")
        console.print("[dim]Run 'my-voice-tts record' to create one.[/dim]")
        return 0
    
    table = Table(title="Available Voice Samples")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Modified")
    
    for recording in recordings:
        stat = recording.stat()
        size_kb = stat.st_size / 1024
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            recording.name,
            f"{size_kb:.1f} KB",
            modified
        )
    
    console.print(table)
    console.print(f"\n[dim]Voice samples directory: {VOICES_DIR}[/dim]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
