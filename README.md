# My Voice TTS

🎤 **Local voice cloning TTS** - Read blog posts in your cloned voice using Chatterbox TTS.

## Features

- **Voice Recording** - Record your voice samples for cloning
- **Voice Cloning** - Clone your voice from ~5-15 seconds of audio
- **Multiple Models** - Choose between high-quality (Chatterbox) or fast (Turbo) modes
- **Blog Reading** - Fetch and read blog posts in your voice
- **100% Local** - No API calls, all processing on your Mac

## Requirements

- macOS 12.0+ (Apple Silicon M1/M2/M3/M4)
- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- Hugging Face account (for Turbo model download)

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Login to Hugging Face (Optional)
Required only if you want to use the **Turbo** model (faster).

```bash
uv run hf auth login
```
Paste your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) containing `read` permission.

### 3. Record Your Voice

```bash
uv run my-voice-tts record
```

This will guide you to read a sample text and save your voice recording.

### 4. Generate Speech

**Default (High Quality):**
```bash
uv run my-voice-tts speak -v voices/your_voice.wav -t "Hello there!"
```

**Turbo Mode (2-3x Faster):**
```bash
uv run my-voice-tts speak -v voices/your_voice.wav -t "Hello there!" --model turbo
```

### 5. Read a Blog Post

```bash
uv run my-voice-tts read-blog -u "https://example.com/blog-post" -v voices/your_voice.wav --model turbo
```

## Commands

| Command | Option | Description |
|---------|--------|-------------|
| `record` | | Record your voice sample |
| `speak` | `--model [chatterbox\|turbo]` | Generate speech (default: chatterbox) |
| `read-blog` | `--model [chatterbox\|turbo]` | Read a blog post |
| `voices` | | List available voice samples |

## Voice Recording Tips

For best results:
- Record in a **quiet environment**
- Speak **clearly and naturally**
- Record at least **10-15 seconds**
- Include **varied speech patterns** (questions, statements)

## Output

Generated audio files are saved to the `output/` directory.

## License

MIT
