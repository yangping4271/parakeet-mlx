# Parakeet MLX

An implementation of the Parakeet models - Nvidia's ASR(Automatic Speech Recognition) models - for Apple Silicon using MLX.

## Installation

> [!NOTE]
> Make sure you have `ffmpeg` installed on your system first, otherwise CLI won't work properly.

Using [uv](https://docs.astral.sh/uv/) - recommended way:

```bash
git clone https://github.com/your-repo/parakeet-mlx.git
cd parakeet-mlx
uv install
```

For the CLI, after local installation:

```bash
uv tool install .
```

## CLI Quick Start

```bash
parakeet-mlx <audio_files> [OPTIONS]
```

## Arguments

- `audio_files`: One or more audio files to transcribe (WAV, MP3, etc.)

## Options

- `--model` (default: `mlx-community/parakeet-tdt-0.6b-v2`)
  - Hugging Face repository of the model to use

- `--output-dir` (default: current directory)
  - Directory to save transcription outputs

- `--output-format` (default: srt)
  - Output format (txt/srt/vtt/json/all)

- `--output-template` (default: `{filename}`)
  - Template for output filenames, `{filename}`, `{index}`, `{date}` is supported.

- `--timestamps` (default: False)
  - Enable word-level timestamps in SRT/VTT outputs

- `--verbose` / `-v` (default: False)
  - Print detailed progress information

- `--chunk-duration` (default: 120 seconds)
  - Chunking duration in seconds for long audio, `0` to disable chunking

- `--overlap-duration` (default: 15 seconds)
  - Overlap duration in seconds if using chunking

- `--fp32` / `--bf16` (default: `bf16`)
  - Determine the precision to use

## Examples

```bash
# Basic transcription
parakeet-mlx audio.mp3

# Multiple files with word-level timestamps of VTT subtitle
parakeet-mlx *.mp3 --output-format vtt --timestamps

# Generate all output formats
parakeet-mlx audio.mp3 --output-format all
```

## Acknowledgments

- Thanks to [Nvidia](https://www.nvidia.com/) for training these awesome models and writing cool papers and providing nice implementation.
- Thanks to [MLX](https://github.com/ml-explore/mlx) project for providing the framework that made this implementation possible.
- Thanks to [audiofile](https://github.com/audeering/audiofile) and [audresample](https://github.com/audeering/audresample), [numpy](https://numpy.org), [librosa](https://librosa.org) for audio processing.
- Thanks to [dacite](https://github.com/konradhalas/dacite) for config management.

## License

Apache 2.0
