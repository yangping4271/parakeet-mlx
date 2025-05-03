# Parakeet MLX

An implementation of the Parakeet models - Nvidia's ASR(Automatic Speech Recognition) models - for Apple Silicon using MLX.

> Currently only for Parakeet TDT models. Support for additional Parakeet model variants is planned.

## Installation

Using [uv](https://docs.astral.sh/uv/) - recommended way:

```bash
uv add git+https://github.com/senstella/parakeet-mlx
```

Using pip:

```bash
pip install git+https://github.com/senstella/parakeet-mlx
```

## Quick Start

Transcribe a file:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("senstella/parakeet-tdt-0.6b-v2-mlx")

result = model.transcribe("audio_file.wav")

print(result.text)
```

Check timestamps:

```py
from parakeet_mlx import from_pretrained

model = from_pretrained("senstella/parakeet-tdt-0.6b-v2-mlx")

result = model.transcribe("audio_file.wav")

print(result.sentences)
# [AlignedSentence(text="Hello World.", start=1.01, end=2.04, duration=1.03, tokens=[...])]
```

## Timestamp Result

- `AlignedResult`: Top-level result containing the full text and sentences
  - `text`: Full transcribed text
  - `sentences`: List of `AlignedSentence`
- `AlignedSentence`: Sentence-level alignments with start/end times
  - `text`: Sentence text
  - `start`: Start time in seconds
  - `end`: End time in seconds
  - `duration`: Between `start` and `end`.
  - `tokens`: List of `AlignedToken`
- `AlignedToken`: Word/token-level alignments with precise timestamps
  - `text`: Token text
  - `start`: Start time in seconds
  - `end`: End time in seconds
  - `duration`: Between `start` and `end`.

### Low-Level API

To transcribe log-mel spectrum directly, you can do the following:

```python
import mlx.core as mx
from parakeet_mlx.audio import get_logmel, load_audio

# Load and preprocess audio manually
audio = load_audio("audio.wav", model.preprocessor_config.sample_rate)
mel = get_logmel(audio, model.preprocessor_config)

# Generate transcription with alignments
# Accepts both [batch, sequence, feat] and [sequence, feat]
# `alignments` is list of AlignedResult. (no matter you fed batch dimension or not!)
alignments = model.generate(mel)
```

## Todo

- [ ] Add CLI for better usability
- [ ] Streaming input (Although RTF is MUCH higher than 1 currently - it should be much sufficient to stream with current state)
- [ ] Compiling for RNNT decoder
- [ ] Add support for other Parakeet varients
- [ ] Remove librosa dependency


## Acknowledgments

- Thanks to [Nvidia](https://www.nvidia.com/) for training this awesome models and writing cool papers and providing nice implementation.
- Thanks to [MLX](https://github.com/ml-explore/mlx) project for providing the framework that made this implementation possible.
- Thanks to [audiofile](https://github.com/audeering/audiofile) and [audresample](https://github.com/audeering/audresample), [numpy](https://numpy.org), [librosa](https://librosa.org) for audio processing.
- Thanks to [dacite](https://github.com/konradhalas/dacite) for config management.

## License

Apache 2.0
