from dataclasses import dataclass
from pathlib import Path

import audiofile
import audresample
import librosa
import mlx.core as mx
import numpy as np


@dataclass
class PreprocessArgs:
    sample_rate: int
    normalize: str
    window_size: float
    window_stride: float
    window: str
    features: int
    n_fft: int
    dither: float
    pad_to: int
    pad_value: float

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)


def load_audio(filename: Path, sampling_rate: int) -> mx.array:
    signal, original_sampling_rate = audiofile.read(str(filename), always_2d=True)

    signal = audresample.resample(signal, original_sampling_rate, sampling_rate)

    signal = mx.array(signal)

    if signal.shape[0] >= 1:
        signal = signal.mean(axis=0)
    else:
        signal = signal.squeeze(0)

    return signal  # (audio_length, )


# will implement without librosa later
def get_logmel(x: mx.array, args: PreprocessArgs) -> mx.array:
    if args.dither > 0:
        x = x + mx.random.normal(x.shape) * args.dither

    if args.pad_to > 0:
        if x.shape[-1] < args.pad_to:
            pad_length = args.pad_to - x.shape[-1]
            x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    mel_spec = librosa.feature.melspectrogram(
        y=np.asarray(x),
        sr=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=args.window,
        n_mels=args.features,
        fmin=0,
        fmax=None,
        norm="slaney",
    )

    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    if args.normalize == "per_feature":
        mean = np.mean(mel_spec, axis=1, keepdims=True)
        std = np.std(mel_spec, axis=1, keepdims=True)
        normalized_mel = (mel_spec - mean) / (std + 1e-5)
    else:
        mean = np.mean(mel_spec)
        std = np.std(mel_spec)
        normalized_mel = (mel_spec - mean) / (std + 1e-5)

    normalized_mel = normalized_mel.T
    normalized_mel = np.expand_dims(normalized_mel, axis=0)

    return mx.array(normalized_mel)
