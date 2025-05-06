import functools
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
    pad_to: int = 0
    pad_value: float = 0

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)

    def __post_init__(self):
        # slow slow slow. will remove librosa depedency later!
        self._filterbanks = mx.array(
            librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.features,
                fmin=0,
                fmax=None,
                norm="slaney",
            ),
            dtype=mx.float32,
        )


def load_audio(
    filename: Path, sampling_rate: int, dtype: mx.Dtype = mx.bfloat16
) -> mx.array:
    signal, original_sampling_rate = audiofile.read(str(filename), always_2d=True)

    signal = audresample.resample(signal, original_sampling_rate, sampling_rate)

    signal = mx.array(signal)

    if signal.shape[0] >= 1:
        signal = signal.mean(axis=0)
    else:
        signal = signal.squeeze(0)

    return signal.astype(dtype)  # (audio_length, )


# thanks to https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/audio.py
@functools.lru_cache(None)
def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])


@functools.lru_cache(None)
def hamming(size):
    return mx.array(np.hamming(size + 1)[:-1])


@functools.lru_cache(None)
def blackman(size):
    return mx.array(np.blackman(size + 1)[:-1])


@functools.lru_cache(None)
def bartlett(size):
    return mx.array(np.bartlett(size + 1)[:-1])


def stft(
    x, n_fft, hop_length=None, win_length=None, window=None, axis=-1, pad_mode="reflect"
):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = n_fft // 4
    if window is None:
        window = mx.ones(win_length)

    if win_length != n_fft:
        if win_length > n_fft:
            window = window[:n_fft]
        else:
            padding = [(0, n_fft - win_length)]
            window = mx.pad(window, padding)

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = win_length // 2
    x = _pad(x, padding, pad_mode)

    strides = [hop_length, 1]
    t = (x.size - win_length + hop_length) // hop_length
    shape = [t, n_fft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def get_logmel(x: mx.array, args: PreprocessArgs) -> mx.array:
    original_dtype = x.dtype

    if args.pad_to > 0:
        if x.shape[-1] < args.pad_to:
            pad_length = args.pad_to - x.shape[-1]
            x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    window = (
        hanning(args.win_length).astype(x.dtype)
        if args.window == "hanning"
        else hamming(args.win_length).astype(x.dtype)
        if args.window == "hamming"
        else blackman(args.win_length).astype(x.dtype)
        if args.window == "blackman"
        else bartlett(args.win_length).astype(x.dtype)
        if args.window == "bartlett"
        else None
    )

    x = stft(x, args.n_fft, args.hop_length, args.win_length, window)
    x = mx.square(mx.abs(x)).astype(original_dtype)
    x = mx.matmul(args._filterbanks.astype(x.dtype), x.T)
    x = mx.log(x + 1e-5)

    if args.normalize == "per_feature":
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.std(x, axis=1, keepdims=True)
        normalized_mel = (x - mean) / (std + 1e-5)
    else:
        mean = mx.mean(x)
        std = mx.std(x)
        normalized_mel = (x - mean) / (std + 1e-5)

    normalized_mel = normalized_mel.T
    normalized_mel = mx.expand_dims(normalized_mel, axis=0)

    return normalized_mel.astype(original_dtype)
