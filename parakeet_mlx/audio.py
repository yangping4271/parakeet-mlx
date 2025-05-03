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
    window_size: float
    window_stride: float
    window: str
    features: int
    n_fft: int


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
    mel_spec = librosa.feature.melspectrogram(
        y=np.asarray(x),
        sr=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=160,
        n_mels=args.features,
        fmin=0,
        fmax=8000,
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    mean = np.mean(log_mel_spec)
    std = np.std(log_mel_spec)
    normalized_mel = (log_mel_spec - mean) / (std + 1e-5)

    normalized_mel = normalized_mel.T
    normalized_mel = np.expand_dims(normalized_mel, axis=0)

    return mx.array(normalized_mel)
