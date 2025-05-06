import functools
from dataclasses import dataclass
from pathlib import Path

import audiofile
import audresample
import librosa
import mlx.core as mx
import numpy as np
from parakeet_mlx.alignment import AlignedResult, AlignedToken, tokens_to_sentences, sentences_to_result


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
            prefix = x[1: padding + 1][::-1]
            suffix = x[-(padding + 1): -1][::-1]
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
    x = mx.square(mx.abs(x)).astype(mx.float16)
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


def get_logmel_fp16(
        x: mx.array,
        args: PreprocessArgs,
        *,
        power_dtype: mx.Dtype = mx.float16
) -> mx.array:
    """
    Identical signature to parakeet_mlx.audio.get_logmel but keeps the heavy spectrogram in float16, lets you override args.n_fft before the call
    """
    if args.pad_to and x.shape[-1] < args.pad_to:
        x = mx.pad(x, ((0, args.pad_to - x.shape[-1]),),
                   constant_values=args.pad_value)

    window = {
        "hanning": hanning,
        "hamming": hamming,
        "blackman": blackman,
        "bartlett": bartlett,
    }.get(args.window, hanning)(args.win_length).astype(power_dtype)

    X = stft(
        x,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=window
    )

    power = mx.square(mx.abs(X)).astype(power_dtype)

    mel = mx.matmul(args._filterbanks.astype(power_dtype), power.T)
    mel = mx.log(mel + 1e-5)

    if args.normalize == "per_feature":
        mel = (mel - mx.mean(mel, 1, keepdims=True)) / (mx.std(mel, 1, keepdims=True) + 1e-5)
    else:
        mel = (mel - mx.mean(mel)) / (mx.std(mel) + 1e-5)

    mel = mx.expand_dims(mel.T, 0)
    return mel.astype(power_dtype)


def transcribe_long_audio(
        model,
        path,
        *,
        segment_sec: float = 30.0,
        overlap_sec: float = 0.5,
        dedupe_gap: float = 0.30,
        dtype=mx.bfloat16
) -> AlignedResult:
    sr = model.preprocessor_config.sample_rate
    seg = int(segment_sec * sr)
    hop = int((segment_sec - overlap_sec) * sr)

    # shrink FFT and rebuild filterbank once
    if model.preprocessor_config.n_fft > 1024:
        model.preprocessor_config.n_fft = 1024
        import librosa
        model.preprocessor_config._filterbanks = mx.array(
            librosa.filters.mel(
                sr=sr, n_fft=1024,
                n_mels=model.preprocessor_config.features,
                fmin=0, fmax=None, norm="slaney"),
            dtype=mx.float32)

    audio = load_audio(Path(path), sr, dtype)
    n_samples = len(audio)

    global_tokens: list[AlignedToken] = []

    for start in range(0, n_samples, hop):
        end = min(start + seg, n_samples)
        snippet = audio[start:end]
        if not len(snippet):
            break

        mel = get_logmel_fp16(snippet, model.preprocessor_config)
        seg_res = model.generate(mel)[0]

        offset = start / sr
        for tok in _tokens_from_result(seg_res):
            tok.start += offset
            global_tokens.append(tok)

        _clear_mlx_cache()

    global_tokens.sort(key=lambda t: t.start)
    deduped = []
    for tok in global_tokens:
        if (not deduped or
                tok.text != deduped[-1].text or
                tok.start - deduped[-1].start >= dedupe_gap):
            deduped.append(tok)

    sentences = tokens_to_sentences(deduped)
    return sentences_to_result(sentences)


def _tokens_from_result(res) -> list[AlignedToken]:
    if hasattr(res, "tokens"):
        return res.tokens
    toks = []
    for s in getattr(res, "sentences", []):
        toks.extend(getattr(s, "tokens", []))
    return toks


def _clear_mlx_cache():
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    else:
        try:
            mx.metal.clear_cache()
        except AttributeError:
            pass
