import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import typer
from mlx.core import bfloat16, float32
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from typing_extensions import Annotated

from parakeet_mlx import AlignedResult, AlignedSentence, AlignedToken, from_pretrained

app = typer.Typer(no_args_is_help=True)


# helpers
def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds >= 0
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def to_txt(result: AlignedResult) -> str:
    """Format transcription result as plain text."""
    return result.text.strip()


def to_srt(result: AlignedResult, timestamps: bool = False) -> str:
    """
    Format transcription result as an SRT file.
    """
    srt_content = []
    entry_index = 1
    if timestamps:
        words = result.words
        for word in words:
            if not word.text.strip():
                continue
            start_time = format_timestamp(word.start, decimal_marker=",")
            end_time = format_timestamp(word.end, decimal_marker=",")
            text = word.text.strip()

            srt_content.append(f"{entry_index}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
            entry_index += 1
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=",")
            end_time = format_timestamp(sentence.end, decimal_marker=",")
            text = sentence.text.strip()

            srt_content.append(f"{entry_index}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
            entry_index += 1

    return "\n".join(srt_content)


def to_vtt(result: AlignedResult, timestamps: bool = False) -> str:
    """
    Format transcription result as a VTT file.
    """
    vtt_content = ["WEBVTT", ""]
    if timestamps:
        words = result.words
        for word in words:
            if not word.text.strip():
                continue
            start_time = format_timestamp(word.start, decimal_marker=".")
            end_time = format_timestamp(word.end, decimal_marker=".")
            text = word.text.strip()

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
    else:
        for sentence in result.sentences:
            start_time = format_timestamp(sentence.start, decimal_marker=".")
            end_time = format_timestamp(sentence.end, decimal_marker=".")
            text_line = sentence.text.strip()

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text_line)
            vtt_content.append("")

    return "\n".join(vtt_content)


def _aligned_token_to_dict(token: AlignedToken) -> Dict[str, Any]:
    return {
        "text": token.text,
        "start": round(token.start, 3),
        "end": round(token.end, 3),
        "duration": round(token.duration, 3),
    }


def _aligned_sentence_to_dict(sentence: AlignedSentence) -> Dict[str, Any]:
    return {
        "text": sentence.text,
        "start": round(sentence.start, 3),
        "end": round(sentence.end, 3),
        "duration": round(sentence.duration, 3),
        "tokens": [_aligned_token_to_dict(token) for token in sentence.tokens],
    }


def to_json(result: AlignedResult) -> str:
    output_dict = {
        "text": result.text,
        "sentences": [
            _aligned_sentence_to_dict(sentence) for sentence in result.sentences
        ],
    }
    return json.dumps(output_dict, indent=2, ensure_ascii=False)


@app.command("transcribe")
def transcribe(
    audios: Annotated[
        List[Path],
        typer.Argument(
            help="Files to transcribe",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        str, typer.Option(help="Hugging Face repository of model to use")
    ] = "mlx-community/parakeet-tdt-0.6b-v2",
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save transcriptions")
    ] = Path("."),
    output_format: Annotated[
        str, typer.Option(help="Format for output files (txt, srt, vtt, json, all)")
    ] = "srt",
    output_template: Annotated[
        str,
        typer.Option(
            help="Template for output filenames, e.g. '{filename}_{date}_{index}'"
        ),
    ] = "{filename}",
    timestamps: Annotated[
        bool,
        typer.Option(help="Output word-level timestamps in srt/vtt formats"),
    ] = False,
    chunk_duration: Annotated[
        float,
        typer.Option(
            help="Chunking duration in seconds for long audio, 0 to disable chunking."
        ),
    ] = 60 * 2,
    overlap_duration: Annotated[
        float, typer.Option(help="Overlap duration in seconds if using chunking")
    ] = 15,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print out process and debug messages"),
    ] = False,
    fp32: Annotated[
        bool, typer.Option("--fp32/--bf16", help="Use FP32 precision")
    ] = False,
):
    """
    Transcribe audio files using Parakeet MLX models.
    """
    if verbose:
        print(f"Loading model: [bold cyan]{model}[/bold cyan]...")

    try:
        loaded_model = from_pretrained(model, dtype=bfloat16 if not fp32 else float32)
        if verbose:
            print("[green]Model loaded successfully.[/green]")
    except Exception as e:
        print(f"[bold red]Error loading model {model}:[/bold red] {e}")
        raise typer.Exit(code=1)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[bold red]Error creating output directory {output_dir}:[/bold red] {e}")
        raise typer.Exit(code=1)

    if verbose:
        print(f"Output directory: [bold cyan]{output_dir.resolve()}[/bold cyan]")
        print(f"Output format(s): [bold cyan]{output_format}[/bold cyan]")
        if output_format in ["srt", "vtt", "all"] and timestamps:
            print("Timestamps: [bold cyan]Enabled[/bold cyan]")

    formatters = {
        "txt": to_txt,
        "srt": lambda r: to_srt(r, timestamps=timestamps),
        "vtt": lambda r: to_vtt(r, timestamps=timestamps),
        "json": to_json,
    }

    formats_to_generate = []
    if output_format == "all":
        formats_to_generate = list(formatters.keys())
    elif output_format in formatters:
        formats_to_generate = [output_format]
    else:
        print(
            f"[bold red]Error: Invalid output format '{output_format}'. Choose from {list(formatters.keys()) + ['all']}.[/bold red]"
        )
        raise typer.Exit(code=1)

    total_files = len(audios)
    if verbose:
        print(f"Transcribing {total_files} file(s)...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Transcribing...", total=total_files)

        for i, audio_path in enumerate(audios):
            if verbose:
                print(
                    f"\nProcessing file {i + 1}/{total_files}: [bold cyan]{audio_path.name}[/bold cyan]"
                )
            else:
                progress.update(
                    task, description=f"Processing [cyan]{audio_path.name}[/cyan]..."
                )

            try:
                result: AlignedResult = loaded_model.transcribe(
                    audio_path,
                    dtype=bfloat16 if not fp32 else float32,
                    chunk_duration=chunk_duration if chunk_duration != 0 else None,
                    overlap_duration=overlap_duration,
                    chunk_callback=lambda current, full: progress.update(
                        task, total=total_files * full, completed=full * i + current
                    ),
                )

                if verbose:
                    for sentence in result.sentences:
                        start, end, text = sentence.start, sentence.end, sentence.text
                        line = f"[blue][{format_timestamp(start)} --> {format_timestamp(end)}][/blue] {text.strip()}"
                        print(line)

                base_filename = audio_path.stem
                template_vars = {
                    "filename": base_filename,
                    "date": datetime.datetime.now().strftime("%Y%m%d"),
                    "index": str(i + 1),
                }

                output_basename = output_template.format(**template_vars)

                for fmt in formats_to_generate:
                    formatter = formatters[fmt]
                    output_content = formatter(result)
                    output_filename = f"{output_basename}.{fmt}"
                    output_filepath = output_dir / output_filename

                    try:
                        with open(output_filepath, "w", encoding="utf-8") as f:
                            f.write(output_content)
                        if verbose:
                            print(
                                f"[green]Saved {fmt.upper()}:[/green] {output_filepath.absolute()}"
                            )
                    except Exception as e:
                        print(
                            f"[bold red]Error writing output file {output_filepath}:[/bold red] {e}"
                        )

            except Exception as e:
                print(f"[bold red]Error transcribing file {audio_path}:[/bold red] {e}")

            progress.update(task, total=total_files, completed=i + 1)

    print(
        f"\n[bold green]Transcription complete.[/bold green] Outputs saved in '{output_dir.resolve()}'"
    )


if __name__ == "__main__":
    app()