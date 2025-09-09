#!/usr/bin/env python
"""Merge example audio files and run the VANPY pipeline.

This script is intended for GitHub Codespaces. It concatenates all audio files
placed in ``src/speech_examples`` into a single ``merged.wav`` file and then
executes the default pipeline defined in ``src/run.py``. The HuggingFace token is
read from the environment variable ``hf`` (Codespaces secret) and written to a
temporary ``.env`` file so that pyannote models can authenticate.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydub import AudioSegment
from vanpy.utils.utils import concat_audio_files_in_dir


def merge_audio_files(examples_dir: Path) -> None:
    """Merge all audio files in ``examples_dir`` into ``merged.wav``.

    Any existing ``merged.wav`` will be overwritten. Original files are removed
    after merging so that only the combined file is processed by the pipeline.
    """
    output = examples_dir / "merged.wav"

    # Convert any non-wav files to wav so they can be concatenated
    wav_files = []
    for f in examples_dir.iterdir():
        if f.suffix.lower() != ".wav" and f.is_file():
            wav_path = f.with_suffix(".wav")
            try:
                AudioSegment.from_file(f).export(wav_path, format="wav")
            finally:
                f.unlink()
            wav_files.append(wav_path)
        elif f.is_file():
            wav_files.append(f)

    concat_audio_files_in_dir(str(examples_dir), str(output))

    # Remove intermediate wav files, keep only merged output
    for f in wav_files:
        if f != output and f.exists():
            f.unlink()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "src" / "speech_examples"
    merge_audio_files(examples_dir)

    hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN") or os.getenv("hf") or os.getenv("HF_TOKEN")
    if hf_token:
        with open(repo_root / ".env", "w", encoding="utf-8") as env_file:
            env_file.write(f"huggingface_ACCESS_TOKEN={hf_token}\n")

    subprocess.run(["python", "src/run.py"], check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
