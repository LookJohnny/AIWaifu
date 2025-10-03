"""
Voice Sample Converter - MP4 to WAV with Voice Characteristic Analysis
Converts MP4 files to WAV and names them based on voice characteristics
"""
import os
import librosa
import numpy as np
from moviepy import VideoFileClip
from pathlib import Path


def extract_audio_from_mp4(mp4_path, wav_path):
    """Extract audio from MP4 and save as WAV"""
    print(f"Converting {mp4_path} to {wav_path}...")
    video = VideoFileClip(str(mp4_path))
    audio = video.audio
    audio.write_audiofile(str(wav_path), codec='pcm_s16le', fps=22050, nbytes=2, logger=None)
    video.close()
    print(f"[OK] Saved: {wav_path}")


def analyze_voice_characteristics(wav_path):
    """
    Analyze voice characteristics and return descriptive name

    Returns:
        str: Descriptive name like "female_high_soft" or "male_low_energetic"
    """
    print(f"\nAnalyzing {wav_path}...")

    # Load audio
    y, sr = librosa.load(wav_path, sr=22050)

    # 1. Pitch analysis (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    avg_pitch = np.mean(pitch_values) if pitch_values else 0

    # 2. Energy/Volume analysis
    rms = librosa.feature.rms(y=y)[0]
    avg_energy = np.mean(rms)

    # 3. Spectral characteristics
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Classify characteristics
    # Gender (based on pitch)
    if avg_pitch > 180:
        gender = "female"
    elif avg_pitch > 140:
        gender = "androgynous"
    else:
        gender = "male"

    # Pitch range (based on average pitch within gender)
    if gender == "female":
        if avg_pitch > 240:
            pitch_desc = "high"
        elif avg_pitch > 200:
            pitch_desc = "mid"
        else:
            pitch_desc = "low"
    else:  # male or androgynous
        if avg_pitch > 160:
            pitch_desc = "high"
        elif avg_pitch > 120:
            pitch_desc = "mid"
        else:
            pitch_desc = "low"

    # Energy level
    if avg_energy > 0.05:
        energy_desc = "energetic"
    elif avg_energy > 0.02:
        energy_desc = "balanced"
    else:
        energy_desc = "soft"

    # Tone quality (based on spectral centroid)
    if spectral_centroid > 2500:
        tone_desc = "bright"
    elif spectral_centroid > 1500:
        tone_desc = "clear"
    else:
        tone_desc = "warm"

    # Build descriptive name
    descriptive_name = f"{gender}_{pitch_desc}_{tone_desc}"

    # Print analysis
    print(f"  - Average Pitch: {avg_pitch:.1f} Hz")
    print(f"  - Energy Level: {avg_energy:.4f}")
    print(f"  - Spectral Centroid: {spectral_centroid:.1f} Hz")
    print(f"  - Classification: {gender.upper()} voice, {pitch_desc} pitch, {tone_desc} tone")
    print(f"  - Suggested name: {descriptive_name}")

    return descriptive_name


def convert_all_mp4_files(input_dir="voice_samples"):
    """Convert all MP4 files in directory to WAV with descriptive names"""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory {input_dir} not found")
        return

    # Find all MP4 files
    mp4_files = list(input_path.glob("*.MP4")) + list(input_path.glob("*.mp4"))

    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return

    print(f"Found {len(mp4_files)} MP4 file(s)")
    print("=" * 60)

    converted_files = []

    for mp4_file in mp4_files:
        # Convert to WAV first (temp file)
        temp_wav = input_path / f"temp_{mp4_file.stem}.wav"

        try:
            # Extract audio
            extract_audio_from_mp4(mp4_file, temp_wav)

            # Analyze voice characteristics
            descriptive_name = analyze_voice_characteristics(temp_wav)

            # Rename with descriptive name
            final_wav = input_path / f"{descriptive_name}.wav"

            # If file exists, add number suffix
            counter = 1
            while final_wav.exists():
                final_wav = input_path / f"{descriptive_name}_{counter}.wav"
                counter += 1

            # Rename temp file to final name
            temp_wav.rename(final_wav)

            print(f"[OK] Final file: {final_wav.name}")
            print("=" * 60)

            converted_files.append({
                "original": mp4_file.name,
                "converted": final_wav.name,
                "path": str(final_wav)
            })

        except Exception as e:
            print(f"[ERROR] Error converting {mp4_file.name}: {e}")
            if temp_wav.exists():
                temp_wav.unlink()
            continue

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    for item in converted_files:
        print(f"{item['original']} → {item['converted']}")

    print(f"\n[OK] Successfully converted {len(converted_files)}/{len(mp4_files)} files")
    print(f"[OK] All WAV files saved in: {input_path.absolute()}")

    return converted_files


if __name__ == "__main__":
    print("=" * 60)
    print("Voice Sample Converter - MP4 to WAV")
    print("=" * 60)
    print()

    # Convert all MP4 files
    convert_all_mp4_files("voice_samples")
