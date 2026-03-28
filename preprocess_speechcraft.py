"""preprocess_speechcraft.py — 为所有样本生成 SpeechCraft 风格的类别标签

使用 SpeechCraft PitchEnergy + Clustering 的核心逻辑:
1. librosa 提取 mean_pitch + mean_energy
2. 计算 speed = audio_duration / len(text)
3. 分位数分组为 3 类 (低/中/高 or 慢/中/快)
4. 输出 JSON: {utt_id: {pitch_cat: 0/1/2, energy_cat: 0/1/2, speed_cat: 0/1/2}}
"""

import csv
import json
import os
import wave
from typing import Dict

import numpy as np


def _get_wav_duration(path: str) -> float:
    try:
        import torchaudio
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        pass
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _extract_pitch_energy(path: str):
    import librosa
    wav, sr = librosa.load(path, sr=16000)
    if len(wav) == 0:
        return 0.0, 0.0

    pitches, magnitudes = librosa.core.piptrack(y=wav, sr=sr)
    pitch_vals = [np.mean(p[p > 0]) for p in pitches.T if np.sum(p > 0) > 0]
    mean_pitch = float(np.mean(pitch_vals)) if pitch_vals else 0.0

    energy = librosa.feature.rms(y=wav)
    mean_energy = float(np.mean(energy[~np.isnan(energy)]))
    return mean_pitch, mean_energy


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/PQP")
    p.add_argument("--split_root", default="data/PQP/in-scope")
    p.add_argument("--output", default="data/PQP/sc_labels.json")
    args = p.parse_args()

    audio_index: Dict[str, str] = {}
    for root, _, files in os.walk(args.data_root):
        for fn in files:
            if fn.lower().endswith(".wav"):
                utt_id = os.path.splitext(fn)[0]
                audio_index[utt_id] = os.path.join(root, fn)

    samples = []
    for split in ["train.tsv", "dev.tsv", "test.tsv"]:
        tsv_path = os.path.join(args.split_root, split)
        if not os.path.exists(tsv_path):
            continue
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                utt_id = row[0].strip()
                text = row[1].strip()
                if utt_id in audio_index:
                    samples.append({"utt_id": utt_id, "text": text, "wav_path": audio_index[utt_id]})

    print(f"Processing {len(samples)} samples...")

    records = []
    for i, s in enumerate(samples):
        if i % 200 == 0:
            print(f"  [{i}/{len(samples)}]")
        try:
            mean_pitch, mean_energy = _extract_pitch_energy(s["wav_path"])
            duration = _get_wav_duration(s["wav_path"])
            text_len = max(len(s["text"]), 1)
            speed = duration / text_len
        except Exception as e:
            print(f"  WARN: {s['utt_id']} failed: {e}")
            mean_pitch, mean_energy, speed = 0.0, 0.0, 0.0
        records.append({
            "utt_id": s["utt_id"],
            "pitch": mean_pitch,
            "energy": mean_energy,
            "speed": speed,
        })

    pitches = np.array([r["pitch"] for r in records if r["pitch"] > 0])
    energies = np.array([r["energy"] for r in records if r["energy"] > 0])
    speeds = np.array([r["speed"] for r in records if r["speed"] > 0])

    pitch_thresholds = np.percentile(pitches, [33, 67]) if len(pitches) > 0 else [0, 0]
    energy_thresholds = np.percentile(energies, [33, 67]) if len(energies) > 0 else [0, 0]
    speed_thresholds = np.percentile(speeds, [33, 67]) if len(speeds) > 0 else [0, 0]

    print(f"  Pitch thresholds:  {pitch_thresholds}")
    print(f"  Energy thresholds: {energy_thresholds}")
    print(f"  Speed thresholds:  {speed_thresholds}")

    def categorize(val, thresholds):
        if val <= thresholds[0]:
            return 0
        elif val <= thresholds[1]:
            return 1
        else:
            return 2

    result = {}
    for r in records:
        result[r["utt_id"]] = {
            "pitch_cat": categorize(r["pitch"], pitch_thresholds),
            "energy_cat": categorize(r["energy"], energy_thresholds),
            "speed_cat": categorize(r["speed"], speed_thresholds),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(result)} labels to {args.output}")


if __name__ == "__main__":
    main()
