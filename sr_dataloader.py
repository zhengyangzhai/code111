"""sr_dataloader.py — SR 14 类意图分类数据加载器

核心功能:
1. 解析 TextGrid 提取话语级别的 (时间边界, 文本, 意图标签)
2. 从对话 WAV 中按时间裁切单条话语音频
3. 从 PitchTier 中提取对应时间段的韵律特征
4. 提取 TextGrid 韵律标注特征 (stress/boundary/syllable)
5. 支持过采样平衡稀有类别
6. 自动划分 train/dev/test
"""

import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Wav2Vec2 mask_time_length defaults to 10; need >= 10 frames. ~0.2s at 16kHz -> 3200 samples.
MIN_AUDIO_SAMPLES = 3200
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from dataloader import (
    FRAME_ACOUSTIC_DIM,
    PROSODY_FEAT_DIM,
    add_noise,
    extract_frame_acoustic_features,
    load_wav_mono_16k,
    parse_pitchtier,
    speed_perturb,
)

TEXTGRID_FEAT_DIM = 10


# ---------------------------------------------------------------------------
# SR-specific TextGrid tier parser (case-insensitive)
# ---------------------------------------------------------------------------
def _parse_textgrid_tier(filepath: str, tier_name: str) -> List[Tuple[float, float, str]]:
    """Parse a named tier from a TextGrid file (case-insensitive match).

    Returns [(xmin, xmax, text), ...] for the first tier whose name
    matches ``tier_name`` ignoring case.
    """
    intervals: List[Tuple[float, float, str]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return intervals

    in_target_tier = False
    in_intervals = False
    cur_xmin, cur_xmax, cur_text = None, None, None
    for line in lines:
        line = line.strip()
        if line.startswith("name") and tier_name.lower() in line.lower():
            in_target_tier = True
            continue
        if in_target_tier and line.startswith("name"):
            break
        if in_target_tier and "intervals:" in line:
            in_intervals = True
            continue
        if in_target_tier and in_intervals:
            if line.startswith("xmin"):
                cur_xmin = float(line.split("=")[1].strip())
            elif line.startswith("xmax"):
                cur_xmax = float(line.split("=")[1].strip())
            elif line.startswith("text"):
                raw = line.split("=", 1)[1].strip().strip('"')
                cur_text = raw
                if cur_xmin is not None and cur_xmax is not None:
                    intervals.append((cur_xmin, cur_xmax, cur_text))
                cur_xmin, cur_xmax, cur_text = None, None, None
    return intervals


# ---------------------------------------------------------------------------
# SR-specific frame acoustic feature extraction (with time offset)
# ---------------------------------------------------------------------------
def _extract_frame_acoustic_features_sr(
    wav: np.ndarray,
    sr: int = 16000,
    pitchtier_path: Optional[str] = None,
    time_offset: float = 0.0,
) -> np.ndarray:
    """Wrapper around extract_frame_acoustic_features for SR cropped audio.

    Extracts features with the base function, then patches in F0 using the
    correct time offset so PitchTier timestamps align with cropped audio.
    """
    feats = extract_frame_acoustic_features(wav, sr=sr, pitchtier_path=None)

    if pitchtier_path is None or not os.path.exists(pitchtier_path):
        return feats

    hop_length, win_length = 320, 640
    n_frames = feats.shape[0]
    if n_frames < 1:
        return feats

    try:
        _, _, points = parse_pitchtier(pitchtier_path)
    except Exception:
        return feats

    if len(points) < 2:
        return feats

    frame_times = np.arange(n_frames) * hop_length / sr + time_offset
    pt_times = np.array([p[0] for p in points], dtype=np.float32)
    pt_f0 = np.array([p[1] for p in points], dtype=np.float32)
    f0_interp = np.interp(frame_times, pt_times, pt_f0, left=0.0, right=0.0)
    feats[:, 0] = f0_interp
    feats[:, 1] = (f0_interp > 50.0).astype(np.float32)

    voiced_f0 = f0_interp[f0_interp > 50.0]
    mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 1.0
    f0_diff = np.abs(np.diff(f0_interp, prepend=f0_interp[0]))
    feats[:, 8] = f0_diff / (mean_f0 + 1e-6)

    return feats


# ---------------------------------------------------------------------------
# SR 14-class label system
# ---------------------------------------------------------------------------
SR_VALID_LABELS = ["F", "A", "T", "W", "S", "D", "R", "E", "V", "I", "H", "C", "K", "M"]
SR_LABEL2ID = {label: i for i, label in enumerate(SR_VALID_LABELS)}
SR_ID2LABEL = {i: label for i, label in enumerate(SR_VALID_LABELS)}
SR_NUM_LABELS = len(SR_VALID_LABELS)

SR_LAYER_MAP = {
    "F": "factual", "A": "factual", "T": "factual", "W": "factual",
    "S": "attitude", "D": "attitude", "R": "attitude",
    "E": "emotion", "V": "emotion",
    "I": "commitment", "H": "commitment",
    "C": "continuation", "K": "continuation", "M": "continuation",
}


# ---------------------------------------------------------------------------
# TextGrid features scoped to utterance time range
# ---------------------------------------------------------------------------
def extract_textgrid_features_range(
    filepath: str, xmin: float, xmax: float
) -> np.ndarray:
    """Extract TEXTGRID_FEAT_DIM features for a specific utterance time range.

    SR TextGrids use uppercase tier names (SYLLABLE) and lack stress/boundary
    tiers, so those features remain zero.
    """
    feat = np.zeros(TEXTGRID_FEAT_DIM, dtype=np.float32)
    if not filepath or not os.path.exists(filepath):
        return feat

    syllables = _parse_textgrid_tier(filepath, "SYLLABLE")

    speech_syls = [
        (x0, x1, t) for x0, x1, t in syllables
        if t and t not in ("sil", "sp", "")
        and x0 >= xmin - 0.01 and x1 <= xmax + 0.01
    ]
    if len(speech_syls) < 1:
        return feat

    syl_durs = np.array([x1 - x0 for x0, x1, _ in speech_syls], dtype=np.float32)
    total_dur = max(xmax - xmin, 1e-6)
    speech_dur = float(syl_durs.sum())
    mean_syl_dur = float(syl_durs.mean()) if len(syl_durs) > 0 else 1e-6

    last_syl_text = speech_syls[-1][2]
    last_syl_dur = syl_durs[-1]

    if "ma" in last_syl_text.lower():
        feat[0] = last_syl_dur / total_dur

    feat[1] = last_syl_dur / max(mean_syl_dur, 1e-6)
    feat[2] = mean_syl_dur

    if len(syl_durs) > 1:
        feat[3] = float(syl_durs.std()) / max(mean_syl_dur, 1e-6)

    # feat[4-7]: stress/boundary tiers not present in SR data, left as 0

    feat[8] = speech_dur / total_dur
    feat[9] = len(speech_syls) / 10.0

    return feat


# ---------------------------------------------------------------------------
# TextGrid parsing for SR
# ---------------------------------------------------------------------------
def _parse_sr_textgrid(filepath: str) -> List[Dict]:
    """Parse SR TextGrid and return list of utterance dicts.

    Each dict: {xmin, xmax, text, intention, utt_idx}
    """
    utterances = []

    utt_intervals = _parse_textgrid_tier(filepath, "utterance")
    word_intervals = _parse_textgrid_tier(filepath, "WORD")
    int_intervals = _parse_textgrid_tier(filepath, "intention")

    utt_spans = [(x0, x1, t) for x0, x1, t in utt_intervals if t and t.strip() and t.strip().isdigit()]

    for x0_u, x1_u, utt_idx_str in utt_spans:
        text_parts = []
        for x0_w, x1_w, w in word_intervals:
            if w and w not in ("sil", "sp", "") and x0_w >= x0_u - 0.01 and x1_w <= x1_u + 0.01:
                text_parts.append(w)
        text = " ".join(text_parts).strip()

        intention = ""
        for x0_i, x1_i, t in int_intervals:
            if t and x0_i >= x0_u - 0.01 and x1_i <= x1_u + 0.01:
                intention = t.strip()
                break

        if not text or not intention:
            continue

        parts = [p.strip() for p in intention.split("-") if p.strip()]
        valid_parts = [p for p in parts if p in SR_LABEL2ID]
        if not valid_parts:
            continue

        primary_label = valid_parts[0]

        utterances.append({
            "xmin": x0_u,
            "xmax": x1_u,
            "text": text,
            "intention_raw": intention,
            "label": primary_label,
            "label_id": SR_LABEL2ID[primary_label],
            "utt_idx": utt_idx_str,
        })

    return utterances


def build_sr_samples(sr_root: str) -> List[Dict]:
    """Scan SR directory and build a flat list of utterance samples."""
    samples = []
    total_utts = 0
    x_only_dropped = 0
    no_valid_label = 0

    for fn in sorted(os.listdir(sr_root)):
        if not fn.lower().endswith(".textgrid"):
            continue
        base = os.path.splitext(fn)[0]
        tg_path = os.path.join(sr_root, fn)
        wav_path = os.path.join(sr_root, base + ".wav")
        pt_path = os.path.join(sr_root, base + ".PitchTier")

        if not os.path.exists(wav_path):
            continue

        utt_intervals = _parse_textgrid_tier(tg_path, "utterance")
        int_intervals = _parse_textgrid_tier(tg_path, "intention")
        utt_spans = [(x0, x1, t) for x0, x1, t in utt_intervals
                      if t and t.strip() and t.strip().isdigit()]
        for x0_u, x1_u, _ in utt_spans:
            for x0_i, x1_i, t in int_intervals:
                if t and t.strip() and t.strip() not in ("sp", ""):
                    if x0_i >= x0_u - 0.01 and x1_i <= x1_u + 0.01:
                        total_utts += 1
                        raw = t.strip()
                        parts = [p.strip() for p in raw.split("-") if p.strip()]
                        valid = [p for p in parts if p in SR_LABEL2ID]
                        if not valid:
                            if all(p == "X" for p in parts):
                                x_only_dropped += 1
                            else:
                                no_valid_label += 1
                        break

        utts = _parse_sr_textgrid(tg_path)
        for u in utts:
            samples.append({
                "wav_path": wav_path,
                "tg_path": tg_path,
                "pt_path": pt_path if os.path.exists(pt_path) else None,
                "base_name": base,
                **u,
            })

    print(f"[build_sr_samples] Total labeled utterances: {total_utts}")
    print(f"[build_sr_samples] Kept: {len(samples)}")
    print(f"[build_sr_samples] Dropped (X-only): {x_only_dropped}")
    print(f"[build_sr_samples] Dropped (no valid label): {no_valid_label}")

    return samples


def _extract_speaker(base_name: str) -> str:
    """Extract speaker ID from filename like R_subA01_1_lit_他能记得吗 → subA01."""
    parts = base_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return base_name


def split_sr_samples(
    samples: List[Dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Speaker-level split to prevent data leakage.

    All utterances from the same speaker go into the same split.
    Speakers are shuffled then partitioned by ratio.
    """
    rng = random.Random(seed)

    by_speaker: Dict[str, List[Dict]] = {}
    for s in samples:
        speaker = _extract_speaker(s["base_name"])
        by_speaker.setdefault(speaker, []).append(s)

    speakers = sorted(by_speaker.keys())
    rng.shuffle(speakers)

    n = len(speakers)
    n_dev = max(1, int(n * dev_ratio))
    n_test = max(1, int(n * (1 - train_ratio - dev_ratio)))
    n_train = n - n_dev - n_test

    train_spk = speakers[:n_train]
    dev_spk = speakers[n_train:n_train + n_dev]
    test_spk = speakers[n_train + n_dev:]

    train = [s for sp in train_spk for s in by_speaker[sp]]
    dev = [s for sp in dev_spk for s in by_speaker[sp]]
    test = [s for sp in test_spk for s in by_speaker[sp]]

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


# ---------------------------------------------------------------------------
# Prosody extraction for a time range within a PitchTier
# ---------------------------------------------------------------------------
def extract_prosody_features_range(
    pt_path: Optional[str], xmin: float, xmax: float
) -> np.ndarray:
    """Extract PROSODY_FEAT_DIM features for a specific time range."""
    feat = np.zeros(PROSODY_FEAT_DIM, dtype=np.float32)
    if pt_path is None or not os.path.exists(pt_path):
        return feat
    try:
        _, _, points = parse_pitchtier(pt_path)
    except Exception:
        return feat

    pts = [(t, f) for t, f in points if xmin <= t <= xmax]
    if len(pts) < 3:
        return feat

    times = np.array([p[0] for p in pts], dtype=np.float32)
    f0s = np.array([p[1] for p in pts], dtype=np.float32)

    feat[0] = np.mean(f0s)
    feat[1] = np.std(f0s)
    feat[2] = np.min(f0s)
    feat[3] = np.max(f0s)
    feat[4] = feat[3] - feat[2]
    feat[5] = np.median(f0s)

    if len(times) > 1:
        t_c = times - times.mean()
        denom = (t_c ** 2).sum()
        if denom > 1e-10:
            feat[6] = ((t_c * (f0s - f0s.mean())).sum()) / denom

    n_tail = max(3, int(len(f0s) * 0.3))
    tail_f0, tail_t = f0s[-n_tail:], times[-n_tail:]
    if len(tail_t) > 1:
        t_c = tail_t - tail_t.mean()
        denom = (t_c ** 2).sum()
        if denom > 1e-10:
            feat[7] = ((t_c * (tail_f0 - tail_f0.mean())).sum()) / denom

    feat[8] = np.percentile(f0s, 25)
    feat[9] = np.percentile(f0s, 75)
    feat[10] = feat[9] - feat[8]

    duration = xmax - xmin
    if duration > 0 and len(times) > 1:
        feat[11] = (times[-1] - times[0]) / duration

    feat[12] = f0s[-1]
    feat[13] = f0s[0]
    feat[14] = f0s[-1] / (f0s[0] + 1e-6)
    feat[15] = float(np.argmax(f0s)) / max(len(f0s) - 1, 1)
    if len(f0s) > 1:
        delta_f0 = np.diff(f0s)
        feat[16] = np.mean(np.abs(delta_f0))
        feat[17] = np.std(delta_f0)

    return feat


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _build_pqp_lookup(pqp_root: str) -> Dict[str, str]:
    """Build a lookup table mapping (speaker_num_condition) -> PQP base path.

    PQP files live in subdirectories GA/GB/GC/GD with speaker IDs zero-padded
    to 3 digits (subA001), while SR uses 2-digit padding (subA01).
    Text suffixes may also differ between SR and PQP.
    We match by: speaker_letter + speaker_number + question_number + condition.
    """
    import re as _re
    lookup: Dict[str, str] = {}
    for grp in ["GA", "GB", "GC", "GD"]:
        gpath = os.path.join(pqp_root, grp)
        if not os.path.isdir(gpath):
            continue
        for f in os.listdir(gpath):
            if not f.endswith(".wav"):
                continue
            m = _re.match(r"Q_(sub[A-D])(\d+)_(\d+)_(lit|deep)_", f)
            if m:
                letter, num, qnum, cond = m.group(1), m.group(2), m.group(3), m.group(4)
                key = f"{letter}{int(num)}_{qnum}_{cond}"
                lookup[key] = os.path.join(gpath, f[:-4])  # strip .wav
    return lookup


def _get_pqp_path(sr_base_name: str, pqp_root: str,
                   pqp_lookup: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """Find PQP file paths matching an SR base_name.

    Uses a pre-built lookup table to handle:
    - PQP files in subdirectories (GA/GB/GC/GD)
    - Different zero-padding (subA01 -> subA001)
    - Possible text suffix differences
    """
    import re as _re
    if pqp_root is None:
        return None

    if pqp_lookup is not None:
        # Match by speaker + question number + condition
        m = _re.match(r"R_(sub[A-D])(\d+)_(\d+)_(lit|deep)_", sr_base_name)
        if m:
            letter, num, qnum, cond = m.group(1), m.group(2), m.group(3), m.group(4)
            key = f"{letter}{int(num)}_{qnum}_{cond}"
            if key in pqp_lookup:
                base = pqp_lookup[key]
                wav_path = base + ".wav"
                tg_path = base + ".TextGrid"
                pt_path = base + ".PitchTier"
                return {
                    "tg_path": tg_path if os.path.exists(tg_path) else None,
                    "wav_path": wav_path,
                    "pt_path": pt_path if os.path.exists(pt_path) else None,
                }
        return None

    # Fallback: old logic (flat directory, same naming)
    if sr_base_name.startswith("R_"):
        pqp_base = "Q_" + sr_base_name[2:]
    else:
        pqp_base = sr_base_name

    tg_path = os.path.join(pqp_root, pqp_base + ".TextGrid")
    wav_path = os.path.join(pqp_root, pqp_base + ".wav")
    pt_path = os.path.join(pqp_root, pqp_base + ".PitchTier")

    if not os.path.exists(tg_path) and not os.path.exists(wav_path):
        return None

    return {
        "tg_path": tg_path,
        "wav_path": wav_path,
        "pt_path": pt_path if os.path.exists(pt_path) else None,
    }


class SRDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        tokenizer_name: str = "bert-base-chinese",
        feature_extractor_name: str = "TencentGameMate/chinese-wav2vec2-base",
        text_max_length: int = 128,
        audio_sampling_rate: int = 16000,
        augment: bool = False,
        pqp_root: Optional[str] = None,
    ):
        self.samples = samples
        self.text_max_length = text_max_length
        self.audio_sampling_rate = audio_sampling_rate
        self.augment = augment
        self.pqp_root = pqp_root
        self.has_pqp = pqp_root is not None

        # Build PQP lookup table for proper path matching
        if pqp_root is not None:
            self.pqp_lookup = _build_pqp_lookup(pqp_root)
            print(f"[SRDataset] PQP lookup built: {len(self.pqp_lookup)} entries")
        else:
            self.pqp_lookup = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]

        # ---- Crop audio to utterance time range ----
        full_wav = load_wav_mono_16k(item["wav_path"], target_sr=self.audio_sampling_rate)
        sr = self.audio_sampling_rate
        start_sample = int(item["xmin"] * sr)
        end_sample = int(item["xmax"] * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(full_wav), end_sample)
        wav = full_wav[start_sample:end_sample]

        if len(wav) < 160:
            wav = np.zeros(160, dtype=np.float32)

        # ---- Augmentation ----
        if self.augment:
            if random.random() < 0.4:
                wav = speed_perturb(wav, random.uniform(0.9, 1.1))
            if random.random() < 0.4:
                wav = add_noise(wav, snr_db=random.uniform(15.0, 25.0))
            if random.random() < 0.3:
                wav = np.clip(wav * random.uniform(0.7, 1.3), -1.0, 1.0).astype(np.float32)

        # ---- Prosody features (from PitchTier, cropped to utterance range) ----
        prosody = extract_prosody_features_range(item["pt_path"], item["xmin"], item["xmax"])

        # ---- TextGrid features (scoped to utterance time range) ----
        tg_feat = extract_textgrid_features_range(item["tg_path"], item["xmin"], item["xmax"])

        # ---- Text encoding ----
        text_feat = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.text_max_length,
            padding=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # ---- Audio encoding ----
        audio_feat = self.feature_extractor(
            wav,
            sampling_rate=self.audio_sampling_rate,
            padding=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # ---- Frame acoustic features (with time offset for cropped audio) ----
        frame_feats = _extract_frame_acoustic_features_sr(
            wav, sr=self.audio_sampling_rate,
            pitchtier_path=item["pt_path"],
            time_offset=item["xmin"],
        )

        # ---- SpeechCraft placeholder (not available for SR) ----
        sc_feat = np.zeros(3, dtype=np.int64)

        result = {
            "text_input_ids": text_feat["input_ids"].squeeze(0),
            "text_attention_mask": text_feat["attention_mask"].squeeze(0),
            "audio_input_values": audio_feat["input_values"].squeeze(0),
            "audio_attention_mask": audio_feat["attention_mask"].squeeze(0),
            "prosody_features": torch.tensor(prosody, dtype=torch.float32),
            "textgrid_features": torch.tensor(tg_feat, dtype=torch.float32),
            "speechcraft_features": torch.tensor(sc_feat, dtype=torch.long),
            "frame_acoustic_features": torch.tensor(frame_feats, dtype=torch.float32),
            "label": torch.tensor(item["label_id"], dtype=torch.long),
        }

        # ---- PQP context features (if pqp_root provided) ----
        if self.has_pqp:
            pqp_paths = _get_pqp_path(item["base_name"], self.pqp_root,
                                       pqp_lookup=self.pqp_lookup)
            if pqp_paths is not None and os.path.exists(pqp_paths.get("wav_path", "")):
                # Load PQP audio (full utterance, not cropped)
                pqp_wav = load_wav_mono_16k(pqp_paths["wav_path"], target_sr=self.audio_sampling_rate)
                if len(pqp_wav) < 160:
                    pqp_wav = np.zeros(160, dtype=np.float32)

                # Extract PQP question text from filename
                import re as _re
                pqp_basename = os.path.basename(pqp_paths["wav_path"])
                pqp_text_match = _re.match(r"Q_sub[A-D]\d+_\d+_(?:lit|deep)_(.+)\.wav", pqp_basename)
                pqp_question_text = pqp_text_match.group(1) if pqp_text_match else item["text"]

                # PQP text encoding (use PQP question text as context)
                pqp_text_feat = self.tokenizer(
                    pqp_question_text,
                    truncation=True,
                    max_length=self.text_max_length,
                    padding=False,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                # PQP audio encoding (full audio)
                pqp_audio_feat = self.feature_extractor(
                    pqp_wav,
                    sampling_rate=self.audio_sampling_rate,
                    padding=False,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                # PQP prosody features (full audio prosody)
                pqp_prosody = np.zeros(PROSODY_FEAT_DIM, dtype=np.float32)
                if pqp_paths.get("pt_path") and os.path.exists(pqp_paths["pt_path"]):
                    # Use full audio duration for prosody
                    pqp_wav_len = len(pqp_wav) / self.audio_sampling_rate
                    pqp_prosody = extract_prosody_features_range(pqp_paths["pt_path"], 0.0, pqp_wav_len)

                # PQP frame acoustic features
                pqp_frame_feats = extract_frame_acoustic_features(pqp_wav, sr=self.audio_sampling_rate)

                result.update({
                    "pqp_text_input_ids": pqp_text_feat["input_ids"].squeeze(0),
                    "pqp_text_attention_mask": pqp_text_feat["attention_mask"].squeeze(0),
                    "pqp_audio_input_values": pqp_audio_feat["input_values"].squeeze(0),
                    "pqp_audio_attention_mask": pqp_audio_feat["attention_mask"].squeeze(0),
                    "pqp_prosody_features": torch.tensor(pqp_prosody, dtype=torch.float32),
                    "pqp_frame_acoustic_features": torch.tensor(pqp_frame_feats, dtype=torch.float32),
                })
            else:
                # PQP not found, use zero tensors as fallback
                result.update({
                    "pqp_text_input_ids": text_feat["input_ids"].squeeze(0).clone(),
                    "pqp_text_attention_mask": torch.zeros_like(text_feat["attention_mask"].squeeze(0)),
                    "pqp_audio_input_values": audio_feat["input_values"].squeeze(0).clone(),
                    "pqp_audio_attention_mask": torch.zeros_like(audio_feat["attention_mask"].squeeze(0)),
                    "pqp_prosody_features": torch.zeros(PROSODY_FEAT_DIM, dtype=torch.float32),
                    "pqp_frame_acoustic_features": torch.tensor(frame_feats[:1, :], dtype=torch.float32) if frame_feats.shape[0] > 0 else torch.zeros(1, FRAME_ACOUSTIC_DIM, dtype=torch.float32),
                })

        return result


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
@dataclass
class SRCollator:
    tokenizer: AutoTokenizer
    feature_extractor: Wav2Vec2FeatureExtractor
    has_pqp: bool = False

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        text_features = [
            {"input_ids": x["text_input_ids"], "attention_mask": x["text_attention_mask"]}
            for x in batch
        ]
        padded_text = self.tokenizer.pad(text_features, padding=True, return_tensors="pt")

        for x in batch:
            av = x["audio_input_values"]
            if av.shape[-1] < MIN_AUDIO_SAMPLES:
                pad_len = MIN_AUDIO_SAMPLES - av.shape[-1]
                x["audio_input_values"] = F.pad(av, (0, pad_len))

        audio_features = [{"input_values": x["audio_input_values"]} for x in batch]
        padded_audio = self.feature_extractor.pad(
            audio_features, padding=True, return_attention_mask=True, return_tensors="pt"
        )

        prosody = torch.stack([x["prosody_features"] for x in batch], dim=0)
        textgrid = torch.stack([x["textgrid_features"] for x in batch], dim=0)
        speechcraft = torch.stack([x["speechcraft_features"] for x in batch], dim=0)
        labels = torch.stack([x["label"] for x in batch], dim=0)

        frame_feats_list = [x["frame_acoustic_features"] for x in batch]
        max_frame_len = max(f.shape[0] for f in frame_feats_list)
        frame_dim = frame_feats_list[0].shape[-1]
        padded_frame_feats = torch.zeros(len(batch), max_frame_len, frame_dim)
        frame_feat_mask = torch.zeros(len(batch), max_frame_len)
        for i, f in enumerate(frame_feats_list):
            L = f.shape[0]
            padded_frame_feats[i, :L, :] = f
            frame_feat_mask[i, :L] = 1.0

        result = {
            "text_input_ids": padded_text["input_ids"],
            "text_attention_mask": padded_text["attention_mask"],
            "audio_input_values": padded_audio["input_values"],
            "audio_attention_mask": padded_audio["attention_mask"],
            "prosody_features": prosody,
            "textgrid_features": textgrid,
            "speechcraft_features": speechcraft,
            "frame_acoustic_features": padded_frame_feats,
            "frame_acoustic_mask": frame_feat_mask,
            "labels": labels,
        }

        # PQP features (if available)
        if self.has_pqp:
            # PQP text
            pqp_text_features = [
                {"input_ids": x["pqp_text_input_ids"], "attention_mask": x["pqp_text_attention_mask"]}
                for x in batch
            ]
            padded_pqp_text = self.tokenizer.pad(pqp_text_features, padding=True, return_tensors="pt")

            for x in batch:
                av = x["pqp_audio_input_values"]
                if av.shape[-1] < MIN_AUDIO_SAMPLES:
                    pad_len = MIN_AUDIO_SAMPLES - av.shape[-1]
                    x["pqp_audio_input_values"] = F.pad(av, (0, pad_len))

            # PQP audio
            pqp_audio_features = [{"input_values": x["pqp_audio_input_values"]} for x in batch]
            padded_pqp_audio = self.feature_extractor.pad(
                pqp_audio_features, padding=True, return_attention_mask=True, return_tensors="pt"
            )

            # PQP prosody
            pqp_prosody = torch.stack([x["pqp_prosody_features"] for x in batch], dim=0)

            # PQP frame acoustic
            pqp_frame_feats_list = [x["pqp_frame_acoustic_features"] for x in batch]
            max_pqp_frame_len = max(f.shape[0] for f in pqp_frame_feats_list) if pqp_frame_feats_list else 1
            pqp_frame_dim = pqp_frame_feats_list[0].shape[-1] if pqp_frame_feats_list else FRAME_ACOUSTIC_DIM
            padded_pqp_frame_feats = torch.zeros(len(batch), max_pqp_frame_len, pqp_frame_dim)
            pqp_frame_feat_mask = torch.zeros(len(batch), max_pqp_frame_len)
            for i, f in enumerate(pqp_frame_feats_list):
                L = f.shape[0]
                padded_pqp_frame_feats[i, :L, :] = f
                pqp_frame_feat_mask[i, :L] = 1.0

            result.update({
                "pqp_text_input_ids": padded_pqp_text["input_ids"],
                "pqp_text_attention_mask": padded_pqp_text["attention_mask"],
                "pqp_audio_input_values": padded_pqp_audio["input_values"],
                "pqp_audio_attention_mask": padded_pqp_audio["attention_mask"],
                "pqp_prosody_features": pqp_prosody,
                "pqp_frame_acoustic_features": padded_pqp_frame_feats,
                "pqp_frame_acoustic_mask": pqp_frame_feat_mask,
            })

        return result


# ---------------------------------------------------------------------------
# Weighted sampler for class balancing
# ---------------------------------------------------------------------------
def build_weighted_sampler(samples: List[Dict]) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples rare classes."""
    labels = [s["label_id"] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (SR_NUM_LABELS * counts[l]) for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def create_sr_dataloader(
    samples: List[Dict],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    tokenizer_name: str = "bert-base-chinese",
    feature_extractor_name: str = "TencentGameMate/chinese-wav2vec2-base",
    text_max_length: int = 128,
    audio_sampling_rate: int = 16000,
    augment: bool = False,
    use_weighted_sampler: bool = False,
    pqp_root: Optional[str] = None,
) -> DataLoader:
    ds = SRDataset(
        samples=samples,
        tokenizer_name=tokenizer_name,
        feature_extractor_name=feature_extractor_name,
        text_max_length=text_max_length,
        audio_sampling_rate=audio_sampling_rate,
        augment=augment,
        pqp_root=pqp_root,
    )
    collator = SRCollator(ds.tokenizer, ds.feature_extractor, has_pqp=ds.has_pqp)

    sampler = None
    if use_weighted_sampler and not shuffle:
        shuffle = False
    if use_weighted_sampler:
        sampler = build_weighted_sampler(samples)
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
