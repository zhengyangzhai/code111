"""
dataloader.py  —  增强版数据加载器
改进：
1. 解析 PitchTier 提取 F0 韵律统计特征（均值/标准差/范围/斜率/尾部升降等）
2. 支持音频数据增强（速度扰动 + 加噪）
3. 构建 PitchTier 索引，与音频索引对齐
"""

import csv
import json
import os
import random
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

# Wav2Vec2 mask_time_length requires sequence_length >= 10; ~0.2s at 16kHz -> 3200 samples.
MIN_AUDIO_SAMPLES = 3200


LABEL2ID = {"lit": 0, "deep": 1}
ID2LABEL = {0: "lit", 1: "deep"}

# ---- 韵律特征维度 ----
PROSODY_FEAT_DIM = 18

# ---- 帧级声学特征维度 (eGeMAPS-inspired, 用于双流注入) ----
# 0:F0  1:voiced  2:logRMS  3:ZCR  4:spectral_centroid
# 5:spectral_rolloff  6:spectral_flux  7:spectral_tilt
# 8:jitter  9:shimmer
FRAME_ACOUSTIC_DIM = 10


# ===========================================================================
#  PitchTier 解析 & 韵律特征
# ===========================================================================
def parse_pitchtier(filepath: str) -> Tuple[float, float, List[Tuple[float, float]]]:
    """解析 Praat PitchTier 文件，返回 (xmin, xmax, [(time, f0_hz), ...])"""
    points: List[Tuple[float, float]] = []
    xmin, xmax = 0.0, 0.0
    cur_time = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("xmin"):
                xmin = float(line.split("=")[1])
            elif line.startswith("xmax"):
                xmax = float(line.split("=")[1])
            elif line.startswith("number"):
                cur_time = float(line.split("=")[1])
            elif line.startswith("value") and cur_time is not None:
                val = float(line.split("=")[1])
                points.append((cur_time, val))
                cur_time = None
    return xmin, xmax, points


def extract_prosody_features(filepath: str) -> np.ndarray:
    """
    从 PitchTier 文件提取 12 维韵律统计特征:
      [ f0_mean, f0_std, f0_min, f0_max, f0_range,
        f0_median, f0_slope, f0_final_slope,
        f0_q25, f0_q75, f0_iqr, voiced_ratio ]
    """
    feat = np.zeros(PROSODY_FEAT_DIM, dtype=np.float32)
    try:
        xmin, xmax, points = parse_pitchtier(filepath)
    except Exception:
        return feat

    if len(points) < 3:
        return feat

    times = np.array([p[0] for p in points], dtype=np.float32)
    f0s = np.array([p[1] for p in points], dtype=np.float32)

    # 基本统计
    feat[0] = np.mean(f0s)
    feat[1] = np.std(f0s)
    feat[2] = np.min(f0s)
    feat[3] = np.max(f0s)
    feat[4] = feat[3] - feat[2]  # range
    feat[5] = np.median(f0s)

    # 全局斜率 (线性回归)
    if len(times) > 1:
        t_centered = times - times.mean()
        denom = (t_centered ** 2).sum()
        if denom > 1e-10:
            feat[6] = ((t_centered * (f0s - f0s.mean())).sum()) / denom
        else:
            feat[6] = 0.0

    # 尾部斜率（最后 30% 的 F0 趋势 — 对疑问句升调至关重要）
    n_tail = max(3, int(len(f0s) * 0.3))
    tail_f0 = f0s[-n_tail:]
    tail_t = times[-n_tail:]
    if len(tail_t) > 1:
        t_c = tail_t - tail_t.mean()
        denom = (t_c ** 2).sum()
        if denom > 1e-10:
            feat[7] = ((t_c * (tail_f0 - tail_f0.mean())).sum()) / denom
        else:
            feat[7] = 0.0

    # 分位数
    feat[8] = np.percentile(f0s, 25)
    feat[9] = np.percentile(f0s, 75)
    feat[10] = feat[9] - feat[8]  # IQR

    # 有声比例 = voiced_duration / total_duration
    duration = xmax - xmin
    if duration > 0 and len(times) > 1:
        voiced_dur = times[-1] - times[0]
        feat[11] = voiced_dur / duration
    else:
        feat[11] = 0.0

    # ---- 新增特征 (12-17)：对 lit/deep 韵律判别至关重要 ----
    feat[12] = f0s[-1]                                      # f0_offset (结尾 F0)
    feat[13] = f0s[0]                                       # f0_onset (起始 F0)
    feat[14] = f0s[-1] / (f0s[0] + 1e-6)                   # offset/onset 比值
    feat[15] = float(np.argmax(f0s)) / max(len(f0s) - 1, 1) # F0 峰值位置 (0~1)
    if len(f0s) > 1:
        delta_f0 = np.diff(f0s)
        feat[16] = np.mean(np.abs(delta_f0))                # 平均绝对 ΔF0
        feat[17] = np.std(delta_f0)                         # ΔF0 标准差

    return feat


def build_pitchtier_index(data_root: str) -> Dict[str, str]:
    """遍历数据目录，建立 utt_id → PitchTier 路径的映射"""
    index: Dict[str, str] = {}
    for root, _, files in os.walk(data_root):
        for fn in files:
            if fn.lower().endswith(".pitchtier"):
                utt_id = os.path.splitext(fn)[0]
                index[utt_id] = os.path.join(root, fn)
    return index


# ===========================================================================
#  帧级声学特征提取 (eGeMAPS-inspired)
# ===========================================================================
def extract_frame_acoustic_features(
    wav: np.ndarray,
    sr: int = 16000,
    pitchtier_path: Optional[str] = None,
    hop_length: int = 320,
    win_length: int = 640,
) -> np.ndarray:
    """从原始波形提取帧级物理声学特征 (eGeMAPS-inspired)。

    hop_length=320 与 Wav2Vec2 CNN 下采样因子一致 (20ms @ 16kHz)，
    保证帧数与 Wav2Vec2 输出序列长度近似对齐。

    每帧 FRAME_ACOUSTIC_DIM=10 维特征：
        0: F0 基频 (Hz, 来自 PitchTier 插值)
        1: voiced flag (1=有声, 0=无声)
        2: log RMS energy (对数能量)
        3: zero crossing rate (过零率)
        4: spectral centroid (归一化谱重心)
        5: spectral rolloff 85% (归一化谱滚降)
        6: spectral flux (帧间频谱变化)
        7: spectral tilt (低/高频能量比 log)
        8: jitter (基频微扰 |ΔF0|/mean_F0)
        9: shimmer (振幅微扰 |Δenergy|/mean_energy)

    Returns:
        features: np.ndarray [num_frames, FRAME_ACOUSTIC_DIM]
    """
    n_frames = max(1, 1 + (len(wav) - win_length) // hop_length)
    features = np.zeros((n_frames, FRAME_ACOUSTIC_DIM), dtype=np.float32)

    if len(wav) < win_length:
        return features

    # === F0 from PitchTier (feature 0, 1) ===
    frame_times = np.arange(n_frames) * hop_length / sr
    if pitchtier_path is not None:
        try:
            _, _, points = parse_pitchtier(pitchtier_path)
            if len(points) >= 2:
                pt_times = np.array([p[0] for p in points], dtype=np.float32)
                pt_f0 = np.array([p[1] for p in points], dtype=np.float32)
                f0_interp = np.interp(frame_times, pt_times, pt_f0, left=0.0, right=0.0)
                features[:, 0] = f0_interp
                features[:, 1] = (f0_interp > 50.0).astype(np.float32)
        except Exception:
            pass

    # === 尝试用 librosa 提取频谱特征 (更准确) ===
    try:
        import librosa
        _HAS_LIBROSA = True
    except ImportError:
        _HAS_LIBROSA = False

    if _HAS_LIBROSA:
        # RMS energy (feature 2)
        rms = librosa.feature.rms(y=wav, frame_length=win_length, hop_length=hop_length)[0]
        n = min(n_frames, len(rms))
        features[:n, 2] = np.log1p(rms[:n] * 1000)

        # Zero crossing rate (feature 3)
        zcr = librosa.feature.zero_crossing_rate(
            y=wav, frame_length=win_length, hop_length=hop_length
        )[0]
        n = min(n_frames, len(zcr))
        features[:n, 3] = zcr[:n]

        # STFT for spectral features
        S = np.abs(librosa.stft(wav, n_fft=win_length, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=win_length)

        # Spectral centroid (feature 4, normalized)
        sc = librosa.feature.spectral_centroid(S=S ** 2, sr=sr, n_fft=win_length)[0]
        n = min(n_frames, len(sc))
        features[:n, 4] = sc[:n] / sr

        # Spectral rolloff (feature 5, normalized)
        sro = librosa.feature.spectral_rolloff(
            S=S ** 2, sr=sr, n_fft=win_length, roll_percent=0.85
        )[0]
        n = min(n_frames, len(sro))
        features[:n, 5] = sro[:n] / sr

        # Spectral flux (feature 6)
        if S.shape[1] > 1:
            flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
            flux = np.concatenate([[0.0], flux])
            n = min(n_frames, len(flux))
            features[:n, 6] = flux[:n]

        # Spectral tilt (feature 7): log(low_energy / high_energy)
        freq_boundary = np.searchsorted(freqs, 1000.0)
        if 0 < freq_boundary < len(freqs) and S.shape[1] > 0:
            low_e = np.sum(S[:freq_boundary, :] ** 2, axis=0)
            high_e = np.sum(S[freq_boundary:, :] ** 2, axis=0) + 1e-10
            tilt = np.log1p(low_e / high_e)
            n = min(n_frames, len(tilt))
            features[:n, 7] = tilt[:n]
    else:
        # --- Numpy fallback (仅基本能量与过零率) ---
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + win_length, len(wav))
            frame = wav[start:end]
            if len(frame) == 0:
                continue
            features[i, 2] = np.log1p(np.sqrt(np.mean(frame ** 2)) * 1000)
            features[i, 3] = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0

    # === Jitter (feature 8): F0 逐帧微扰 ===
    f0_vals = features[:, 0]
    f0_diff = np.abs(np.diff(f0_vals, prepend=f0_vals[0]))
    voiced_f0 = f0_vals[f0_vals > 50.0]
    mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 1.0
    features[:, 8] = f0_diff / (mean_f0 + 1e-6)

    # === Shimmer (feature 9): 能量逐帧微扰 ===
    energy_vals = features[:, 2]
    energy_diff = np.abs(np.diff(energy_vals, prepend=energy_vals[0]))
    mean_energy = np.mean(energy_vals) + 1e-6
    features[:, 9] = energy_diff / mean_energy

    return features


# ===========================================================================
#  音频 I/O
# ===========================================================================
def _normalize_waveform(wav: np.ndarray) -> np.ndarray:
    if wav.dtype.kind in {"i", "u"}:
        max_val = np.iinfo(wav.dtype).max
        wav = wav.astype(np.float32) / float(max_val)
    else:
        wav = wav.astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    return wav


def _resample_linear(wav: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return wav if wav.dtype == np.float32 else wav.astype(np.float32)
    src_len = len(wav)
    if src_len == 0:
        return wav
    tgt_len = int(round(src_len * float(tgt_sr) / float(src_sr)))
    if tgt_len <= 1:
        return np.array([float(wav[0])], dtype=np.float32)
    wav = np.asarray(wav, dtype=np.float32)

    # 长音频用 torchaudio 重采样，避免 np.interp 分配大块 float64 导致内存不足
    if src_len > 65536 or tgt_len > 65536:
        try:
            import torchaudio
            t = torch.from_numpy(wav).unsqueeze(0)
            t = torchaudio.functional.resample(t, src_sr, tgt_sr)
            return t.squeeze(0).numpy().astype(np.float32, copy=False)
        except Exception:
            pass

    old_idx = np.linspace(0.0, 1.0, num=src_len, endpoint=True, dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=tgt_len, endpoint=True, dtype=np.float32)
    chunk_size = 65536
    if tgt_len <= chunk_size:
        out = np.interp(new_idx, old_idx, wav).astype(np.float32, copy=False)
        return out
    out = np.empty(tgt_len, dtype=np.float32)
    for start in range(0, tgt_len, chunk_size):
        end = min(start + chunk_size, tgt_len)
        chunk = np.interp(new_idx[start:end], old_idx, wav).astype(np.float32, copy=False)
        out[start:end] = chunk
    return out


def load_wav_mono_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    try:
        import torchaudio
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0).numpy().astype(np.float32)
        audio = _resample_linear(audio, sr, target_sr)
        return np.clip(audio, -1.0, 1.0)
    except Exception:
        pass

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        wav = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        wav = np.frombuffer(raw, dtype=np.int32)
    elif sampwidth == 1:
        wav = np.frombuffer(raw, dtype=np.uint8)
        wav = wav.astype(np.int16) - 128
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if n_channels > 1:
        wav = wav.reshape(-1, n_channels).mean(axis=1)

    wav = _normalize_waveform(wav)
    wav = _resample_linear(wav, sr, target_sr)
    return wav


def build_audio_index(audio_root: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root, _, files in os.walk(audio_root):
        for fn in files:
            if fn.lower().endswith(".wav"):
                utt_id = os.path.splitext(fn)[0]
                index[utt_id] = os.path.join(root, fn)
    return index


# ===========================================================================
#  数据增强
# ===========================================================================
def speed_perturb(wav: np.ndarray, factor: float) -> np.ndarray:
    """通过线性插值实现速度扰动，factor>1 加速，factor<1 减速"""
    if abs(factor - 1.0) < 1e-3:
        return wav
    src_len = len(wav)
    tgt_len = int(round(src_len / factor))
    if tgt_len <= 1:
        return wav
    old_idx = np.linspace(0.0, 1.0, num=src_len, endpoint=True)
    new_idx = np.linspace(0.0, 1.0, num=tgt_len, endpoint=True)
    return np.interp(new_idx, old_idx, wav).astype(np.float32)


def add_noise(wav: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """添加高斯白噪声"""
    signal_power = np.mean(wav ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(wav)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(wav + noise, -1.0, 1.0)


# ===========================================================================
#  Dataset
# ===========================================================================
class PQPDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        audio_root: str,
        tokenizer_name: str = "bert-base-chinese",
        feature_extractor_name: str = "TencentGameMate/chinese-wav2vec2-base",
        text_max_length: int = 64,
        audio_sampling_rate: int = 16000,
        audio_index: Optional[Dict[str, str]] = None,
        pitchtier_index: Optional[Dict[str, str]] = None,
        sc_labels: Optional[Dict[str, Dict]] = None,
        augment: bool = False,
    ):
        self.tsv_path = tsv_path
        self.audio_root = audio_root
        self.text_max_length = text_max_length
        self.audio_sampling_rate = audio_sampling_rate
        self.augment = augment

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_name)
        self.audio_index = audio_index if audio_index is not None else build_audio_index(audio_root)
        self.pitchtier_index = pitchtier_index if pitchtier_index is not None else build_pitchtier_index(audio_root)
        self.sc_labels = sc_labels

        all_samples = self._read_tsv(tsv_path)
        self.samples = [s for s in all_samples if s["utt_id"] in self.audio_index]
        skipped = len(all_samples) - len(self.samples)
        if skipped > 0:
            print(f"[PQPDataset] WARNING: skipped {skipped} samples with missing audio in {tsv_path}")

    @staticmethod
    def _read_tsv(tsv_path: str) -> List[Dict]:
        rows: List[Dict] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_no, row in enumerate(reader, start=1):
                if not row or len(row) < 3:
                    continue
                utt_id = row[0].strip()
                text = row[1].strip()
                label_name = row[2].strip().lower()
                if label_name not in LABEL2ID:
                    raise ValueError(f"Invalid label '{label_name}' at {tsv_path}:{line_no}")
                rows.append({"utt_id": utt_id, "text": text, "label": LABEL2ID[label_name]})
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        utt_id = item["utt_id"]
        text = item["text"]
        label = item["label"]

        # ---- 音频 ----
        wav_path = self.audio_index.get(utt_id)
        if wav_path is None:
            raise FileNotFoundError(f"Audio not found for id: {utt_id}")
        wav = load_wav_mono_16k(wav_path, target_sr=self.audio_sampling_rate)

        # ---- 数据增强（仅训练集） ----
        if self.augment:
            r = random.random()
            if r < 0.3:
                factor = random.choice([0.9, 0.95, 1.05, 1.1])
                wav = speed_perturb(wav, factor)
            elif r < 0.5:
                snr = random.uniform(15.0, 25.0)
                wav = add_noise(wav, snr_db=snr)
            elif r < 0.65:
                gain = random.uniform(0.7, 1.3)
                wav = np.clip(wav * gain, -1.0, 1.0).astype(np.float32)

        # ---- 韵律特征 ----
        pt_path = self.pitchtier_index.get(utt_id)
        if pt_path is not None:
            prosody = extract_prosody_features(pt_path)
        else:
            prosody = np.zeros(PROSODY_FEAT_DIM, dtype=np.float32)

        # ---- 文本编码 ----
        text_feat = self.tokenizer(
            text,
            truncation=True,
            max_length=self.text_max_length,
            padding=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # ---- 音频编码 ----
        audio_feat = self.feature_extractor(
            wav,
            sampling_rate=self.audio_sampling_rate,
            padding=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # ---- 帧级声学特征 (eGeMAPS-inspired, 用于双流注入 + 差分建模) ----
        frame_feats = extract_frame_acoustic_features(
            wav, sr=self.audio_sampling_rate, pitchtier_path=pt_path,
        )

        # ---- SpeechCraft 类别特征 (pitch_cat, energy_cat, speed_cat) ----
        if self.sc_labels is not None and utt_id in self.sc_labels:
            sc = self.sc_labels[utt_id]
            sc_feat = np.array([sc["pitch_cat"], sc["energy_cat"], sc["speed_cat"]], dtype=np.int64)
        else:
            sc_feat = np.zeros(3, dtype=np.int64)

        return {
            "utt_id": utt_id,
            "text": text,
            "text_input_ids": text_feat["input_ids"].squeeze(0),
            "text_attention_mask": text_feat["attention_mask"].squeeze(0),
            "audio_input_values": audio_feat["input_values"].squeeze(0),
            "audio_attention_mask": audio_feat["attention_mask"].squeeze(0),
            "prosody_features": torch.tensor(prosody, dtype=torch.float32),
            "frame_acoustic_features": torch.tensor(frame_feats, dtype=torch.float32),
            "speechcraft_features": torch.tensor(sc_feat, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ===========================================================================
#  Collator
# ===========================================================================
@dataclass
class PQPCollator:
    tokenizer: AutoTokenizer
    feature_extractor: Wav2Vec2FeatureExtractor

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
        speechcraft = torch.stack([x["speechcraft_features"] for x in batch], dim=0)
        labels = torch.stack([x["label"] for x in batch], dim=0)
        utt_ids = [x["utt_id"] for x in batch]
        texts = [x["text"] for x in batch]

        # ---- 帧级声学特征 padding ----
        frame_feats_list = [x["frame_acoustic_features"] for x in batch]
        max_frame_len = max(f.shape[0] for f in frame_feats_list)
        frame_dim = frame_feats_list[0].shape[-1]  # FRAME_ACOUSTIC_DIM
        padded_frame_feats = torch.zeros(len(batch), max_frame_len, frame_dim)
        frame_feat_mask = torch.zeros(len(batch), max_frame_len)
        for i, f in enumerate(frame_feats_list):
            L = f.shape[0]
            padded_frame_feats[i, :L, :] = f
            frame_feat_mask[i, :L] = 1.0

        return {
            "text_input_ids": padded_text["input_ids"],
            "text_attention_mask": padded_text["attention_mask"],
            "audio_input_values": padded_audio["input_values"],
            "audio_attention_mask": padded_audio["attention_mask"],
            "prosody_features": prosody,
            "speechcraft_features": speechcraft,
            "frame_acoustic_features": padded_frame_feats,
            "frame_acoustic_mask": frame_feat_mask,
            "labels": labels,
            "utt_ids": utt_ids,
            "texts": texts,
        }


def create_dataloader(
    tsv_path: str,
    audio_root: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    tokenizer_name: str = "bert-base-chinese",
    feature_extractor_name: str = "TencentGameMate/chinese-wav2vec2-base",
    text_max_length: int = 64,
    audio_sampling_rate: int = 16000,
    audio_index: Optional[Dict[str, str]] = None,
    pitchtier_index: Optional[Dict[str, str]] = None,
    sc_labels: Optional[Dict[str, Dict]] = None,
    augment: bool = False,
) -> DataLoader:
    ds = PQPDataset(
        tsv_path=tsv_path,
        audio_root=audio_root,
        tokenizer_name=tokenizer_name,
        feature_extractor_name=feature_extractor_name,
        text_max_length=text_max_length,
        audio_sampling_rate=audio_sampling_rate,
        audio_index=audio_index,
        pitchtier_index=pitchtier_index,
        sc_labels=sc_labels,
        augment=augment,
    )
    collator = PQPCollator(ds.tokenizer, ds.feature_extractor)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
