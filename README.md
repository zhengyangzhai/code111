# CDD-Net for Chinese Pragmatic Intention Recognition

This repository contains the research code for **CDD-Net (Consistency-Discrepancy Decoupling Network)**, a multimodal framework for Chinese pragmatic spoken language understanding.

The project supports two related tasks:

- **PQP**: Pragmatic Question Pair classification (literal vs. deep)
- **SR**: Speech Act Recognition with 14 fine-grained response intention labels

The codebase includes training scripts, data loaders, model implementations, evaluation utilities, and preprocessing helpers.

## Highlights

- Cross-modal consistency/discrepancy modeling with token-level discrepancy localization
- Support for both PQP and SR tasks
- Multiple baselines and ablations, including `cdd`, `drbf`, `pbcf`, `mult`, `misa`, and CCMT
- Local training runs are written to `output/` during experiments

## Repository Structure

```text
.
|-- run.py                         # PQP training entrypoint
|-- run_sr.py                      # SR training entrypoint
|-- run_sr_ccmt.py                 # CCMT baseline for SR
|-- run_contradiction.py           # Intent contradiction baseline
|-- eval_checkpoint.py             # Evaluate saved SR checkpoints
|-- ensemble_eval.py               # Ensemble evaluation on saved runs
|-- run_repeat_experiments.py      # Multi-seed experiment launcher
|-- entrain.py                     # PQP training loop
|-- entrain_sr.py                  # SR training loop
|-- entrain_sr_ccmt.py             # CCMT training loop
|-- entrain_contradiction.py       # Contradiction baseline training loop
|-- model.py                       # Main CDD-Net / baseline model definitions
|-- model_sr_ccmt.py               # CCMT model
|-- model_contradiction.py         # Contradiction baseline model
|-- dataloader.py                  # PQP data loader
|-- sr_dataloader.py               # SR data loader
|-- preprocess_speechcraft.py      # SpeechCraft preprocessing helper
|-- make_paper_figures_*.py        # Figure generation scripts
|-- test_ccmt_smoke.py             # Small smoke test
|-- data/
|   |-- PQP/                       # PQP splits and labels
|   `-- SR/                        # SR audio and annotation files
|-- output/                        # Experiment outputs, logs, checkpoints
`-- SpeechCraft-master/            # Third-party dependency snapshot
```

## Environment

The main code uses:

- Python
- PyTorch
- NumPy
- Hugging Face `transformers`
- `tqdm`
- `matplotlib` (for figure scripts)

Pretrained backbones used by default:

- `bert-base-chinese`
- `TencentGameMate/chinese-wav2vec2-base`

Minimal installation example:

```bash
pip install -r requirements.txt
```

This repository was developed for GPU-based training. CUDA-enabled PyTorch is recommended for reproducing the reported results.

## Expected Data Layout

### PQP

The default PQP layout is:

```text
data/PQP/
|-- in-scope/
|   |-- train.tsv
|   |-- dev.tsv
|   `-- test.tsv
`-- sc_labels.json
```

The full PQP dataset is **not included** in this repository. The release repository only expects the directory structure above. The dataset link is provided in the anonymous supplementary materials.

### SR

The SR training script expects speech and annotation files under:

```text
data/SR/
|-- <sample>.wav
|-- <sample>.TextGrid
`-- <sample>.PitchTier
```

Each SR sample should provide the same basename across the three files above.
`sr_dataloader.py` scans `data/SR/` for `.TextGrid` files and then loads the
matching `.wav` and `.PitchTier` files with the same basename.

The full SR dataset is **not included** in this repository. For submission support, the dataset has been prepared in a separate anonymous repository / supplementary release. Please refer to the anonymous supplementary materials for the dataset access link.

Note: the default SR setup also uses `data/PQP/` as contextual input. If you run
`run_sr.py` with default arguments, both `data/SR/` and `data/PQP/` should be
available.

## Training

### 1. PQP Training

Default multimodal entrypoint:

```bash
python run.py --modality multimodal --exp_name pqp_multimodal
```

Representative multimodal baselines and ablations:

```bash
# multimodal baselines
python run.py --modality multimodal --exp_name pqp_multimodal
python run.py --modality text_only
python run.py --modality audio_only
python run.py --modality drbf
python run.py --modality pbcf

# multimodal feature ablations
python run.py --modality multimodal --no_prosody --exp_name pqp_multimodal_no_f0
python run.py --modality multimodal --no_frame_acoustic --exp_name pqp_multimodal_no_fa
```

### 2. SR Training

Minimal CDD entrypoint:

```bash
python run_sr.py --modality cdd --exp_name sr_cdd
```

Recommended default CDD setting for reproducing the paper's main SR result:

```bash
python run_sr.py \
  --modality cdd \
  --swa_start_epoch 28 \
  --rfr_gate_tau 1.8 \
  --rfr_beta_init 1.0 \
  --exp_name sr_B_align_seed73 \
  --seed 73
```

Full reproducible command for the reported 75.18% SR result
(`output/sr_B_align_seed73_20260323_002040`):

```bash
python run_sr.py \
  --modality cdd \
  --rfr_gate_tau 1.8 \
  --rfr_beta_init 1.0 \
  --swa_start_epoch 28 \
  --epochs 50 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --encoder_lr 1e-5 \
  --classifier_lr 5e-4 \
  --dropout 0.3 \
  --lambda_con 0.01 \
  --lambda_orth 0.001 \
  --lambda_align 0.005 \
  --lambda_sep 0.001 \
  --lambda_recon 0.01 \
  --mixup_alpha 0.3 \
  --focal_gamma 2.0 \
  --rdrop_alpha 1.0 \
  --augment_train \
  --exp_name sr_B_align_seed73 \
  --seed 73
```

One-line version:

```bash
python run_sr.py --modality cdd --rfr_gate_tau 1.8 --rfr_beta_init 1.0 --swa_start_epoch 28 --epochs 50 --batch_size 4 --gradient_accumulation_steps 8 --encoder_lr 1e-5 --classifier_lr 5e-4 --dropout 0.3 --lambda_con 0.01 --lambda_orth 0.001 --lambda_align 0.005 --lambda_sep 0.001 --lambda_recon 0.01 --mixup_alpha 0.3 --focal_gamma 2.0 --rdrop_alpha 1.0 --augment_train --exp_name sr_B_align_seed73 --seed 73
```

Representative SR ablations and baselines:

```bash
# text-only baseline
python run_sr.py --modality text_only --exp_name sr_text_only

# audio-only baseline
python run_sr.py --modality audio_only --exp_name sr_audio_only

# reproduced external baselines
python run_sr.py --modality mult --exp_name sr_mult
python run_sr.py --modality misa --exp_name sr_misa
python run_sr_ccmt.py --exp_name sr_ccmt_seed42 --seed 42
python run_sr_ccmt.py --exp_name sr_ccmt_seed73 --seed 73

# multimodal ablations on CDD-Net
python run_sr.py --modality cdd --no_prosody --exp_name sr_no_prosody
python run_sr.py --modality cdd --no_frame_acoustic --exp_name sr_no_frame_acoustic
python run_sr.py --modality cdd --no_token_disc --exp_name sr_no_tldl
python run_sr.py --modality cdd --no_dual_contrastive --exp_name sr_no_dscl
python run_sr.py --modality cdd --no_dgcp --exp_name sr_no_dgcp
python run_sr.py --modality cdd --no_rfr --exp_name sr_no_rfr
```

For the calibrated SR setting reported in the paper, it is recommended to keep
`--swa_start_epoch 28`, `--rfr_gate_tau 1.8`, and `--rfr_beta_init 1.0`
fixed across seeds, and vary only `--seed` and `--exp_name`.

## Evaluation

To evaluate an interrupted or completed SR run from saved checkpoints:

```bash
python eval_checkpoint.py output/<run_dir>
```

To evaluate an ensemble over saved runs:

```bash
python ensemble_eval.py
```

## Release Checklist

If you are preparing this repository as paper support material, the recommended upload set is:

### Keep

- `run.py`, `run_sr.py`, `run_sr_ccmt.py`, `run_contradiction.py`
- `entrain*.py`
- `model*.py`
- `dataloader.py`, `sr_dataloader.py`
- `eval_checkpoint.py`, `ensemble_eval.py`, `run_repeat_experiments.py`
- `preprocess_speechcraft.py`
- `make_paper_figures_*.py`
- `test_ccmt_smoke.py`
- `requirements.txt`
- `README.md`
- `LICENSE` (recommended to add later for the public release)

### Optional

- `sample-sigconf.tex`
- paper-related markdown notes
- a small toy example or sample subset of the dataset

### Do Not Upload

- `output/` (logs, checkpoints, summaries)
- large model checkpoints such as `*.pt`
- local logs such as `debug.log`, `run_output.log`
- duplicate archives such as `SpeechCraft-master.zip`
- temporary caches such as `__pycache__/`
- personal screenshots / chat images / local scratch files
- local diagnosis / one-off debug scripts with personal paths
- full private dataset, if it is not cleared for release

## Suggested Final Packaging

Before uploading to GitHub, a clean paper-support repository should ideally contain:

1. Source code
2. Training / evaluation scripts
3. A concise README
4. A license file
5. A `.gitignore`
6. Either:
   - a public dataset link, or
   - a separate supplementary / anonymous dataset repository, or
   - a clearly documented sample subset plus instructions for obtaining the full data

## Citation

If you release this repository publicly, add the final paper citation here after acceptance.
