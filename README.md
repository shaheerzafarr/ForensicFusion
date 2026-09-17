# ForensicFusion

This is a clean Kaggle-first pipeline for binary real-vs-generated image detection.
It selects exactly **100,000 real** and **100,000 fake** images for training, then
selects additional images for validation and testing. Fake selection is balanced
across every generator that has fake examples, and content duplicates are removed
before deterministic **70% train / 20% validation / 10% test** allocation.

Exact split sizes:

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 100,000 | 100,000 | 200,000 |
| Validation | 28,571 | 28,571 | 57,142 |
| Test | 14,286 | 14,286 | 28,572 |
| **Total** | **142,857** | **142,857** | **285,714** |

## Kaggle usage

Add the Artifact dataset and this repository to a Kaggle notebook, enable a GPU,
then run:

```bash
pip install -r requirements.txt
python prepare_data.py --config config_kaggle.yaml
python train.py --config config_kaggle.yaml
python evaluate.py --config config_kaggle.yaml
```

`train.py` never evaluates on the test split. It chooses the checkpoint and decision
threshold using validation data only. `evaluate.py` loads that frozen checkpoint and
runs the final test once.

The configured dataset path is retained. The loader also detects Kaggle's usual
`/kaggle/input/artifact-dataset` mount automatically.

## Design

- Deterministic, capped balanced sampling prevents StyleGAN2 or another large source
  from dominating the fake class.
- Byte-level hashes are computed before splitting, so identical files cannot leak
  between training, validation, and test.
- Every sufficiently populated generator is represented in every split.
- A pretrained ConvNeXt-Tiny semantic stream is fused with a fixed high-pass residual
  stream that targets generation artifacts.
- Mild JPEG, resize, color, blur, and flip augmentation is applied equally to both
  classes to improve robustness without changing class balance.
- Checkpoint selection uses validation PR-AUC. Precision, recall, F1, ROC-AUC,
  PR-AUC, confusion counts, and accuracy are reported.

No pipeline can promise a particular accuracy on unseen data. This setup is designed
to produce honest, reproducible metrics and strong within-generator performance;
real-world performance should also be checked on a separate, externally sourced set.

## Commands

Rebuild the manifest:

```bash
python prepare_data.py --config config_kaggle.yaml --force
```

Resume training from `latest.pt`:

```bash
python train.py --config config_kaggle.yaml --resume
```

Use `--no-pretrained` only if Kaggle internet is disabled and pretrained weights are
not already cached. Accuracy will usually be lower.
