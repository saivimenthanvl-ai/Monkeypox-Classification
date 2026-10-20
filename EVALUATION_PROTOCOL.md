# Reproducibility and dataset integrity

This repository documents a VGG16 transfer-learning experiment on a **299-image four-class sample**. The reported disjoint test split has 30 images, with 27 correct (90.0%; approximate 95% Wilson interval 74–96%). This is a small-sample research result and not evidence of clinical utility.

## Implemented safeguards

- Stratified train/validation/test splitting happens before any oversampling.
- Only the training images are oversampled and augmented.
- VGG16-specific preprocessing and a stable class order are used for all splits.
- The notebook reports a classification report and confusion matrix on the held-out sample.

## Dataset audit

```bash
python audit_dataset.py . --out audit_report.json
```

The read-only SHA-256 audit flags byte-identical files, including cross-class collisions. It cannot identify patient-level overlap, near-duplicate lesions, or correlated image acquisition.

## Interpretation

The previous 95.76% result was affected by pre-split oversampling and is not a reliable measurement. The only supported outcome in the current repository is the corrected 299-image sample experiment. The separate 770-image source count describes dataset provenance, not an evaluated performance claim.
