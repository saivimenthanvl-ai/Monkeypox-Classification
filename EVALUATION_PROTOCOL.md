# Full 770-image evaluation protocol

The current notebook's **90.00% test accuracy (27/30)** is from a 299-image sample, not the full 770-image Kaggle collection. The historical 95.76% result was affected by oversampling before splitting and must not be reused.

## Before running

On Kaggle attach the exact `Monkeypox Skin Image Dataset` with four folders: `Chickenpox`, `Measles`, `Monkeypox`, `Normal`. Confirm the expected class counts: 107, 91, 279, 293 (770 total). The notebook automatically points at its Kaggle directory if present; confirm `DATA_DIR` and the printed counts before training.

Audit the dataset:

```bash
python audit_dataset.py "/path/to/Monkeypox Skin Image Dataset" --out audit_report.json
```

This finds byte-identical images but **cannot verify patient-level independence** or detect all perceptual duplicates.

## Full run checklist

1. Freeze an original-image stratified split with seed 42 **before** any rebalancing.
2. Check that no path appears in more than one split and investigate hash/perceptual duplicates and patient identity where available.
3. Apply oversampling and random augmentations to **training only**.
4. Fit the documented VGG16 classifier on the full 770 images using the notebook and record software versions.
5. Evaluate untouched validation and test sets; save class mapping, counts, confusion matrix and classification report.
6. Report test accuracy with correct numerator/denominator, per-class recall, macro-F1 and a confidence interval.
7. Repeat with different seeds or cross-validation for a stability estimate. Do not compare two models using the test set for hyperparameter selection.

No 770-image result is claimed until the full notebook actually runs and its outputs are archived. This is a research classifier, not a validated medical diagnostic system.
