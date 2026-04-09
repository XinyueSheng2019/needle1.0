# NEEDLE2.0

## Summary

- Added a full end-to-end training workflow in `needle_train/` (dataset build, preprocessing, training, evaluation).
- Added image and light-curve processing pipelines in `image/` and `light_curve/`.
- Added curated metadata/resources in `info/` for scaling and label management.
- Improved project docs and metadata (`README.md`, badges, MIT `LICENSE`).
- Updated `.gitignore` to exclude large generated artifacts and keep commits lightweight.

## Detailed Changelog

### Highlights

- Expanded from base project scaffolding to a modular research pipeline for transient classification.
- Unified workflow from raw inputs to model training outputs.

### Added

#### Training pipeline (`needle_train/`)

- Data construction and splitting (`get_train_valid_sets.py`, `build_data.py`)
- Preprocessing and augmentation (`preprocessing.py`, `augmentor_pipeline.py`)
- Model training/inference tooling (`run_model.py`, `training.py`, `transient_model.py`)
- Evaluation and precision-focused scripts (`evaluate_model_post_training.py`, `precision_optimized_predict.py`)

#### Image pipeline (`image/`)

- Preprocessing, masking/restoration, demo notebooks, and quality-control helpers.

#### Light-curve pipeline (`light_curve/`)

- GP fitting and upsampling workflow for photometric feature generation.

#### Reference metadata (`info/`)

- Scaling metadata, train/validation object lists, and label-support files.

### Documentation and Project Metadata

- Reworked `README.md` with quick start, input/output expectations, badges, and license section.
- Added MIT `LICENSE`.
- Updated `.gitignore` to reduce accidental commits of generated outputs.

### Upgrade Notes (from NEEDLE1.0 baseline)

- Configure paths in `config.py` before execution.
- Recommended run order:
  1. `python needle_train/get_train_valid_sets.py`
  2. `python needle_train/run_model.py`
